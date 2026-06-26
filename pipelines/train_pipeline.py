"""
Adverse Event Seriousness Classification Pipeline
Client: Nova Pharm, USA | Domain: Healthcare / Clinical Safety
"""
 
import os
import json
import warnings
import numpy as np
import pandas as pd
import sqlite3
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
 
from datetime import datetime
from pathlib import Path
 
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score,
    recall_score
)
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
 
from xgboost import XGBClassifier
 
warnings.filterwarnings("ignore")
 
# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_PATH      = BASE_DIR / "data" / "pharmacovigilance_cases.csv"
DB_PATH        = BASE_DIR / "data" / "pv_cases.db"
REPORTS_DIR    = BASE_DIR / "reports"
MODELS_DIR     = BASE_DIR / "models"
MLRUNS_DIR = BASE_DIR / "models" / "mlruns"
MLFLOW_URI = (BASE_DIR / "models" / "mlruns").resolve().as_uri()
 
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True) 
 
# ──────────────────────────────────────────────────────────────────────────────
# 1. SQL LAYER — load CSV → SQLite, query structured features
# ──────────────────────────────────────────────────────────────────────────────
def build_sqlite_db(csv_path: Path, db_path: Path) -> None:
    """Persist raw CSV into SQLite for SQL-based feature engineering."""
    print("  [SQL] Writing to SQLite …")
    df = pd.read_csv(csv_path)
    con = sqlite3.connect(db_path)
    df.to_sql("cases", con, if_exists="replace", index=False)
    # Index hot-path columns
    con.execute("CREATE INDEX IF NOT EXISTS idx_serious ON cases(serious)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ae      ON cases(adverse_event)")
    con.commit()
    con.close()
    print(f"  [SQL] DB ready  →  {db_path}  ({len(df):,} rows)")
 
 
def load_structured_features_from_sql(db_path: Path) -> pd.DataFrame:
    """
    Mimic a real-world pattern where structured case data lives in a
    relational store (EDC / safety DB) and is joined at training time.
    """
    print("  [SQL] Querying structured features …")
    con = sqlite3.connect(db_path)
    query = """
        SELECT
            case_id,
            patient_id,
            drug_code,
            adverse_event,
            severity,
            priority_score,
            reporter_type,
            report_date,
            narrative,
            serious
        FROM cases
    """
    df = pd.read_sql_query(query, con)
    con.close()
    return df
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
SEVERITY_MAP = {"mild": 0, "moderate": 1, "severe": 2, "life-threatening": 3}
 
AE_RISK_MAP = {
    "anaphylaxis":    4,
    "arrhythmia":     3,
    "hepatotoxicity": 3,
    "rash":           1,
    "headache":       1,
    "nausea":         1,
}
 
 
def engineer_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create numeric/categorical features from structured columns."""
    df = df.copy()
 
    # Severity ordinal
    df["severity_num"] = df["severity"].map(SEVERITY_MAP).fillna(1).astype(int)
 
    # AE clinical risk score
    df["ae_risk"] = df["adverse_event"].map(AE_RISK_MAP).fillna(2).astype(int)
 
    # Reporter type one-hot (4 levels)
    reporter_dummies = pd.get_dummies(df["reporter_type"], prefix="rep")
    df = pd.concat([df, reporter_dummies], axis=1)
 
    # Report recency (days since oldest record)
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    ref = df["report_date"].min()
    df["days_since_first"] = (df["report_date"] - ref).dt.days.fillna(0).astype(int)
 
    # Narrative length (proxy for documentation quality)
    df["narrative_len"] = df["narrative"].fillna("").str.len()
 
    return df
 
 
def build_hybrid_feature_matrix(df: pd.DataFrame, tfidf_unigram, tfidf_bigram):
    """
    Combine:
      • TF-IDF unigrams  (narrative)
      • TF-IDF bigrams   (narrative)
      • Structured numeric/categorical features
    """
    narratives = df["narrative"].fillna("").tolist()
 
    X_tfidf_1 = tfidf_unigram.transform(narratives)
    X_tfidf_2 = tfidf_bigram.transform(narratives)
 
    struct_cols = (
        ["severity_num", "ae_risk", "priority_score",
         "days_since_first", "narrative_len"]
        + [c for c in df.columns if c.startswith("rep_")]
    )
    X_struct = csr_matrix(df[struct_cols].astype(float).values)
 
    return hstack([X_tfidf_1, X_tfidf_2, X_struct])
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 3. THRESHOLD OPTIMISATION (maximise recall on the serious class)
# ──────────────────────────────────────────────────────────────────────────────
def optimise_threshold(y_true, y_proba, min_recall: float = 0.90) -> float:
    """
    Walk the precision-recall curve and pick the highest threshold that
    still satisfies min_recall.  Falls back to 0.5 if constraint can't
    be met.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds has len = len(precisions) - 1
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= min_recall:
            return float(t)
    return 0.5
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 4. SHAP EXPLAINABILITY REPORT
# ──────────────────────────────────────────────────────────────────────────────
def generate_shap_report(
    model,
    X_sample,
    feature_names,
    out_path: Path,
    top_n: int = 20,
) -> None:
    """
    Produce a SHAP bar chart of the top-N features and save a JSON
    summary aligned with FDA documentation expectations.
    """
    print("  [SHAP] Computing SHAP values (sample) …")
    explainer = shap.TreeExplainer(model)
    # Convert to dense for SHAP if needed
    X_dense = X_sample.toarray() if hasattr(X_sample, "toarray") else X_sample
    shap_values = explainer.shap_values(X_dense)
 
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]
 
    top_features = [feature_names[i] for i in top_idx]
    top_values   = [float(mean_abs[i]) for i in top_idx]
 
    # ── Bar chart ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#d62728" if v > np.median(top_values) else "#1f77b4"
              for v in top_values]
    ax.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(
        f"Top {top_n} Predictive Features – Adverse Event Seriousness\n"
        "(FDA SHAP Explainability Report)",
        fontsize=13, fontweight="bold",
    )
    ax.axvline(np.median(top_values), color="grey", linestyle="--",
               linewidth=0.8, label="Median importance")
    ax.legend(fontsize=9)
    plt.tight_layout()
    chart_path = out_path / "shap_feature_importance.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
 
    # ── JSON summary ───────────────────────────────────────────────────────
    json_path = out_path / "shap_summary.json"
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "client": "Nova Pharm, USA",
        "model": "XGBClassifier",
        "top_features": [
            {"rank": i + 1, "feature": f, "mean_abs_shap": round(v, 6)}
            for i, (f, v) in enumerate(zip(top_features, top_values))
        ],
    }
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
 
    print(f"  [SHAP] Chart  → {chart_path}")
    print(f"  [SHAP] JSON   → {json_path}")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline() -> dict:
    print("\n" + "=" * 70)
    print("  Nova Pharm – Adverse Event Seriousness Classifier")
    print("=" * 70)
 
    # ── 5.1 Data ingestion ────────────────────────────────────────────────
    print("\n[1/7] Data ingestion …")
    build_sqlite_db(DATA_PATH, DB_PATH)
    df = load_structured_features_from_sql(DB_PATH)
    print(f"      Loaded {len(df):,} cases  |  Serious: {df['serious'].mean()*100:.1f}%")
 
    # ── 5.2 Feature engineering ───────────────────────────────────────────
    print("\n[2/7] Feature engineering …")
    df = engineer_structured_features(df)
 
    # TF-IDF vectorisers (fit on full corpus; train/test split done inside CV)
    print("      Fitting TF-IDF vectorisers …")
    narratives = df["narrative"].fillna("").tolist()
 
    tfidf_unigram = TfidfVectorizer(
        max_features=5_000,
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=3,
    )
    tfidf_bigram = TfidfVectorizer(
        max_features=3_000,
        ngram_range=(2, 2),
        sublinear_tf=True,
        min_df=5,
    )
    tfidf_unigram.fit(narratives)
    tfidf_bigram.fit(narratives)
 
    # Feature names (for SHAP labels)
    struct_cols = (
        ["severity_num", "ae_risk", "priority_score",
         "days_since_first", "narrative_len"]
        + [c for c in df.columns if c.startswith("rep_")]
    )
    feature_names = (
        list(tfidf_unigram.get_feature_names_out())
        + list(tfidf_bigram.get_feature_names_out())
        + struct_cols
    )
 
    X = build_hybrid_feature_matrix(df, tfidf_unigram, tfidf_bigram)
    y = df["serious"].values
    print(f"      Feature matrix shape: {X.shape}")
 
    # ── 5.3 Stratified cross-validation ──────────────────────────────────
    print("\n[3/7] Stratified 5-fold cross-validation …")
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
    # Scale of positives vs negatives (≈1:1 here, but kept for generality)
    scale_pos = (y == 0).sum() / (y == 1).sum()
 
    xgb_params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.6,
        scale_pos_weight=scale_pos,   # class-weight tuning
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    xgb_clf = XGBClassifier(**xgb_params)
    
    cv_roc  = cross_val_score(xgb_clf, X, y, cv=skf, scoring="roc_auc",   n_jobs=1)
    cv_rec  = cross_val_score(xgb_clf, X, y, cv=skf, scoring="recall",    n_jobs=1)
    cv_f1   = cross_val_score(xgb_clf, X, y, cv=skf, scoring="f1",        n_jobs=1)
 
    print(f"      ROC-AUC : {cv_roc.mean():.4f}  (±{cv_roc.std():.4f})")
    print(f"      Recall  : {cv_rec.mean():.4f}  (±{cv_rec.std():.4f})")
    print(f"      F1      : {cv_f1.mean():.4f}  (±{cv_f1.std():.4f})")
 
    # ── 5.4 Final model fit + threshold optimisation ──────────────────────
    print("\n[4/7] Fitting final model + optimising threshold …")
 
    # Train on 90 %, hold out 10 % for threshold search
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42
    )
    xgb_clf.fit(X_tr, y_tr)
 
    val_proba   = xgb_clf.predict_proba(X_val)[:, 1]
    best_thresh = optimise_threshold(y_val, val_proba, min_recall=0.90)
    val_pred    = (val_proba >= best_thresh).astype(int)
 
    print(f"      Optimal threshold : {best_thresh:.3f}")
    print(f"      Recall  @ threshold : {recall_score(y_val, val_pred):.4f}")
    print(f"      F1      @ threshold : {f1_score(y_val, val_pred):.4f}")
 
    # ── 5.5 Evaluation report ─────────────────────────────────────────────
    print("\n[5/7] Generating evaluation report …")
    roc_auc   = roc_auc_score(y_val, val_proba)
    avg_prec  = average_precision_score(y_val, val_proba)
    report    = classification_report(y_val, val_pred,
                                      target_names=["Non-Serious", "Serious"])
    cm        = confusion_matrix(y_val, val_pred)
 
    print(f"\n{'─'*50}")
    print(report)
    print(f"  ROC-AUC         : {roc_auc:.4f}")
    print(f"  Avg Precision   : {avg_prec:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    print(f"{'─'*50}")
 
    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Non-Serious", "Pred Serious"],
                yticklabels=["True Non-Serious", "True Serious"], ax=ax)
    ax.set_title("Confusion Matrix – Threshold-Optimised Model")
    plt.tight_layout()
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
 
    # PR curve plot
    prec_arr, rec_arr, _ = precision_recall_curve(y_val, val_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rec_arr, prec_arr, lw=2, color="#d62728")
    ax.axvline(0.90, color="grey", linestyle="--", label="Min recall = 0.90")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve  (AP = {avg_prec:.3f})")
    ax.legend(); plt.tight_layout()
    pr_path = REPORTS_DIR / "pr_curve.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
 
    # ── 5.6 SHAP explainability ───────────────────────────────────────────
    print("\n[6/7] SHAP explainability report …")
    shap_sample_size = min(500, X_val.shape[0])
    idx_sample = np.random.default_rng(0).choice(X_val.shape[0],
                                                  size=shap_sample_size,
                                                  replace=False)
    X_val_sample = X_val[idx_sample]
    generate_shap_report(xgb_clf, X_val_sample, feature_names, REPORTS_DIR)
 
    # ── 5.7 MLflow logging ────────────────────────────────────────────────
    print("\n[7/7] Logging to MLflow …")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("NovaPharm-PV-Classifier")
 
    with mlflow.start_run(run_name="xgb_hybrid_tfidf") as run:
        mlflow.log_params({
            "model":            "XGBClassifier",
            "n_estimators":     xgb_params["n_estimators"],
            "max_depth":        xgb_params["max_depth"],
            "learning_rate":    xgb_params["learning_rate"],
            "tfidf_unigram_feats": 5_000,
            "tfidf_bigram_feats":  3_000,
            "threshold":        round(best_thresh, 4),
            "min_recall_target": 0.90,
            "cv_folds":         5,
        })
        mlflow.log_metrics({
            "cv_roc_auc_mean":  round(cv_roc.mean(), 4),
            "cv_recall_mean":   round(cv_rec.mean(), 4),
            "cv_f1_mean":       round(cv_f1.mean(), 4),
            "val_roc_auc":      round(roc_auc, 4),
            "val_avg_precision": round(avg_prec, 4),
            "val_recall":       round(recall_score(y_val, val_pred), 4),
            "val_f1":           round(f1_score(y_val, val_pred), 4),
            "optimal_threshold": round(best_thresh, 4),
        })
        mlflow.log_artifacts(str(REPORTS_DIR), artifact_path="reports")
        mlflow.xgboost.log_model(xgb_model=xgb_clf,name="model")
 
        run_id = run.info.run_id
        print(f"  MLflow run ID : {run_id}")
 
    # ── Summary dict ──────────────────────────────────────────────────────
    results = {
        "run_id":         run_id,
        "cv_roc_auc":     round(cv_roc.mean(), 4),
        "cv_recall":      round(cv_rec.mean(), 4),
        "cv_f1":          round(cv_f1.mean(), 4),
        "val_roc_auc":    round(roc_auc, 4),
        "val_avg_prec":   round(avg_prec, 4),
        "val_recall":     round(recall_score(y_val, val_pred), 4),
        "val_f1":         round(f1_score(y_val, val_pred), 4),
        "threshold":      round(best_thresh, 4),
        "model":          xgb_clf,
        "tfidf_unigram":  tfidf_unigram,
        "tfidf_bigram":   tfidf_bigram,
        "feature_names":  feature_names,
    }
 
    print("\n" + "=" * 70)
    print("  Pipeline complete.")
    print("=" * 70 + "\n")
    return results
 
 
if __name__ == "__main__":
    run_pipeline()