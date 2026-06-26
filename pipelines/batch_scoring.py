"""
Batch Scoring Pipeline with Data Drift Detection
Simulates production scoring of new PV cases.
"""
 
import json
import numpy as np
import pandas as pd
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime
from scipy.stats import ks_2samp
from scipy.sparse import hstack, csr_matrix
 
warnings.filterwarnings("ignore")
 
BASE_DIR    = Path(__file__).resolve().parent.parent
DB_PATH     = BASE_DIR / "data" / "pv_cases.db"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
 
SEVERITY_MAP = {"mild": 0, "moderate": 1, "severe": 2, "life-threatening": 3}
AE_RISK_MAP  = {
    "anaphylaxis":    4, "arrhythmia": 3,
    "hepatotoxicity": 3, "rash":       1,
    "headache":       1, "nausea":     1,
}
 
 
def engineer_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["severity_num"]   = df["severity"].map(SEVERITY_MAP).fillna(1).astype(int)
    df["ae_risk"]        = df["adverse_event"].map(AE_RISK_MAP).fillna(2).astype(int)
    reporter_dummies     = pd.get_dummies(df["reporter_type"], prefix="rep")
    df = pd.concat([df, reporter_dummies], axis=1)
    df["report_date"]    = pd.to_datetime(df["report_date"], errors="coerce")
    df["days_since_first"] = 0  # batch scoring: use 0 as relative baseline
    df["narrative_len"]  = df["narrative"].fillna("").str.len()
    return df
 
 
def align_struct_cols(df: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    """Ensure the structured feature matrix has exactly the expected columns."""
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[expected_cols]
 
 
# def build_batch_features(df, tfidf_unigram, tfidf_bigram, struct_cols):
#     narratives  = df["narrative"].fillna("").tolist()
#     X_tfidf_1   = tfidf_unigram.transform(narratives)
#     X_tfidf_2   = tfidf_bigram.transform(narratives)
#     X_struct    = csr_matrix(align_struct_cols(df, struct_cols).values)
#     return hstack([X_tfidf_1, X_tfidf_2, X_struct])
def build_batch_features(df, tfidf_unigram, tfidf_bigram, struct_cols):
    narratives = df["narrative"].fillna("").tolist()

    X_tfidf_1 = tfidf_unigram.transform(narratives)
    X_tfidf_2 = tfidf_bigram.transform(narratives)

    aligned = align_struct_cols(df, struct_cols).copy()

    # Convert bool/object columns to numeric
    aligned = aligned.apply(pd.to_numeric, errors="coerce")
    aligned = aligned.fillna(0).astype(np.float32)

    X_struct = csr_matrix(aligned.values)

    return hstack([X_tfidf_1, X_tfidf_2, X_struct])
 
 
# ──────────────────────────────────────────────────────────────────────────────
# DRIFT DETECTION
# ──────────────────────────────────────────────────────────────────────────────
DRIFT_COLS   = ["severity_num", "ae_risk", "priority_score", "narrative_len"]
DRIFT_ALPHA  = 0.05   # KS test significance level
 
 
def detect_drift(ref_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Run Kolmogorov-Smirnov test on key numeric features.
    Returns a drift report dict.
    """
    report = {"timestamp": datetime.utcnow().isoformat() + "Z", "features": {}}
    any_drift = False
    for col in DRIFT_COLS:
        if col not in ref_df.columns or col not in new_df.columns:
            continue
        stat, p_value = ks_2samp(ref_df[col].dropna(), new_df[col].dropna())
        drifted = bool(p_value < DRIFT_ALPHA)
        if drifted:
            any_drift = True
        report["features"][col] = {
            "ks_stat":  round(float(stat), 5),
            "p_value":  round(float(p_value), 5),
            "drifted":  drifted,
        }
    report["any_drift_detected"] = any_drift
    return report
 
 
# ──────────────────────────────────────────────────────────────────────────────
# MAIN BATCH SCORING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def run_batch_scoring(model_artifacts: dict, n_new: int = 5_000) -> pd.DataFrame:
    """
    Simulate loading new cases from the DB (last n_new rows),
    score them, run drift detection vs. training distribution,
    and save a scored CSV + drift report.
    """
    print("\n" + "─" * 60)
    print("  Batch Scoring Pipeline")
    print("─" * 60)
 
    model         = model_artifacts["model"]
    tfidf_unigram = model_artifacts["tfidf_unigram"]
    tfidf_bigram  = model_artifacts["tfidf_bigram"]
    threshold     = model_artifacts["threshold"]
    feature_names = model_artifacts["feature_names"]
 
    struct_cols = (
        ["severity_num", "ae_risk", "priority_score",
         "days_since_first", "narrative_len"]
        + [f for f in feature_names
           if f not in list(tfidf_unigram.get_feature_names_out())
              + list(tfidf_bigram.get_feature_names_out())
              + ["severity_num", "ae_risk", "priority_score",
                 "days_since_first", "narrative_len"]]
    )
 
    # ── Load "new" cases from SQLite (simulate production ingest) ──────────
    print(f"  [SCORE] Loading {n_new:,} new cases from SQLite …")
    con = sqlite3.connect(DB_PATH)
    new_df = pd.read_sql_query(
        f"SELECT * FROM cases ORDER BY ROWID DESC LIMIT {n_new}", con
    )
    ref_df = pd.read_sql_query(
        f"SELECT * FROM cases ORDER BY ROWID ASC  LIMIT {n_new}", con
    )
    con.close()
 
    # ── Feature engineering ────────────────────────────────────────────────
    new_df = engineer_structured_features(new_df)
    ref_df = engineer_structured_features(ref_df)
 
    # ── Drift detection ────────────────────────────────────────────────────
    print("  [DRIFT] Running KS drift tests …")
    drift_report = detect_drift(ref_df, new_df)
    if drift_report["any_drift_detected"]:
        drifted = [k for k, v in drift_report["features"].items() if v["drifted"]]
        print(f"  [DRIFT] ⚠  Drift detected in: {drifted}")
    else:
        print("  [DRIFT] ✓  No significant drift detected.")
 
    drift_path = REPORTS_DIR / "drift_report.json"
    with open(drift_path, "w") as fh:
        json.dump(drift_report, fh, indent=2)
    print(f"  [DRIFT] Report → {drift_path}")
 
    # ── Score ──────────────────────────────────────────────────────────────
    print("  [SCORE] Scoring …")
    X_new    = build_batch_features(new_df, tfidf_unigram, tfidf_bigram, struct_cols)
    probas   = model.predict_proba(X_new)[:, 1]
    preds    = (probas >= threshold).astype(int)
 
    scored_df = new_df[["case_id", "patient_id", "adverse_event",
                         "severity", "serious"]].copy()
    scored_df["serious_prob"]   = probas.round(4)
    scored_df["serious_pred"]   = preds
    scored_df["threshold_used"] = threshold
 
    scored_path = REPORTS_DIR / "batch_scored_cases.csv"
    scored_df.to_csv(scored_path, index=False)
    print(f"  [SCORE] Scored {len(scored_df):,} cases → {scored_path}")
 
    # Quick accuracy check (label available in simulation)
    from sklearn.metrics import recall_score, f1_score
    recall = recall_score(scored_df["serious"], scored_df["serious_pred"])
    f1     = f1_score(scored_df["serious"], scored_df["serious_pred"])
    print(f"  [SCORE] Recall={recall:.4f}  F1={f1:.4f}  (vs ground truth)")
 
    return scored_df
 
 
if __name__ == "__main__":
    print("Import and call run_batch_scoring(model_artifacts) to run the batch scoring pipeline.")