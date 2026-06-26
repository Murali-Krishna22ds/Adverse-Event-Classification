"""
Master runner — executes training and batch scoring end-to-end.
Run:  python run_all.py
"""
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
 
from pipelines.train_pipeline import run_pipeline
from pipelines.batch_scoring  import run_batch_scoring
 
 
if __name__ == "__main__":
    model_artifacts = run_pipeline()
    run_batch_scoring(model_artifacts, n_new=5_000)
 
    print("\n✓  All artefacts written to ./reports/")
    print("   • shap_feature_importance.png")
    print("   • shap_summary.json")
    print("   • confusion_matrix.png")
    print("   • pr_curve.png")
    print("   • drift_report.json")
    print("   • batch_scored_cases.csv")
    print("   MLflow experiments → ./models/mlruns/\n")
 