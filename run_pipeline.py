from pathlib import Path
import sys

# ---------------------------
# 0) Figure out paths
# ---------------------------
ROOT = Path(__file__).resolve().parent          # NFLPLAYCALLER/
SRC_DIR = ROOT / "src"
MODELS_DIR = ROOT / "models"

print(">>> Project root:", ROOT)
print(">>> src dir:", SRC_DIR)
print(">>> models dir:", MODELS_DIR)

# Make sure src/ is on sys.path so `nfl_run_pass` can be imported
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    print(">>> Added to sys.path:", SRC_DIR)

# ---------------------------
# 1) Import the pipeline
# ---------------------------
print(">>> Importing run_training_pipeline...")
from nfl_run_pass.pipeline import run_training_pipeline

# ---------------------------
# 2) Run the pipeline
# ---------------------------
if __name__ == "__main__":
    print(">>> About to run training pipeline...")
    try:
        results = run_training_pipeline(
            artifacts_dir=MODELS_DIR,     # save into models/
            save_artifacts_flag=True,
        )
    except Exception as e:
        print(">>> ERROR while running pipeline:", repr(e))
        raise

    print(">>> Training complete.")
    print(">>> Metrics:", results["metrics"])
    print(">>> Artifacts saved to:")
    if results["artifact_paths"] is not None:
        for name, path in results["artifact_paths"].items():
            print(f"    {name}: {path}")
    else:
        print("    (no artifact_paths returned)")
