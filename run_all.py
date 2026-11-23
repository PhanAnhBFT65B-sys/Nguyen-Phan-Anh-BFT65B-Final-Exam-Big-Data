"""Run the whole pipeline end to end and keep a log of every step.

    python run_all.py              # full pipeline (download ~25-60 min the first time, resumable)
    python run_all.py --skip-spark # without the two PySpark steps (e.g. no Java on this laptop)
    python run_all.py --from s04   # restart from a given step

Logs are written to outputs/logs/<step>.log (committed, so graders see the outputs).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["s01_download", "s02_storage_benchmark", "s03_data_quality", "s04_build_panel",
         "s05_benchmark", "s06_spark_demo", "s07_features", "s08_model", "s09_spark_mllib", "s10_summary", "s11_report_figures"]
SPARK = {"s06_spark_demo", "s09_spark_mllib"}

ap = argparse.ArgumentParser()
ap.add_argument("--skip-spark", action="store_true")
ap.add_argument("--from", dest="start", default="s01")
args = ap.parse_args()
logs = ROOT / "outputs" / "logs"
logs.mkdir(parents=True, exist_ok=True)
started = False
for step in STEPS:
    started = started or step.startswith(args.start)
    if not started or (args.skip_spark and step in SPARK):
        continue
    print(f"=== {step} ===", flush=True)
    t0 = time.time()
    with open(logs / f"{step}.log", "w", encoding="utf-8") as fh:
        p = subprocess.run([sys.executable, "-W", "ignore", f"code/{step}.py"], cwd=ROOT,
                           stdout=fh, stderr=subprocess.STDOUT)
    print(f"    exit {p.returncode} in {time.time() - t0:,.0f}s  (log: outputs/logs/{step}.log)", flush=True)
    if p.returncode != 0:
        sys.exit(f"Step {step} failed - read the log above.")
print("All steps finished.")
