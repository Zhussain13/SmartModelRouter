"""Runs the whole pipeline in one go: label -> train -> evaluate.

    python run_pipeline.py --demo          # simulated, finishes in seconds
    python run_pipeline.py --limit 20      # real models, first 20 questions
    python run_pipeline.py                 # real models, all questions
"""
import argparse
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true", help="use the built-in simulator")
ap.add_argument("--limit", type=int, default=None, help="label only the first N questions")
args = ap.parse_args()

extra = ["--demo"] if args.demo else []
steps = [
    ["build_labels.py"] + (["--limit", str(args.limit)] if args.limit else []),
    ["train_router.py"],
    ["evaluate.py"],
]
for i, step in enumerate(steps, 1):
    print(f"\n========== Step {i}/3: python {' '.join(step + extra)} ==========")
    code = subprocess.call([sys.executable] + step + extra)
    if code != 0:
        sys.exit(f"Step {i} failed (exit code {code}). Fix the error above and run again; "
                 "labelling resumes where it stopped.")
print("\nDone. Now try the chat app:  streamlit run app.py" + (" -- --demo" if args.demo else ""))
