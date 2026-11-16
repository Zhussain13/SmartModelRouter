"""Step 1: create the training dataset.

For each question:
  1. get an answer from the SMALL model, with its self-reported confidence
  2. get an answer from the LARGE model
  3. use the large model as a JUDGE to mark whether each answer is correct
  4. save everything. needs_large = 1 means the small model got it wrong.

Safe to stop and re-run: it resumes where it left off.

    python build_labels.py --limit 5        # try a few questions first
    python build_labels.py                  # all of data/queries.csv
    python build_labels.py --demo           # simulated, takes a second
"""
import argparse
import os

import pandas as pd
from tqdm import tqdm

import config
from models import ask_large, ask_small_with_confidence, judge

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="data/queries.csv")
ap.add_argument("--output", default=config.LABELS_PATH)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--demo", action="store_true", help="use the built-in simulator")
args = ap.parse_args()

df = pd.read_csv(args.input)
if args.limit:
    df = df.head(args.limit)

# Resume support: skip questions already done
done = set()
rows = []
if os.path.exists(args.output):
    old = pd.read_csv(args.output)
    rows = old.to_dict("records")
    done = set(old["query"])
    print(f"Resuming: {len(done)} questions already labelled in {args.output}")


def save():
    pd.DataFrame(rows).to_csv(args.output, index=False)


new = 0
try:
    for _, r in tqdm(df.iterrows(), total=len(df), disable=config.DEMO):
        q = str(r["query"])
        if q in done:
            continue
        ref = r.get("reference_answer")
        ref = ref if isinstance(ref, str) and ref.strip() else None
        try:
            s, conf = ask_small_with_confidence(q)
            l = ask_large(q)
            # No reference? Use the large model's answer as the reference.
            small_ok = judge(q, s.text, ref or l.text)
            large_ok = judge(q, l.text, ref) if ref else True
        except Exception as e:
            print(f"\nSkipped (error: {e}) -> {q[:60]}")
            continue

        rows.append(dict(
            query=q, small_answer=s.text, large_answer=l.text,
            small_conf=conf, small_ok=int(small_ok), large_ok=int(large_ok),
            small_cost=s.cost_usd, large_cost=l.cost_usd,
            small_latency=s.latency_s, large_latency=l.latency_s,
            needs_large=int(not small_ok),
            has_reference=int(ref is not None),
        ))
        done.add(q)
        new += 1
        if new % 10 == 0:                         # checkpoint
            save()
finally:
    if rows:                                      # also saves on Ctrl+C
        save()

if not rows:
    raise SystemExit("No rows labelled. Run check_setup.py to find the problem.")

out = pd.DataFrame(rows)
print(f"Saved {len(out)} rows to {args.output}")
print(f"Small model correct: {out.small_ok.mean():.1%} | "
      f"Large model correct: {out.large_ok.mean():.1%} | "
      f"Needs large: {out.needs_large.mean():.1%}")
