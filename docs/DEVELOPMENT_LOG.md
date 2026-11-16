# Development Log

A step-by-step record of how Smart Model Router was built: what was done, how, why, and how
each part was tested. The last section is a template for logging your own experiment runs.

---

## Phase 0: Starting point

**Input:** a project guide (`smart-model-router-guide.md`) describing the problem, the three-tier
design, the tech stack, reference code for each file, and 16 sample questions.

**Requirements agreed before building:**

| Decision | Choice | Reason |
|---|---|---|
| Models | **Free only** | No paid APIs. Gemini is used on the free tier (no billing), and Ollama is local |
| Large model | Support **both** Gemini free tier and a **local Ollama** model | Gemini needs no big machine; local needs no key or internet |
| Fixes to the guide's code | Apply small correctness and speed fixes | Keep the guide's structure but fix issues found in review |
| Dataset | Write about 300 starter questions | The guide says at least 300 are needed for meaningful results |
| Kanglish questions | Separate file | Keep training English-only and test code-mixed questions separately |
| Literature survey | Written for the project | Covers routing, cascades, judges, retrieval and code-mixed NLP |
| Demo mode | Add one | Lets anyone visiting the public repo run the project without installing models |

---

## Phase 1: Core project from the guide

### 1.1 Configuration (`config.py`)

- **How:** one module holds every setting. Each value reads from an environment variable with a
  default, so nothing secret is ever written in code (`GEMINI_API_KEY` comes only from the environment).
- **Why:** students change models, prices and thresholds often, and one file keeps that simple.

### 1.2 Model access (`models.py`)

- **How:** a `Result` dataclass records text, model, tokens in/out, cost and latency for every
  call.
  - `ask_small()` calls Ollama's `/api/chat` with `stream: False` and `temperature 0.2`. Token
    counts come from `prompt_eval_count` and `eval_count`.
  - `ask_large()` calls Gemini through the `google-genai` SDK. Tokens come from
    `usage_metadata`, and **thinking tokens are added to output tokens** because they are billed as output.
  - `ask_small_with_confidence()` appends a `CONFIDENCE: N` instruction and parses the result.
  - `judge()` asks the large model for YES/NO against a reference.
- **Fixes over the guide:**
  - Gemini **rate-limit handling**: free-tier 429/503 errors are retried with growing waits
    (30 s, 60 s, …) instead of skipping the question. The default pause between calls was raised
    from 1 s to 6 s to stay under per-minute limits.
  - Wait time during retries is **not counted as latency**.
  - Confidence parsing accepts markdown (`**CONFIDENCE:** 8`), because small models often bold it.
  - A separate `JUDGE_MODEL` setting allows a model with a bigger free quota to do the grading.

### 1.3 Retrieval (`retriever.py`)

- **How:** MiniLM embeddings with `normalize_embeddings=True`, so cosine similarity is a single
  matrix-vector product. `lookup()` returns the best similarity, answer and matched question.
- **Fix:** `sentence_transformers` is imported only when first needed, so the rest of the code
  (and demo mode) works without PyTorch installed.

### 1.4 Routers (`router.py`)

- **How:** `rule_score()` exactly as in the guide. `features()` = 384-dim embedding + rule score
  + normalised length. `LearnedRouter` loads the saved logistic-regression model.
- **Fix:** added `p_hard_many()` for **batched** prediction. The guide embedded test questions one
  at a time, which is slow with a real embedding model; batching embeds them all in one call.

### 1.5 Labelling (`build_labels.py`)

- **How:** for each question, get the small answer and confidence, get the large answer, then judge
  both. Writes one row per question.
- **Fixes:**
  - The **checkpoint counter** counts only newly labelled rows. In the guide it counted all rows,
    so after resuming, checkpoints happened at the wrong times.
  - **Saves on Ctrl+C** (a `try/finally` block), so a manual stop loses nothing.
  - Added a `has_reference` column so analysis can separate human-referenced labels.
  - Exits with a clear message if nothing could be labelled.

### 1.6 Training (`train_router.py`)

- **How:** stratified 70/30 split, logistic regression with balanced class weights, then a
  classification report and ROC-AUC. Saves the model and the test split.
- **Fix:** a friendly check before splitting. With too few examples of one class,
  `train_test_split(stratify=…)` crashes with a confusing error, so the script now says
  "Need at least 3 'small OK' and 3 'needs large' examples".

### 1.7 Evaluation (`evaluate.py`)

- **How:** replays strategies on the saved test split with no model calls, as described in
  [ROUTER_DESIGN.md §5](ROUTER_DESIGN.md#5-evaluation-method).
- **Fixes:**
  - **Oracle definition.** The guide's oracle sent every question the small model got wrong to the
    large model, even when the large model was also wrong. That costs money and gains no accuracy,
    so it isn't the "perfect router". It is now `small_ok == 0 AND large_ok == 1`.
  - Batched router predictions (see 1.4).
  - Curve points are sorted by cost before plotting, so lines don't zig-zag.
  - The headline is reported for **all three** routers, not only the learned one.
  - `--test` / `--out` options allow evaluating other labelled files, such as the Kanglish set.

### 1.8 Chat app (`app.py`)

- **How:** Streamlit chat. Retrieval is checked first, then the router picks a model. Each reply
  shows its route, `p_hard`, cost and latency. A sidebar slider sets the threshold live and shows
  running totals.
- **Fix:** if a model call fails (for example, Ollama not running), the error is shown in the chat
  instead of crashing the app.

### 1.9 Data

- `data/faq.csv`: the 6 sample FAQs from the guide (to be replaced with real department FAQs).
- `data/queries.csv`: the guide's 16 examples plus hand-written questions, for **327 English
  questions** in total across C, data structures, algorithms, DBMS, OS, computer networks, OOP and
  maths. Easy ones are definitions ("What is a semaphore?"). Hard ones include code debugging,
  output tracing, proofs, numericals and design problems.
- `data/queries_kanglish.csv`: **10 Kannada/English code-mixed questions**, kept separate (see Phase 2).
- The CSV was generated with Python's `csv` module, so questions containing commas and quotes
  (code snippets) are escaped correctly.

### 1.10 Testing in Phase 1

The development environment had no Ollama, no Gemini key, and **no access to Hugging Face** (so the
embedding model could not be downloaded). Testing therefore used fakes:

- a fake Ollama response and a fake Gemini client, including one forced **429 error** to test the
  retry,
- a stub embedder (hashed word counts).

Verified: setup check, labelling of 25 questions, **resume** to the full set, training, evaluation,
graph, and the Streamlit app answering on the retrieval and small routes.

**Outcome:** merged to `main` as pull request #1.

---

## Phase 2: Kanglish split

- **What:** moved the 10 code-mixed questions from `queries.csv` into `queries_kanglish.csv`.
- **Why:** training the router on a mix of English and a handful of Kanglish questions would hide
  the interesting question: *does an English-trained router generalise to code-mixed input?*
  Keeping them separate makes that a clean experiment.
- **How to run it:** label the file with `--input/--output`, then `evaluate.py --test … --out …`.
- **Tested:** the full pipeline, then labelling and evaluation of the Kanglish file, with fake models.

---

## Phase 3: Free large-model choice, demo mode, docs and CI

### 3.1 Two free large-model providers

- **What:** `LARGE_PROVIDER = gemini | ollama`.
- **How:** the Ollama call was generalised into `_ollama(model, …)`, used for the small model and,
  when `LARGE_PROVIDER=ollama`, for the large model (`qwen2.5:14b` by default). `check_setup.py`
  gives provider-specific fix-it hints.
- **Cost:** large-model calls are always priced at an **API-equivalent** list price, so results
  are comparable whichever free provider actually ran (see ROUTER_DESIGN §6).

### 3.2 Demo mode

- **What:** `--demo` on any script, or `SMR_DEMO=1`, runs everything with a simulator.
- **How (`demo_models.py`):** a hidden difficulty per question (wording signals + deterministic
  noise from an MD5 hash). The probability of a correct small answer and the small model's
  confidence fall with difficulty. A hashing embedder replaces MiniLM. Demo answers carry a
  `[DEMO-SMALL]` / `[DEMO-LARGE]` tag so the simulated judge knows which model wrote them.
- **Separate outputs:** real runs write to `outputs/real/`, demo runs to `outputs/demo/`, so
  simulated labels can never mix with real ones.
- **Tuning:**
  - In the first version the small model was right only about 65% of the time, which is too pessimistic for a mostly-easy question set.
    The pass probability was adjusted to give about 74%.
  - The noise term was kept large enough that no router is perfect: the learned router's ROC-AUC is about 0.72 in the demo.
- **Bugs found and fixed while testing:**
  1. Streamlit's markdown changed the tag `[demo:small]` into `[demo]` (the colon syntax is
     special in Streamlit markdown). The tags were renamed to `[DEMO-SMALL]` / `[DEMO-LARGE]`.
  2. The "always large" label was cut off at the right edge of the graph. It is now right-aligned.
  3. The sidebar said "1 questions". Plurals are now correct.

### 3.3 One-command pipeline (`run_pipeline.py`)

Runs label → train → evaluate in sequence, passing `--demo` and `--limit` through. It stops with a
clear message at the first failing step.

### 3.4 Requirements split

| File | Contents | Use |
|---|---|---|
| `requirements-demo.txt` | pandas, numpy, scikit-learn, streamlit, matplotlib, … | Demo mode. Fast install, **no PyTorch** |
| `requirements.txt` | Demo file + `sentence-transformers` + `google-genai` | Real models |
| `requirements-dev.txt` | Demo file + `google-genai` + `pytest` | Tests and CI |

A fresh virtual environment with only `requirements-demo.txt` was created and the full demo
pipeline was run in it to confirm the light install works.

### 3.5 Tests (`tests/`) and CI

12 pytest tests, all passing:

| Test file | What it checks |
|---|---|
| `test_router.py` | Rule scores for easy, code, long and capped questions; batch = single prediction |
| `test_models.py` | Confidence parsing (last value, markdown, missing, clamping); small model is free; local-Ollama large model is priced correctly; Gemini retries after a 429 and counts thinking tokens; judge parses YES; demo mode needs no models |
| `test_pipeline_demo.py` | Copies the project to a temp folder, runs `run_pipeline.py --demo`, and checks all 5 output files exist |

GitHub Actions (`.github/workflows/ci.yml`) runs the tests and the demo pipeline on Python 3.10 and
3.12 for every push to `main` and every pull request.

### 3.6 Documentation

- `README.md`: rewritten for a public audience, with badges, a screenshot, a mermaid diagram, a
  2-minute demo quick start, full setup for both free providers, demo results (clearly labelled as
  simulated), configuration and troubleshooting.
- `docs/LITERATURE_SURVEY.md`: 23 references across cascades, routers, benchmarks, confidence,
  LLM-as-judge, retrieval, small open models and code-mixed NLP, with a comparison table and the
  research gap.
- `docs/ROUTER_DESIGN.md`: how every component works, the features, label definition, evaluation
  method, cost model and limitations.
- `docs/images/`: the demo app screenshot (taken with headless Chromium) and the demo
  cost-vs-accuracy graph.
- `LICENSE`: MIT.

### 3.7 Demo results (simulated)

`python run_pipeline.py --demo` on 327 questions (99 in the held-out test set):

| Strategy | Accuracy | Cost / 1k | % large |
|---|---|---|---|
| Always small | 73.7% | $0.00 | 0% |
| Always large | 100% | $0.98 | 100% |
| Oracle | 100% | $0.33 | 26% |
| Rules @0.5 | 80.8% | $0.22 | 14% |
| Learned @0.5 | 91.9% | $0.60 | 45% |
| Cascade conf<7 | 97.0% | $0.56 | 41% |

The cheapest setting that keeps 95% of large-model accuracy was the cascade at **57%** of the cost,
the learned router at **75%**, and the rule router only at 100%. **These numbers are simulated** and
only prove that the pipeline works end to end.

---

## Phase 4: Real experiment run log (fill this in)

Copy this block for each real run so the report can cite exact settings.

```text
Run ID:            R1
Date:
Machine:           (CPU / RAM / GPU)
Small model:       qwen2.5:3b
Large provider:    gemini | ollama
Large model:
Judge model:
Questions labelled: ___ / 327   (Kanglish: ___ / 10)
Time taken:
Small correct:     ___ %    Large correct: ___ %    Needs large: ___ %
Router ROC-AUC:
Headline:          learned router reaches 95% of large accuracy at ___ % of its cost
                   cascade: ___ %     rules: ___ %
Kanglish result:
Problems hit / notes:
```

### Checklist before reporting real results

- [ ] Replace `data/faq.csv` with real department FAQs
- [ ] Add real student questions to `data/queries.csv` (a Google Form works well)
- [ ] Add reference answers for at least some questions (reduces judge bias)
- [ ] Spot-check about 30 judge labels by hand and report the agreement
- [ ] Run the full pipeline and save `outputs/real/results.csv` and `results.png`
- [ ] Run the Kanglish evaluation
- [ ] Fill in the run log above
