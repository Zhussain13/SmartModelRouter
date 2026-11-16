# Smart Model Router

*Send each question to the cheapest AI that can still answer it correctly.*

A student-help chatbot that decides, **for every question**, whether to answer from
stored FAQs, a **free small local model** (Qwen 2.5 3B via Ollama), or a **large model**
(Google Gemini, used on its **free tier**). We then measure how close each routing
strategy gets to large-model accuracy and how much it would cost.

> **No paid services.** Ollama runs locally and is free. Gemini is used through a free
> Google AI Studio key **without billing enabled**, so you can never be charged.
> The "cost" numbers in the results are calculated at Gemini's *list price*. They show what
> each strategy *would* cost in production, which is what the project measures.

---

## 1. The problem

| Option | Quality | Cost | Speed |
|---|---|---|---|
| **Large API model** (Gemini, GPT, Claude) | Excellent | You pay for every token | Depends on internet |
| **Small local model** (Qwen 3B, Llama 3B on your own PC) | Good on easy questions, weak on hard ones | Free | Fast |

- Send **every** question to the large model and you pay too much. Most student questions
  are easy ("What is a pointer?"), and a free small model answers them just as well.
- Send **every** question to the small model and quality drops on hard ones
  ("Why does my linked-list code crash?").
- Many questions need no AI at all ("When is the IA-2 exam?"), because the answer is already in a document.

> **Problem statement:** How can a chatbot automatically decide, for each incoming question,
> whether to answer from stored documents, a free small model, or a large model, so that
> it stays close to large-model accuracy while spending as little as possible?

## 2. The solution: a router

Think of a hospital front desk: simple questions are answered on the spot, routine cases
go to a nurse, and only hard cases go to the specialist.

```
                 User question
                       |
          [1] FAQ lookup (retrieval)
          similar question already answered?
             | yes                  | no
      return stored answer   [2] Router predicts
          (cost = 0)          "how hard is this?"
                               |            |
                             easy          hard
                               |            |
                        Small local    Large API
                          model          model
                         (free)      (free tier)
                               \         /
                        Answer + log cost + route
```

| Route | Example question | Why |
|---|---|---|
| **Retrieval** | "When is the IA-2 exam?" | The answer exists in the FAQ file, so no AI is needed |
| **Small model** | "Explain what a pointer is in C." | Textbook knowledge that a 3B model handles well |
| **Large model** | "Why does this linked-list code segfault?" + code | Needs multi-step reasoning |

### Three routers, compared

1. **Rule router (baseline):** hand-written rules. Long question? Contains code? Words like
   "why", "debug", "prove"? Then it's hard.
2. **Learned router:** logistic regression on sentence embeddings, trained to predict
   *"will the small model get this wrong?"*
3. **Cascade router:** always try the small model first. It reports its confidence (1–10),
   and if confidence is low, the question escalates to the large model.

They are compared against three reference points: **always small**, **always large** and an
**oracle** (a perfect router that escalates only when the small model is wrong *and* the large model is right).

### What we measure

**Accuracy** and **cost per 1,000 questions** for every strategy, plotted as a
**cost-vs-accuracy curve** (`results.png`). The target result looks like:

> *"The learned router reaches 95% of the large model's accuracy at 30% of its cost."*

---

## 3. Tech stack

| Part | Tool |
|---|---|
| Language | Python 3.10+ |
| Small local model | [Ollama](https://ollama.com) + `qwen2.5:3b` (8 GB RAM is enough) |
| Large model | Google Gemini API, free tier (`gemini-2.5-flash` by default) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, runs on CPU) |
| Router classifier | scikit-learn Logistic Regression |
| Demo app | Streamlit |
| Charts | matplotlib |

## 4. Project structure

```
SmartModelRouter/
├── requirements.txt
├── config.py            # all settings in one place (models, prices, thresholds)
├── models.py            # talks to Ollama (small) and Gemini (large); LLM judge
├── retriever.py         # FAQ lookup using embeddings
├── router.py            # rule router + learned router
├── check_setup.py       # step 0: verify everything is installed
├── build_labels.py      # step 1: create the training dataset
├── train_router.py      # step 2: train the learned router
├── evaluate.py          # step 3: compare all strategies, draw the graph
├── app.py               # step 4: the chat demo
└── data/
    ├── faq.csv          # known questions + answers (for retrieval)
    ├── queries.csv      # 327 English questions used to train/test the router
    └── queries_kanglish.csv  # 10 Kannada/English code-mixed questions (extension)
```

### The dataset (`data/queries.csv`)

327 English starter questions across C, data structures, algorithms, DBMS, OS, computer
networks, OOP and maths:

- ~170 **easy** definition-style questions ("What is a semaphore?")
- ~155 **hard** questions: code debugging, output tracing, proofs, numericals, design

`data/queries_kanglish.csv` holds 10 **Kannada/English code-mixed (Kanglish)** questions,
kept separate so the main router is trained on English only (see *Multilingual test* below).

`reference_answer` is optional. When it is blank, the large model's answer is used as the
reference. **Replace or extend these with real questions from your students** (e.g. via a
Google Form). That is what makes the project original.

---

## 5. How to run it

### Step A: install the tools (one time)

1. **Python 3.10+**: https://www.python.org/downloads/ (on Windows, tick *"Add Python to PATH"*).
2. **Ollama**: https://ollama.com/download
3. Download the small model (about 2 GB, one time):
   ```bash
   ollama pull qwen2.5:3b
   ```
   On Windows/Mac Ollama runs in the background. On Linux, run `ollama serve` in a separate terminal.
4. **Free Gemini API key**: go to https://aistudio.google.com → **Get API key** → **Create API key**.
   Do **not** enable billing on that Google Cloud project. Without billing, the key stays free.

### Step B: set up the project

```bash
git clone https://github.com/Zhussain13/SmartModelRouter.git
cd SmartModelRouter
python -m venv venv
```

Activate the virtual environment. On Windows:

```bash
venv\Scripts\activate
```

On Mac / Linux:

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Set your API key (needed in every new terminal):

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY="paste-your-key-here"
# Windows (Command Prompt)
set GEMINI_API_KEY=paste-your-key-here
# Mac / Linux
export GEMINI_API_KEY="paste-your-key-here"
```

### Step C: run the pipeline

| # | Command | What it does |
|---|---|---|
| 0 | `python check_setup.py` | Confirms Ollama, Gemini and embeddings all work |
| 1 | `python build_labels.py` | Asks both models every question, grades the answers, saves `data/labels.csv` |
| 2 | `python train_router.py` | Trains the learned router, saves `router.joblib` |
| 3 | `python evaluate.py` | Prints the comparison table, saves `results.csv` and `results.png` (no API calls) |
| 4 | `streamlit run app.py` | Opens the chat demo in your browser |

**Want to see it working right now?** Skip steps 1–3 and run step 4. Without
`router.joblib` the app uses the rule router automatically.

Test step 1 on a few questions first:

```bash
python build_labels.py --limit 5
```

If it stops (internet drop, daily quota used up, Ctrl+C), just run it again. It saves
progress and **resumes where it left off**.

### Multilingual test (Kanglish)

After steps 1–3, check how the English-trained router copes with Kanglish questions.
Label them into a separate file:

```bash
python build_labels.py --input data/queries_kanglish.csv --output data/labels_kanglish.csv
```

Then evaluate on that file (this writes `results_kanglish.csv` and `results_kanglish.png`):

```bash
python evaluate.py --test data/labels_kanglish.csv --out results_kanglish
```

Add more Kanglish questions to `data/queries_kanglish.csv` for a meaningful result. 10 is only a start.

### Free-tier limits: what to expect

Each question needs 2–3 Gemini calls (answer + judge). The free tier allows only a limited
number of requests per minute and per day (see https://ai.google.dev/gemini-api/docs/rate-limits).
So:

- `API_SLEEP` defaults to 6 seconds between calls, and rate-limit errors (429) are retried automatically.
- Labelling all 327 questions may take **more than one day** of free quota. Run it, let it hit
  the daily limit, and run it again the next day. It resumes.
- To go faster, grade with a model that has a bigger free quota:
  `JUDGE_MODEL=gemini-2.5-flash-lite` (set it like `GEMINI_API_KEY` above).

### Things to try in the demo

- *"When is IA-2?"* → **retrieval** route, cost $0
- *"What is a compiler?"* → **small** route, cost $0
- *"Why does my code crash? int \*p; \*p=10;"* → **large** route
- Move the threshold slider and watch how routes and costs change.

---

## 6. Configuration

Every setting lives in `config.py` and can be overridden with an environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `SMALL_MODEL` | `qwen2.5:3b` | Ollama model (try `qwen2.5:1.5b` on slow PCs) |
| `LARGE_MODEL` | `gemini-2.5-flash` | Gemini model used as the "large" model |
| `JUDGE_MODEL` | same as `LARGE_MODEL` | Gemini model used to grade answers |
| `LARGE_PRICE_IN` / `LARGE_PRICE_OUT` | `0.30` / `2.50` | List price in USD per 1M tokens, **only used for the cost analysis** |
| `API_SLEEP` | `6.0` | Seconds to pause after each Gemini call |
| `API_MAX_RETRIES` | `5` | Retries on rate-limit errors |

`RETRIEVAL_THRESHOLD` (0.75) and `ROUTER_THRESHOLD` (0.5) are set directly in `config.py`.

If Google renames or retires a model, check the current names at
https://ai.google.dev/gemini-api/docs/models and set `LARGE_MODEL`.

## 7. Common problems

| Error | Fix |
|---|---|
| `Connection refused ... 11434` | Ollama is not running. Open the Ollama app, or run `ollama serve` |
| `model 'qwen2.5:3b' not found` | Run `ollama pull qwen2.5:3b` |
| Gemini `API key not valid` / `PERMISSION_DENIED` | API key not set in *this* terminal, so set it again |
| Gemini `404 model not found` | The model name changed. Set `LARGE_MODEL` to a current name |
| `429 RESOURCE_EXHAUSTED` keeps repeating | Daily free quota used up. Stop and rerun tomorrow (it resumes), or use `JUDGE_MODEL=gemini-2.5-flash-lite` |
| `train_router.py` says it needs more examples | Too few questions, or all labels the same. Label more, especially harder ones |
| Small model very slow | `ollama pull qwen2.5:1.5b` and set `SMALL_MODEL=qwen2.5:1.5b` |

---

## 8. What to put in the report

**Results table** (`results.csv`): accuracy, cost per 1,000 questions, % sent to the large
model and average latency for each strategy.

**Main graph** (`results.png`): the cost-vs-accuracy curve. Points near the **top-left** are
best (high accuracy, low cost).

**Points to discuss:**
- How close does the learned router get to the **oracle**? The gap shows the room for improvement.
- What kinds of questions does the router get wrong?
- Rules vs learned vs cascade: which is best, and why?
- **Limitation:** the large model also acts as the judge, and when there is no reference
  answer its own answer is assumed correct. Adding reference answers reduces this bias.

**Ideas to extend it:**
1. **Multilingual routing:** grow `data/queries_kanglish.csv` and compare the router on English vs
   Kanglish (see *Multilingual test*). Existing research routers are tested almost entirely on English.
2. **Better router model:** fine-tune DistilBERT instead of logistic regression.
3. **Middle tier:** small → medium (e.g. `gemini-2.5-flash-lite`) → large.
4. **Cache:** if the same question repeats, reuse the earlier answer.

**Related work:**
- *FrugalGPT* (Chen, Zaharia, Zou, 2023): LLM cascades to reduce cost
- *RouteLLM* (Ong et al., LMSYS, 2024): learned routers between strong and weak models
- *RouterBench* (Hu et al., 2024): a benchmark for LLM routing systems
