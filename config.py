"""All settings in one place. Every value can be overridden with an environment variable."""
import os
import sys

# ---- Demo mode ----
# Demo mode replaces every model with a built-in simulator, so the whole project
# (pipeline, graph and chat app) runs with no Ollama, no API key and no downloads.
# Turn it on with  --demo  on any script, or the environment variable SMR_DEMO=1.
DEMO = os.getenv("SMR_DEMO", "0") == "1" or "--demo" in sys.argv

# ---- Small model: always local (Ollama) ----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
SMALL_MODEL = os.getenv("SMALL_MODEL", "qwen2.5:3b")      # or "llama3.2:3b"

# ---- Large model: choose a FREE provider ----
#   "gemini" -> Google Gemini API free tier (needs a free key, no download)
#   "ollama" -> a bigger local model through Ollama (no key, fully offline)
LARGE_PROVIDER = os.getenv("LARGE_PROVIDER", "gemini").lower()

# Gemini (free tier). Get a key at https://aistudio.google.com -> "Get API key".
# Do NOT enable billing on the Google Cloud project: then the key stays free.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Local large model (needs ~16 GB RAM for 14B; use qwen2.5:7b on 8 GB machines).
LARGE_OLLAMA_MODEL = os.getenv("LARGE_OLLAMA_MODEL", "qwen2.5:14b")

if DEMO:
    SMALL_MODEL, LARGE_MODEL = "demo-small", "demo-large"
elif LARGE_PROVIDER == "ollama":
    LARGE_MODEL = LARGE_OLLAMA_MODEL
elif LARGE_PROVIDER == "gemini":
    LARGE_MODEL = GEMINI_MODEL
else:
    raise ValueError(f"LARGE_PROVIDER must be 'gemini' or 'ollama', not {LARGE_PROVIDER!r}")

# Model used to grade answers (LLM-as-judge). Defaults to the large model.
# With Gemini, "gemini-2.5-flash-lite" has a bigger free quota and labels faster.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LARGE_MODEL)

# ---- Cost accounting ----
# Everything in this project is free to run. To compare strategies we still need a
# cost, so every large-model call is priced at what a comparable PAID API would charge
# (USD per 1 million tokens; Gemini 2.5 Flash list price). The small model is counted
# as free. Change these to model a different paid API.
LARGE_PRICE_IN = float(os.getenv("LARGE_PRICE_IN", "0.30"))
LARGE_PRICE_OUT = float(os.getenv("LARGE_PRICE_OUT", "2.50"))
SMALL_PRICE_IN = 0.0
SMALL_PRICE_OUT = 0.0

# ---- Retrieval ----
EMBED_MODEL = "all-MiniLM-L6-v2"
# Cosine similarity needed to answer from the FAQ. The demo embedder is cruder,
# so it uses a lower threshold.
RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.55" if DEMO else "0.75"))

# ---- Routing ----
ROUTER_THRESHOLD = 0.5       # p(hard) >= this  -> large model
CASCADE_MIN_CONF = 7         # small-model confidence below this -> escalate

# ---- Gemini free-tier safety ----
API_SLEEP = float(os.getenv("API_SLEEP", "6.0"))          # pause after each Gemini call
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "5"))  # retries on 429 rate limits

# ---- File locations ----
# Demo outputs go to a separate folder so they never mix with real results.
OUT_DIR = os.path.join("outputs", "demo" if DEMO else "real")
LABELS_PATH = os.path.join(OUT_DIR, "labels.csv")
TEST_PATH = os.path.join(OUT_DIR, "test_split.csv")
ROUTER_PATH = os.path.join(OUT_DIR, "router.joblib")
RESULTS_PREFIX = os.path.join(OUT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)
