"""All settings in one place. Every value can be overridden with an env variable."""
import os

# ---- Small local model (Ollama) ----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
SMALL_MODEL = os.getenv("SMALL_MODEL", "qwen2.5:3b")   # or "llama3.2:3b"

# ---- Large API model (Gemini, FREE tier) ----
# Get a free key at https://aistudio.google.com -> "Get API key".
# Do NOT enable billing on the Google Cloud project: without billing the key
# stays on the free tier and you can never be charged.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LARGE_MODEL = os.getenv("LARGE_MODEL", "gemini-2.5-flash")
# Model used to grade answers. Defaults to the large model; set it to a model
# with a bigger free quota (e.g. "gemini-2.5-flash-lite") to label faster.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LARGE_MODEL)

# "List" prices in USD per 1 million tokens.
# On the free tier you pay nothing. These numbers are ONLY used to calculate
# what each routing strategy *would* cost, which is what the project measures.
# Check https://ai.google.dev/pricing and update if they change.
LARGE_PRICE_IN = float(os.getenv("LARGE_PRICE_IN", "0.30"))
LARGE_PRICE_OUT = float(os.getenv("LARGE_PRICE_OUT", "2.50"))
SMALL_PRICE_IN = 0.0     # local model = free
SMALL_PRICE_OUT = 0.0

# ---- Retrieval ----
EMBED_MODEL = "all-MiniLM-L6-v2"
RETRIEVAL_THRESHOLD = 0.75   # cosine similarity needed to answer from FAQ

# ---- Routing ----
ROUTER_THRESHOLD = 0.5       # p(hard) >= this  -> large model
CASCADE_MIN_CONF = 7         # small-model confidence below this -> escalate

# ---- Free-tier safety ----
# Pause after every Gemini call (seconds). 6 s keeps you under ~10 requests/min.
API_SLEEP = float(os.getenv("API_SLEEP", "6.0"))
# On a 429 / rate-limit error, wait and retry this many times.
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "5"))
