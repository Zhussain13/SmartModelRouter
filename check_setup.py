"""Step 0: verify that Ollama, Gemini and the embedding model all work."""
import config
from models import ask_large, ask_small

print("Testing small model via Ollama ...")
try:
    r = ask_small("Say hello in five words.")
    print(f"  OK  [{r.model}] {r.text[:80]!r}  ({r.latency_s:.1f}s)")
except Exception as e:
    print("  FAILED:", e)
    print("  -> Is Ollama running? Did you run: ollama pull", config.SMALL_MODEL)

print("Testing large model via Gemini API (free tier) ...")
try:
    r = ask_large("Say hello in five words.")
    print(f"  OK  [{r.model}] {r.text[:80]!r}  list-price cost=${r.cost_usd:.6f} (you pay $0)")
except Exception as e:
    print("  FAILED:", e)
    print("  -> Is GEMINI_API_KEY set? Is the model name in config.py correct?")

print("Testing embeddings ...")
try:
    from retriever import embed
    print("  OK  vector size =", embed(["test"]).shape[1])
except Exception as e:
    print("  FAILED:", e)
    print("  -> Run: pip install -r requirements.txt (first run downloads ~90 MB)")
