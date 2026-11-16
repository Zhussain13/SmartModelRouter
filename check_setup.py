"""Step 0: verify that the models and the embedding model all work.

    python check_setup.py           # real mode
    python check_setup.py --demo    # demo mode (always works)
"""
import config
from models import ask_large, ask_small

ok = True
mode = "DEMO (simulated models)" if config.DEMO else f"REAL (large provider = {config.LARGE_PROVIDER})"
print(f"Mode: {mode}\n")

print(f"Testing small model [{config.SMALL_MODEL}] ...")
try:
    r = ask_small("Say hello in five words.")
    print(f"  OK  {r.text[:80]!r}  ({r.latency_s:.1f}s)")
except Exception as e:
    ok = False
    print("  FAILED:", e)
    print("  -> Is Ollama running? Did you run: ollama pull", config.SMALL_MODEL)

print(f"Testing large model [{config.LARGE_MODEL}] ...")
try:
    r = ask_large("Say hello in five words.")
    print(f"  OK  {r.text[:80]!r}  API-equivalent cost=${r.cost_usd:.6f} (you pay $0)")
except Exception as e:
    ok = False
    print("  FAILED:", e)
    if config.LARGE_PROVIDER == "ollama":
        print("  -> Did you run: ollama pull", config.LARGE_OLLAMA_MODEL)
    else:
        print("  -> Is GEMINI_API_KEY set in this terminal? Is GEMINI_MODEL a current model name?")

print("Testing embeddings ...")
try:
    from retriever import embed
    print("  OK  vector size =", embed(["test"]).shape[1])
except Exception as e:
    ok = False
    print("  FAILED:", e)
    print("  -> Run: pip install -r requirements.txt (first run downloads ~90 MB)")

print("\nAll good!" if ok else "\nFix the FAILED items above, or try: python check_setup.py --demo")
raise SystemExit(0 if ok else 1)
