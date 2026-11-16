"""Talks to the small model (Ollama) and the large model (Gemini).

Every call records tokens, cost and time so strategies can be compared later.
"""
import re
import time
from dataclasses import dataclass

import requests

import config

SYSTEM = ("You are a helpful teaching assistant for engineering students. "
          "Answer clearly and concisely.")


@dataclass
class Result:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_s: float


# ---------------- Small model (Ollama, runs locally) ----------------
def ask_small(prompt: str, system: str = SYSTEM) -> Result:
    t0 = time.time()
    r = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.SMALL_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    tin = data.get("prompt_eval_count", 0)
    tout = data.get("eval_count", 0)
    cost = (tin * config.SMALL_PRICE_IN + tout * config.SMALL_PRICE_OUT) / 1e6
    return Result(data["message"]["content"], config.SMALL_MODEL,
                  tin, tout, cost, time.time() - t0)


# ---------------- Large model (Gemini API, free tier) ----------------
_client = None


def _gemini():
    global _client
    if _client is None:
        from google import genai
        # If api_key is None, the SDK reads the GEMINI_API_KEY env variable.
        _client = genai.Client(api_key=config.GEMINI_API_KEY or None)
    return _client


def _is_rate_limit(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg


def ask_large(prompt: str, system: str = SYSTEM, model: str | None = None) -> Result:
    from google.genai import types
    model = model or config.LARGE_MODEL
    t0 = time.time()
    for attempt in range(config.API_MAX_RETRIES + 1):
        try:
            resp = _gemini().models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system,
                                                   temperature=0.2),
            )
            break
        except Exception as e:
            if not _is_rate_limit(e) or attempt == config.API_MAX_RETRIES:
                raise
            wait = 30 * (attempt + 1)   # free-tier limits are per minute
            print(f"\n  rate limited, waiting {wait}s (try {attempt + 1}) ...")
            time.sleep(wait)
            t0 = time.time()            # don't count the wait as latency
    latency = time.time() - t0
    u = resp.usage_metadata
    tin = (getattr(u, "prompt_token_count", 0) or 0) if u else 0
    # "Thinking" tokens are billed as output tokens.
    tout = ((getattr(u, "candidates_token_count", 0) or 0)
            + (getattr(u, "thoughts_token_count", 0) or 0)) if u else 0
    cost = (tin * config.LARGE_PRICE_IN + tout * config.LARGE_PRICE_OUT) / 1e6
    time.sleep(config.API_SLEEP)
    return Result(resp.text or "", model, tin, tout, cost, latency)


# ---------------- Small model + self-reported confidence (for cascade) ----------------
CONF_SUFFIX = ("\n\nAfter your answer, write one final line exactly like "
               "'CONFIDENCE: N' where N is 1-10 (how sure you are that "
               "your answer is fully correct).")


def ask_small_with_confidence(prompt: str):
    res = ask_small(prompt + CONF_SUFFIX)
    matches = re.findall(r"CONFIDENCE:\s*(\d+)", res.text)
    conf = int(matches[-1]) if matches else 5
    res.text = re.sub(r"\s*\**CONFIDENCE:\**\s*\d+.*$", "", res.text, flags=re.S).strip()
    return res, max(1, min(conf, 10))


# ---------------- LLM-as-judge (grades answers) ----------------
JUDGE_PROMPT = """You are grading a student-help chatbot.

Question:
{q}

Reference answer:
{ref}

Candidate answer:
{cand}

Is the candidate answer correct and complete enough to genuinely help a student?
Reply with only one word: YES or NO."""


def judge(question: str, candidate: str, reference: str) -> bool:
    res = ask_large(JUDGE_PROMPT.format(q=question, ref=reference, cand=candidate),
                    system="You are a strict but fair grader.",
                    model=config.JUDGE_MODEL)
    return res.text.strip().upper().startswith("YES")
