"""Step 4: the chat demo.

    streamlit run app.py              # real models
    streamlit run app.py -- --demo    # simulated models, no setup needed
"""
import os

import streamlit as st

import config
from models import ask_large, ask_small
from retriever import FAQRetriever
from router import LearnedRouter, RuleRouter, load_tuning

st.set_page_config(page_title="Smart Model Router", page_icon="🔀")


@st.cache_resource
def load():
    faq = FAQRetriever()
    router = LearnedRouter() if os.path.exists(config.ROUTER_PATH) else RuleRouter()
    return faq, router


faq, router = load()

st.title("🔀 Smart Model Router")
st.caption(f"Router: **{router.name}** · small = `{config.SMALL_MODEL}` · "
           f"large = `{config.LARGE_MODEL}`")
if config.DEMO:
    st.info("**Demo mode**: answers are simulated, so you can see the routing without "
            "installing any model. See the README to run with real free models.")

tune = load_tuning()
threshold = st.sidebar.slider("Router threshold (higher = cheaper)", 0.0, 1.0,
                              float(tune[router.name]), 0.05)
if tune["tuned"]:
    st.sidebar.caption(f"Starts at the tuned value {tune[router.name]:.2f}, the cheapest that keeps "
                       f"{tune.get('target', config.ACCURACY_TARGET):.0%} of large-model accuracy.")
if "log" not in st.session_state:
    st.session_state.log = []

q = st.chat_input("Ask a question...")
if q:
    sim, answer, matched = faq.lookup(q)
    if sim >= config.RETRIEVAL_THRESHOLD:
        item = dict(q=q, text=answer, route="retrieval", p=None, cost=0.0,
                    lat=0.0, note=f"· matched FAQ: '{matched}' (sim {sim:.2f})")
    else:
        p = router.p_hard(q)
        route = "large" if p >= threshold else "small"
        with st.spinner(f"Thinking ({route} model)..."):
            try:
                res = ask_large(q) if route == "large" else ask_small(q)
                item = dict(q=q, text=res.text, route=route, p=p,
                            cost=res.cost_usd, lat=res.latency_s, note="")
            except Exception as e:
                item = dict(q=q, text=f"⚠️ Error calling the {route} model: {e}",
                            route=route, p=p, cost=0.0, lat=0.0, note="")
    st.session_state.log.append(item)

ICON = {"retrieval": "📄", "small": "🟢", "large": "🔵"}
for it in st.session_state.log:
    with st.chat_message("user"):
        st.write(it["q"])
    with st.chat_message("assistant"):
        st.write(it["text"])
        p_txt = "" if it["p"] is None else f" · p_hard = {it['p']:.2f}"
        st.caption(f"{ICON[it['route']]} Route: **{it['route']}**{p_txt} · "
                   f"API-equivalent cost ${it['cost']:.6f} · {it['lat']:.1f}s {it['note']}")

log = st.session_state.log
st.sidebar.markdown("### Session stats")
st.sidebar.metric("API-equivalent cost (USD)", f"{sum(i['cost'] for i in log):.6f}")
st.sidebar.caption("Everything here is free to run. The cost shows what the same "
                   "traffic would cost on a paid API.")
for route in ["retrieval", "small", "large"]:
    n = sum(i["route"] == route for i in log)
    st.sidebar.write(f"{ICON[route]} {route}: {n} question{'' if n == 1 else 's'}")
st.sidebar.markdown("### Try")
st.sidebar.markdown("- When is IA-2?\n- What is a compiler?\n"
                    "- Why does my code crash? `int *p; *p=10;`")
