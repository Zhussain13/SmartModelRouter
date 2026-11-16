"""Step 4: the chat demo.  Run with:  streamlit run app.py"""
import os

import streamlit as st

import config
from models import ask_large, ask_small
from retriever import FAQRetriever
from router import LearnedRouter, RuleRouter

st.set_page_config(page_title="Smart Model Router")


@st.cache_resource
def load():
    faq = FAQRetriever()
    router = LearnedRouter() if os.path.exists("router.joblib") else RuleRouter()
    return faq, router


faq, router = load()

st.title("Smart Model Router")
st.caption(f"Router in use: **{router.name}** | small = {config.SMALL_MODEL} | "
           f"large = {config.LARGE_MODEL} (free tier)")

threshold = st.sidebar.slider("Router threshold (higher = cheaper)",
                              0.0, 1.0, config.ROUTER_THRESHOLD, 0.05)
if "log" not in st.session_state:
    st.session_state.log = []

q = st.chat_input("Ask a question...")
if q:
    sim, answer, matched = faq.lookup(q)
    if sim >= config.RETRIEVAL_THRESHOLD:
        item = dict(q=q, text=answer, route="retrieval", p=None, cost=0.0,
                    lat=0.0, note=f"matched FAQ: '{matched}' (sim {sim:.2f})")
    else:
        p = router.p_hard(q)
        route = "large" if p >= threshold else "small"
        with st.spinner(f"Thinking ({route} model)..."):
            try:
                res = ask_large(q) if route == "large" else ask_small(q)
                item = dict(q=q, text=res.text, route=route, p=p,
                            cost=res.cost_usd, lat=res.latency_s, note="")
            except Exception as e:
                item = dict(q=q, text=f"Error calling the {route} model: {e}",
                            route=route, p=p, cost=0.0, lat=0.0, note="")
    st.session_state.log.append(item)

for it in st.session_state.log:
    with st.chat_message("user"):
        st.write(it["q"])
    with st.chat_message("assistant"):
        st.write(it["text"])
        p_txt = "" if it["p"] is None else f" | p_hard = {it['p']:.2f}"
        st.caption(f"Route: **{it['route']}**{p_txt} | list-price cost ${it['cost']:.6f} | "
                   f"{it['lat']:.1f}s {it['note']}")

log = st.session_state.log
st.sidebar.markdown("### Session stats")
st.sidebar.metric("Cost at list price (USD)", f"{sum(i['cost'] for i in log):.6f}")
st.sidebar.caption("Free tier: you are actually charged $0.")
for route in ["retrieval", "small", "large"]:
    st.sidebar.write(f"{route}: {sum(i['route'] == route for i in log)} questions")
