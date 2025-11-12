# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Adaptive AI with Feedback, Mood Chart & Knowledge
# by JCB – your personalized AI companion by JCB
# --------------------------------------------------------------
import streamlit as st
import numpy as np
import os
import pickle
import pandas as pd
import random
from datetime import datetime
from cryptography.fernet import Fernet
import plotly.express as px

# ==============================================================
# 1. Load knowledge from /knowledge/*.txt
# ==============================================================
def load_knowledge():
    knowledge = []
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
    if os.path.exists(knowledge_dir):
        for filename in os.listdir(knowledge_dir):
            if filename.endswith(".txt"):
                path = os.path.join(knowledge_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    knowledge.extend([line.strip() for line in f if line.strip()])
    return knowledge

# ==============================================================
# 2. DeltaAgent – (unchanged – copy from your last working version)
# ==============================================================
class DeltaAgent:
    # ... (the whole class you already have) ...
    pass

# ==============================================================
# 3. Streamlit UI
# ==============================================================
st.set_page_config(page_title="Δ-Zero Chat", layout="wide")
st.title("Δ-Zero Chat – Adaptive AI")
st.markdown(
    "<sub>by JCB – your personalized AI companion by JCB</sub>",
    unsafe_allow_html=True
)

agent = DeltaAgent()

# ---------- Sidebar (unchanged) ----------
st.sidebar.header("Mood Tracker")
mood = st.sidebar.slider("Your mood (optional)", 0.0, 10.0, 5.0, 0.5)
if st.sidebar.button("Record Mood"):
    agent.update_mood(mood)
    st.sidebar.success("Saved!")

if agent.mood_history:
    df = pd.DataFrame(agent.mood_history)
    fig = px.line(df, x="timestamp", y="mood", title="Mood Over Time", markers=True)
    st.sidebar.plotly_chart(fig, use_container_width=True)

st.sidebar.info(f"Chats stored: {len(agent.memory)}")
if agent.knowledge:
    st.sidebar.success(f"Loaded {len(agent.knowledge)} facts")

# ---------- Slot Confidence (small chart) ----------
weights = agent.w / agent.w.sum()
slot_labels = ["Curious", "Calm", "Engaging", "Empathetic", "Analytical"]
conf_df = pd.DataFrame({"Style": slot_labels, "Confidence": weights})
conf_fig = px.bar(
    conf_df,
    x="Style",
    y="Confidence",
    title="AI Personality Confidence",
    color="Confidence",
    color_continuous_scale="Blues",
    height=250
)
st.plotly_chart(conf_fig, use_container_width=True)

# ---------- Chat history ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_bot_idx" not in st.session_state:
    st.session_state.last_bot_idx = -1

def display_chat():
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["sender"] == "user":
            st.markdown(
                f"<div style='background:#D1E7DD;padding:10px;border-radius:8px;margin:5px 0'>"
                f"<b>You:</b> {msg['message']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='background:#F8D7DA;padding:10px;border-radius:8px;margin:5px 0'>"
                f"<b>Δ-Zero:</b> {msg['message']}</div>", unsafe_allow_html=True)
            if i == st.session_state.last_bot_idx:
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Good", key=f"good_{i}"):
                        agent.update(1.0)
                        agent.log_interaction("user", "", "", agent.last_slot,
                                              reward=1.0, feedback="good")
                        st.success("Learning: favoring this style")
                with col2:
                    if st.button("Bad", key=f"bad_{i}"):
                        agent.update(0.0)
                        agent.log_interaction("user", "", "", agent.last_slot,
                                              reward=0.0, feedback="bad")
                        st.error("Learning: avoiding this style")

# --------------------------------------------------------------
#  INPUT + SEND (works with ENTER on ALL Streamlit versions)
# --------------------------------------------------------------
if "msg_to_send" not in st.session_state:
    st.session_state.msg_to_send = ""

def submit_on_enter():
    if st.session_state.msg_to_send.strip():
        st.session_state.pending_message = st.session_state.msg_to_send
        st.session_state.msg_to_send = ""

user_input = st.text_input(
    "Type your message…",
    placeholder="Ask Δ-Zero anything…",
    key="msg_to_send",
    on_change=submit_on_enter
)

col1, col2 = st.columns([5, 1])
with col2:
    send_clicked = st.button("Send")

if send_clicked or getattr(st.session_state, "pending_message", None):
    msg = (st.session_state.pending_message
           if "pending_message" in st.session_state else user_input)

    if msg.strip():
        response, slot = agent.respond(msg)
        agent.log_interaction("user", msg, response, slot)
        agent.save_state()

        st.session_state.chat_history.append({"sender": "user", "message": msg})
        st.session_state.chat_history.append({"sender": "bot",  "message": response})
        st.session_state.last_bot_idx = len(st.session_state.chat_history) - 1

        if "pending_message" in st.session_state:
            del st.session_state.pending_message
        st.rerun()

# Always show the chat
display_chat()

# --------------------------------------------------------------
#  Reuse past messages
# --------------------------------------------------------------
if st.checkbox("Reuse past messages"):
    past = [e["input"] for e in agent.memory[-20:] if e["input"]]
    sel = st.selectbox("Pick one", [""] + past)
    if sel:
        st.session_state.msg_to_send = sel
        st.rerun()

# --------------------------------------------------------------
#  Learning summary
# --------------------------------------------------------------
if st.button("Show Feedback Summary"):
    fb = [e for e in agent.memory if e["feedback"]]
    if fb:
        df = pd.DataFrame(fb)["feedback"].value_counts().reset_index()
        df.columns = ["Feedback", "Count"]
        fig = px.pie(df, names="Feedback", values="Count", title="User Feedback")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feedback yet.")
