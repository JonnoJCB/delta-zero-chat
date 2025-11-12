# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Adaptive AI with Feedback, Mood Chart & Knowledge
# by JCB
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
# 2. DeltaAgent – Adaptive with per-slot replies
# ============================================================== 
class DeltaAgent:
    def __init__(
        self,
        n_slots=5,
        lr=0.07,
        brain_file="global_brain.pkl",
        data_file="chat_log.enc",
        mood_file="mood_history.pkl",
        key_file="secret.key",
    ):
        self.n_slots = n_slots
        self.lr = lr
        self.brain_file = brain_file
        self.data_file = data_file
        self.mood_file = mood_file
        self.key_file = key_file
        self.knowledge = load_knowledge()

        # Short-term memory (clears on app reload)
        self.short_memory = []

        self.memory = []
        self.mood_history = []
        self.last_slot = None

        # Encryption key
        if not os.path.exists(key_file):
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
        with open(key_file, "rb") as f:
            self.cipher = Fernet(f.read())

        # Load brain
        if os.path.exists(brain_file):
            with open(brain_file, "rb") as f:
                saved = pickle.load(f)
                self.w = saved.get("w", np.ones(n_slots) / n_slots)
        else:
            self.w = np.ones(n_slots) / n_slots

        # Load encrypted chat log
        if os.path.exists(data_file):
            try:
                with open(data_file, "rb") as f:
                    encrypted = f.read()
                    if encrypted:
                        decrypted = self.cipher.decrypt(encrypted)
                        df = pd.read_csv(pd.io.common.StringIO(decrypted.decode()))
                        self.memory = df.to_dict("records")
                    else:
                        self.memory = []
            except Exception:
                st.warning("Could not decrypt chat log. Starting fresh.")
                self.memory = []
        else:
            self._save_encrypted_df(pd.DataFrame(columns=[
                "timestamp","user","input","response","slot",
                "reward","feedback","fb_text"
            ]))

        # Load mood
        if os.path.exists(mood_file):
            with open(mood_file, "rb") as f:
                self.mood_history = pickle.load(f)
        else:
            self.mood_history = []

    def choose_slot(self):
        probs = self.w / self.w.sum()
        slot = np.random.choice(range(self.n_slots), p=probs)
        self.last_slot = slot
        return slot

    REPLIES = [
        ["Wow, fascinating!", "I'm intrigued!", "That's wild!"],                    # 0: Curious
        ["I understand.", "That makes sense.", "Clear as day."],                   # 1: Calm
        ["Tell me more!", "Keep going!", "Don't stop now!"],                       # 2: Engaging
        ["How do you feel about that?", "Why do you think so?", "That's deep."],   # 3: Empathetic
        ["Let's analyze this.", "Interesting angle.", "Break it down."]            # 4: Analytical
    ]

    def generate_response(self, user_input, slot):
        base = random.choice(self.REPLIES[slot])
        if self.knowledge and random.random() < 0.2:
            fact = random.choice(self.knowledge)
            base += f" Fun fact: {fact}"
        return base + f" [slot {slot}]"

    def respond(self, user_input):
        slot = self.choose_slot()
        response = self.generate_response(user_input, slot)
        # Add to short-term memory
        self.short_memory.append({"input": user_input, "response": response, "slot": slot})
        return response, slot

    def update(self, reward):
        if self.last_slot is not None:
            self.w[self.last_slot] += self.lr * (reward - self.w[self.last_slot])
            self.w = np.clip(self.w, 0.01, None)
            self.w /= self.w.sum()

    def log_interaction(self, user, user_input, response, slot,
                        reward=None, feedback=None, fb_text=None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "timestamp": ts, "user": user, "input": user_input,
            "response": response, "slot": slot,
            "reward": reward, "feedback": feedback, "fb_text": fb_text
        }
        self.memory.append(entry)
        self._save_encrypted_df(pd.DataFrame(self.memory))

    def _save_encrypted_df(self, df):
        csv = df.to_csv(index=False)
        encrypted = self.cipher.encrypt(csv.encode())
        with open(self.data_file, "wb") as f:
            f.write(encrypted)

    def save_state(self):
        with open(self.brain_file, "wb") as f:
            pickle.dump({"w": self.w}, f)
        with open(self.mood_file, "wb") as f:
            pickle.dump(self.mood_history, f)

    def update_mood(self, mood_value):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.mood_history.append({"timestamp": ts, "mood": mood_value})
        self.save_state()

# ============================================================== 
# 3. Streamlit UI – Original Layout
# ============================================================== 

st.set_page_config(page_title="Δ-Zero Chat", layout="wide")

# Intro animation
if "intro_shown" not in st.session_state:
    placeholder = st.empty()
    intro = "Welcome to Δ-Zero Chat"
    for i in range(len(intro) + 1):
        placeholder.markdown(f"<h2>{intro[:i]}</h2>", unsafe_allow_html=True)
    placeholder.markdown("<i>by JCB</i>", unsafe_allow_html=True)  # <- italic text added
    placeholder.empty()
    st.session_state.intro_shown = True

st.title("Δ-Zero Chat – Adaptive AI")
st.markdown("<sub>by JCB</sub>", unsafe_allow_html=True)

agent = DeltaAgent()

# ---------- Sidebar ----------
st.sidebar.header("Mood Tracker")
mood = st.sidebar.slider("Your mood", 0.0, 10.0, 5.0, 0.5)
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

# ---------- Slot Confidence ----------
weights = agent.w / agent.w.sum()
slot_labels = ["Curious", "Calm", "Engaging", "Empathetic", "Analytical"]
conf_df = pd.DataFrame({"Style": slot_labels, "Confidence": weights})
conf_fig = px.bar(conf_df, x="Style", y="Confidence", title="AI Personality Confidence",
                  color="Confidence", color_continuous_scale="Blues")
st.plotly_chart(conf_fig, use_container_width=True)

# ---------- Chat ----------
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
                        st.rerun()
                with col2:
                    if st.button("Bad", key=f"bad_{i}"):
                        agent.update(0.0)
                        agent.log_interaction("user", "", "", agent.last_slot,
                                              reward=0.0, feedback="bad")
                        st.error("Learning: avoiding this style")
                        st.rerun()

# Input field
user_input = st.text_input("Type your message...", key="user_input")

# Process input
if user_input.strip():
    response, slot = agent.respond(user_input)
    agent.log_interaction("user", user_input, response, slot)
    agent.save_state()

    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    st.session_state.chat_history.append({"sender": "bot", "message": response})
    st.session_state.last_bot_idx = len(st.session_state.chat_history) - 1

    st.rerun()

# Always show chat
display_chat()

# Reuse past messages
if st.checkbox("Reuse past messages"):
    past = [e["input"] for e in agent.memory[-20:] if e["input"]]
    sel = st.selectbox("Pick one", [""] + past)
    if sel:
        st.session_state.user_input = sel

# Learning summary
if st.button("Show Feedback Summary"):
    fb = [e for e in agent.memory if e["feedback"]]
    if fb:
        df = pd.DataFrame(fb)["feedback"].value_counts().reset_index()
        df.columns = ["Feedback", "Count"]
        fig = px.pie(df, names="Feedback", values="Count", title="User Feedback")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feedback yet.")
