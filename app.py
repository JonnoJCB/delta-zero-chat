# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Adaptive AI with Incremental Learning
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================== #
# CONFIG
# ============================================================== #
BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
LEARNED_FILE = os.path.join(KNOWLEDGE_DIR, "learned_facts.txt")
BRAIN_FILE = os.path.join(BASE_DIR, "global_brain.pkl")
DATA_FILE = os.path.join(BASE_DIR, "chat_log.enc")
MOOD_FILE = os.path.join(BASE_DIR, "mood_history.pkl")
KEY_FILE = os.path.join(BASE_DIR, "secret.key")

# ============================================================== #
# LOAD KNOWLEDGE
# ============================================================== #
def load_knowledge():
    """Load all lines from text files inside /knowledge folder."""
    knowledge = []
    if os.path.exists(KNOWLEDGE_DIR):
        for f in os.listdir(KNOWLEDGE_DIR):
            if f.endswith(".txt"):
                path = os.path.join(KNOWLEDGE_DIR, f)
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        for line in file:
                            text = line.strip()
                            if text:
                                knowledge.append(text)
                except Exception as e:
                    st.warning(f"Could not read {f}: {e}")
    return knowledge

# ============================================================== #
# Δ-Zero AGENT
# ============================================================== #
class DeltaAgent:
    REPLIES = [
        ["Wow, fascinating!", "I'm intrigued!", "That's wild!"],          # Curious
        ["I understand.", "That makes sense.", "Clear as day."],          # Calm
        ["Tell me more!", "Keep going!", "Don't stop now!"],              # Engaging
        ["How do you feel about that?", "Why do you think so?", "Deep."], # Empathetic
        ["Let's analyze this.", "Interesting angle.", "Break it down."],  # Analytical
    ]

    def __init__(self, n_slots=5, lr=0.07, context_size=5):
        self.n_slots = n_slots
        self.lr = lr
        self.context_size = context_size
        self.knowledge = load_knowledge()
        self.memory = []
        self.mood_history = []
        self.context = []
        self.last_slot = None

        # Encryption setup
        self.cipher = self._load_or_create_key()

        # Load saved states
        self.w = self._load_brain()
        self.memory = self._load_encrypted_log()
        self.mood_history = self._load_mood()

        # Fit context retriever
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._fit_vectorizer()

    # ------------------- FILE UTILITIES ------------------- #
    def _fit_vectorizer(self):
        if self.knowledge:
            self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge)
        else:
            self.knowledge_vectors = None

    def _load_or_create_key(self):
        if not os.path.exists(KEY_FILE):
            key = Fernet.generate_key()
            with open(KEY_FILE, "wb") as f:
                f.write(key)
        with open(KEY_FILE, "rb") as f:
            return Fernet(f.read())

    def _load_brain(self):
        if os.path.exists(BRAIN_FILE):
            with open(BRAIN_FILE, "rb") as f:
                data = pickle.load(f)
                return data.get("w", np.ones(self.n_slots) / self.n_slots)
        return np.ones(self.n_slots) / self.n_slots

    def _load_encrypted_log(self):
        if not os.path.exists(DATA_FILE):
            return []
        try:
            with open(DATA_FILE, "rb") as f:
                encrypted = f.read()
                if not encrypted:
                    return []
                decrypted = self.cipher.decrypt(encrypted)
                df = pd.read_csv(pd.io.common.StringIO(decrypted.decode()))
                return df.to_dict("records")
        except Exception:
            return []

    def _load_mood(self):
        if os.path.exists(MOOD_FILE):
            with open(MOOD_FILE, "rb") as f:
                return pickle.load(f)
        return []

    def _save_encrypted_df(self, df):
        csv = df.to_csv(index=False)
        encrypted = self.cipher.encrypt(csv.encode())
        with open(DATA_FILE, "wb") as f:
            f.write(encrypted)

    # ------------------- LEARNING ------------------- #
    def _apply_mood_boost(self, mood):
        w = self.w.copy()
        if mood <= 3:
            w[3] *= 1.4  # empathetic
        elif mood >= 7:
            w[0] *= 1.3  # curious
            w[2] *= 1.3  # engaging
        return w / w.sum()

    def choose_slot(self, mood=None):
        probs = self._apply_mood_boost(mood) if mood is not None else self.w
        slot = np.random.choice(self.n_slots, p=probs)
        self.last_slot = slot
        return slot

    # ------------------- RESPONSES ------------------- #
    def generate_response(self, user_input, slot, mood=None):
        if not user_input:
            return random.choice(self.REPLIES[slot])

        # Context retrieval from knowledge
        if self.knowledge and self.knowledge_vectors is not None:
            try:
                query_vec = self.vectorizer.transform([user_input])
                sims = cosine_similarity(query_vec, self.knowledge_vectors)[0]
                best_idx = int(np.argmax(sims))
                if sims[best_idx] > 0.15:
                    return self.knowledge[best_idx] + f" [slot {slot}]"
            except Exception as e:
                print("Context retrieval error:", e)

        # fallback
        base = random.choice(self.REPLIES[slot])
        return base + f" [slot {slot}]"

    def respond(self, user_input, mood=None):
        slot = self.choose_slot(mood)
        response = self.generate_response(user_input, slot, mood)

        # auto-learning: store new interesting facts
        self._auto_learn(user_input, response)

        self.context.append({"input": user_input, "response": response})
        self.context = self.context[-self.context_size * 2:]
        return response, slot

    # ------------------- AUTO-LEARNING ------------------- #
    def _auto_learn(self, user_input, response):
        """If the user says something that looks like new information, store it."""
        if len(user_input.split()) > 3 and not any(user_input.lower() in k.lower() for k in self.knowledge):
            try:
                os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
                with open(LEARNED_FILE, "a", encoding="utf-8") as f:
                    f.write(user_input.strip() + "\n")
                self.knowledge.append(user_input.strip())
                self._fit_vectorizer()
            except Exception as e:
                print("Learning save error:", e)

    # ------------------- UPDATES ------------------- #
    def update(self, reward):
        if self.last_slot is not None:
            self.w[self.last_slot] += self.lr * (reward - self.w[self.last_slot])
            self.w = np.clip(self.w, 0.01, None)
            self.w /= self.w.sum()

    def log_interaction(self, user_input, response, slot, reward=None, feedback=None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {"timestamp": ts, "input": user_input, "response": response,
                 "slot": slot, "reward": reward, "feedback": feedback}
        self.memory.append(entry)
        df = pd.DataFrame(self.memory)
        self._save_encrypted_df(df)

    def save_state(self):
        with open(BRAIN_FILE, "wb") as f:
            pickle.dump({"w": self.w}, f)
        with open(MOOD_FILE, "wb") as f:
            pickle.dump(self.mood_history, f)

    def update_mood(self, mood_value):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.mood_history.append({"timestamp": ts, "mood": mood_value})
        self.save_state()

# ============================================================== #
# STREAMLIT INTERFACE
# ============================================================== #
st.set_page_config(page_title="Δ-Zero Chat", layout="wide")
st.title("🤖 Δ-Zero Chat – Adaptive AI")
st.markdown("<sub>by JCB – self-learning AI companion</sub>", unsafe_allow_html=True)

# Initialize agent
if "agent" not in st.session_state:
    with st.spinner("Booting Δ-Zero neural circuits..."):
        st.session_state.agent = DeltaAgent()
agent = st.session_state.agent

# Sidebar
st.sidebar.header("Mood Tracker")
mood = st.sidebar.slider("Your current mood", 0.0, 10.0, 5.0, 0.5)
if st.sidebar.button("Record Mood"):
    agent.update_mood(mood)
    st.sidebar.success("Mood recorded!")

if agent.mood_history:
    df_mood = pd.DataFrame(agent.mood_history)
    fig_mood = px.line(df_mood, x="timestamp", y="mood",
                       title="Mood Over Time", markers=True)
    st.sidebar.plotly_chart(fig_mood, width="stretch")

st.sidebar.info(f"Total chats: {len(agent.memory)}")
st.sidebar.success(f"Knowledge entries: {len(agent.knowledge)}")

# Confidence display
slot_labels = ["Curious", "Calm", "Engaging", "Empathetic", "Analytical"]
weights = (agent.w / agent.w.sum()).round(3)
conf_df = pd.DataFrame({"Style": slot_labels, "Confidence": weights})
conf_fig = px.bar(conf_df, x="Style", y="Confidence",
                  color="Confidence", title="AI Personality",
                  color_continuous_scale="Blues", height=250)
st.plotly_chart(conf_fig, width="stretch")

# Chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_bot_idx" not in st.session_state:
    st.session_state.last_bot_idx = -1

def render_chat():
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["sender"] == "user":
            st.markdown(
                f"<div style='background:#D1E7DD;padding:10px;border-radius:8px;text-align:right'>"
                f"<b>You:</b> {msg['message']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='background:#F8D7DA;padding:10px;border-radius:8px'>"
                f"<b>Δ-Zero:</b> {msg['message']}</div>", unsafe_allow_html=True)
            if i == st.session_state.last_bot_idx:
                c1, c2 = st.columns(2)
                if c1.button("👍", key=f"good_{i}"):
                    agent.update(1.0)
                    agent.log_interaction("", "", agent.last_slot, reward=1.0, feedback="good")
                    st.rerun()
                if c2.button("👎", key=f"bad_{i}"):
                    agent.update(0.0)
                    agent.log_interaction("", "", agent.last_slot, reward=0.0, feedback="bad")
                    st.rerun()

render_chat()

# Input
if user_input := st.chat_input("Talk to Δ-Zero…"):
    with st.spinner("Δ-Zero is processing..."):
        response, slot = agent.respond(user_input, mood)
    agent.log_interaction(user_input, response, slot)
    agent.save_state()
    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    st.session_state.chat_history.append({"sender": "bot", "message": response})
    st.session_state.last_bot_idx = len(st.session_state.chat_history) - 1
    st.rerun()
