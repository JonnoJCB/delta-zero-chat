# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Adaptive AI with Feedback, Mood Chart & Knowledge
# by JCB – your personalized AI companion
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
from sentence_transformers import SentenceTransformer, util

# ==============================================================
# CONFIG & PATHS
# ==============================================================
BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
CONV_DIR = os.path.join(BASE_DIR, "conversations")
BRAIN_FILE = os.path.join(BASE_DIR, "global_brain.pkl")
DATA_FILE = os.path.join(BASE_DIR, "chat_log.enc")
MOOD_FILE = os.path.join(BASE_DIR, "mood_history.pkl")
KEY_FILE = os.path.join(BASE_DIR, "secret.key")

# ==============================================================
# 1. Load Knowledge & Conversations
# ==============================================================
def load_knowledge():
    knowledge = []
    if os.path.exists(KNOWLEDGE_DIR):
        for filename in os.listdir(KNOWLEDGE_DIR):
            if filename.endswith(".txt"):
                path = os.path.join(KNOWLEDGE_DIR, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        knowledge.extend([line.strip() for line in f if line.strip()])
                except Exception as e:
                    st.warning(f"Failed to read knowledge file {filename}: {e}")
    return knowledge


def load_conversations():
    convs = []
    if not os.path.exists(CONV_DIR):
        return convs
    for filename in os.listdir(CONV_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(CONV_DIR, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    for i in range(0, len(lines) - 1, 2):
                        user_msg = lines[i]
                        bot_msg = lines[i + 1] if i + 1 < len(lines) else ""
                        if user_msg and bot_msg:
                            convs.append({"user": user_msg, "bot": bot_msg})
            except Exception as e:
                st.error(f"Could not read conversation file {filename}: {e}")
    return convs


# ==============================================================
# 2. DeltaAgent – Core AI Logic
# ==============================================================
class DeltaAgent:
    REPLIES = [
        ["Wow, fascinating!", "I'm intrigued!", "That's wild!"],          # 0: Curious
        ["I understand.", "That makes sense.", "Clear as day."],          # 1: Calm
        ["Tell me more!", "Keep going!", "Don't stop now!"],              # 2: Engaging
        ["How do you feel about that?", "Why do you think so?", "Deep."], # 3: Empathetic
        ["Let's analyze this.", "Interesting angle.", "Break it down."],  # 4: Analytical
    ]

    def __init__(self, n_slots=5, lr=0.07, context_size=5):
        self.n_slots = n_slots
        self.lr = lr
        self.context_size = context_size
        self.knowledge = load_knowledge()
        self.conversations = load_conversations()
        self.memory = []
        self.mood_history = []
        self.context = []
        self.last_slot = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Encryption
        self.cipher = self._load_or_create_key()

        # Load brain, memory, mood
        self.w = self._load_brain()
        self.memory = self._load_encrypted_log()
        self.mood_history = self._load_mood()
        self._update_dynamic_convos()

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
            st.warning("Could not decrypt chat log. Starting fresh.")
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

    def _update_dynamic_convos(self):
        self.dynamic_convos = []
        for i in range(0, len(self.memory), 2):
            if i + 1 >= len(self.memory):
                break
            user_msg = self.memory[i].get("input", "")
            bot_msg = self.memory[i + 1].get("response", "") if i + 1 < len(self.memory) else ""
            if user_msg and bot_msg:
                self.dynamic_convos.append({"user": user_msg, "bot": bot_msg})

    def _apply_mood_boost(self, mood):
        w = self.w.copy()
        if mood <= 3:
            w[3] *= 1.4
        elif mood >= 7:
            w[2] *= 1.4
            w[0] *= 1.4
        w = np.clip(w, 0.01, None)
        return w / w.sum()

    def choose_slot(self, mood=None):
        probs = self.w
        if mood is not None:
            probs = self._apply_mood_boost(mood)
        slot = np.random.choice(self.n_slots, p=probs)
        self.last_slot = slot
        return slot

    def generate_response(self, user_input, slot, mood=None):
        context_texts = [
            f"{turn.get('input','')} {turn.get('response','')}"
            for turn in self.context[-self.context_size*2:]
        ]
        all_convos = self.conversations + self.dynamic_convos
        candidate_texts = [c["user"] for c in all_convos] + context_texts

        if candidate_texts:
            try:
                query_emb = self.model.encode(user_input, convert_to_tensor=True)
                text_embs = self.model.encode(candidate_texts, convert_to_tensor=True)
                scores = util.cos_sim(query_emb, text_embs)[0]
                best_idx = scores.argmax().item()
                if scores[best_idx] > 0.5 and best_idx < len(all_convos):
                    return all_convos[best_idx]["bot"] + f" [slot {slot}]"
            except Exception as e:
                st.warning(f"Embedding error: {e}")

        user_words = set(user_input.lower().split())
        if len(user_words) >= 3:
            candidates = []
            for conv in all_convos:
                conv_words = set(conv["user"].lower().split())
                overlap = len(user_words & conv_words)
                if overlap >= 3:
                    candidates.append((conv["bot"], overlap))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0] + f" [slot {slot}]"

        base = random.choice(self.REPLIES[slot])
        if self.knowledge and random.random() < 0.2:
            fact = random.choice(self.knowledge)
            base += f" Fun fact: {fact}"
        return base + f" [slot {slot}]"

    def respond(self, user_input, mood=None):
        slot = self.choose_slot(mood)
        response = self.generate_response(user_input, slot, mood)
        self.context.append({"input": user_input, "response": response})
        if len(self.context) > self.context_size * 2:
            self.context = self.context[-self.context_size * 2:]
        return response, slot

    def update(self, reward):
        if self.last_slot is not None:
            self.w[self.last_slot] += self.lr * (reward - self.w[self.last_slot])
            self.w = np.clip(self.w, 0.01, None)
            self.w /= self.w.sum()

    def log_interaction(self, user_input, response, slot, reward=None, feedback=None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "timestamp": ts,
            "input": user_input,
            "response": response,
            "slot": slot,
            "reward": reward,
            "feedback": feedback
        }
        self.memory.append(entry)
        df = pd.DataFrame(self.memory)
        self._save_encrypted_df(df)
        self._update_dynamic_convos()

    def save_state(self):
        with open(BRAIN_FILE, "wb") as f:
            pickle.dump({"w": self.w}, f)
        with open(MOOD_FILE, "wb") as f:
            pickle.dump(self.mood_history, f)

    def update_mood(self, mood_value):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.mood_history.append({"timestamp": ts, "mood": mood_value})
        self.save_state()


# ==============================================================
# 3. Streamlit UI – Fully Fixed
# ==============================================================
st.set_page_config(page_title="Δ-Zero Chat", layout="wide")
st.title("Δ-Zero Chat – Adaptive AI")
st.markdown("<sub>by JCB – your personalized AI companion</sub>", unsafe_allow_html=True)

# Initialize agent
if "agent" not in st.session_state:
    with st.spinner("Initializing Δ-Zero..."):
        st.session_state.agent = DeltaAgent()
agent = st.session_state.agent

# Initialize chat state
for key in ["chat_history", "last_bot_idx", "chat_container"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else -1 if key == "last_bot_idx" else st.container()

# ---------- Sidebar ----------
st.sidebar.header("Mood Tracker")
mood = st.sidebar.slider("Your current mood", 0.0, 10.0, 5.0, 0.5)
if st.sidebar.button("Record Mood"):
    agent.update_mood(mood)
    st.sidebar.success("Mood recorded!")

if agent.mood_history:
    df_mood = pd.DataFrame(agent.mood_history)
    fig_mood = px.line(df_mood, x="timestamp", y="mood", title="Mood Over Time", markers=True)
    st.sidebar.plotly_chart(fig_mood, use_container_width=True)

st.sidebar.info(f"Total chats: {len(agent.memory)}")
if agent.knowledge:
    st.sidebar.success(f"Knowledge: {len(agent.knowledge)} facts")

# ---------- AI Confidence ----------
slot_labels = ["Curious", "Calm", "Engaging", "Empathetic", "Analytical"]
weights = (agent.w / agent.w.sum()).round(3)
conf_df = pd.DataFrame({"Style": slot_labels, "Confidence": weights})
conf_fig = px.bar(conf_df, x="Style", y="Confidence", title="AI Personality", color="Confidence",
                  color_continuous_scale="Blues", height=250)
st.plotly_chart(conf_fig, use_container_width=True)

# ---------- Chat Renderer ----------
def render_chat():
    with st.session_state.chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["sender"] == "user":
                st.markdown(
                    f"<div style='background:#D1E7DD;padding:12px;border-radius:10px;margin:8px 0;text-align:right'>"
                    f"<b>You:</b> {msg['message']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='background:#F8D7DA;padding:12px;border-radius:10px;margin:8px 0'>"
                    f"<b>Δ-Zero:</b> {msg['message']}</div>", unsafe_allow_html=True)

                if i == st.session_state.last_bot_idx:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Good", key=f"good_{i}"):
                            agent.update(1.0)
                            agent.log_interaction("", "", agent.last_slot, reward=1.0, feedback="good")
                            st.success("Thanks! I'll use this style more.")
                            st.rerun()
                    with col2:
                        if st.button("Bad", key=f"bad_{i}"):
                            agent.update(0.0)
                            agent.log_interaction("", "", agent.last_slot, reward=0.0, feedback="bad")
                            st.error("Got it. I'll avoid this.")
                            st.rerun()

render_chat()

# ---------- Input Handler ----------
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

def submit_message():
    if st.session_state.user_input.strip():
        st.session_state.pending_message = st.session_state.user_input.strip()
        st.session_state.user_input = ""

st.chat_input(
    "Type your message…",
    key="user_input",
    on_submit=submit_message
)

# Process message
if "pending_message" in st.session_state:
    msg = st.session_state.pending_message
    del st.session_state.pending_message

    with st.spinner("Δ-Zero is thinking..."):
        response, slot = agent.respond(msg, mood)

    agent.log_interaction(msg, response, slot)
    agent.save_state()

    st.session_state.chat_history.append({"sender": "user", "message": msg})
    st.session_state.chat_history.append({"sender": "bot", "message": response})
    st.session_state.last_bot_idx = len(st.session_state.chat_history) - 1

    st.rerun()

# ---------- Reuse Past Messages ----------
if st.checkbox("Reuse past messages"):
    past_inputs = [e["input"] for e in agent.memory[-20:] if e.get("input")]
    if past_inputs:
        selected = st.selectbox("Pick a past message", [""] + past_inputs, key="reuse_select")
        if selected:
            st.session_state.user_input = selected
            st.rerun()

# ---------- Feedback Summary ----------
if st.button("Show Feedback Summary"):
    fb = [e for e in agent.memory if e.get("feedback")]
    if fb:
        df = pd.DataFrame(fb)["feedback"].value_counts().reset_index()
        df.columns = ["Feedback", "Count"]
        fig = px.pie(df, names="Feedback", values="Count", title="User Feedback")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feedback yet.")
