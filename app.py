# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Production-Ready with Hidden Reset
# --------------------------------------------------------------

import streamlit as st
import numpy as np
import os
import pickle
import pandas as pd
import random
from datetime import datetime
from cryptography.fernet import Fernet

# ==============================================================
# 1. Load knowledge from /knowledge/*.txt
# ==============================================================
def load_knowledge():
    knowledge = []
    try:
        knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
        if os.path.exists(knowledge_dir):
            for filename in os.listdir(knowledge_dir):
                if filename.endswith(".txt"):
                    path = os.path.join(knowledge_dir, filename)
                    with open(path, "r", encoding="utf-8") as f:
                        knowledge.extend([line.strip() for line in f if line.strip()])
    except Exception:
        pass
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

        self.memory = []
        self.mood_history = []
        self.last_slot = None

        # Encryption
        try:
            if not os.path.exists(key_file):
                key = Fernet.generate_key()
                with open(key_file, "wb") as f:
                    f.write(key)
            with open(key_file, "rb") as f:
                self.cipher = Fernet(f.read())
        except Exception:
            self.cipher = None

        # Load brain
        try:
            if os.path.exists(brain_file):
                with open(brain_file, "rb") as f:
                    saved = pickle.load(f)
                    self.w = saved.get("w", np.ones(n_slots) / n_slots)
            else:
                self.w = np.ones(n_slots) / n_slots
        except Exception:
            self.w = np.ones(n_slots) / n_slots

        # Load encrypted chat
        if self.cipher and os.path.exists(data_file):
            try:
                with open(data_file, "rb") as f:
                    encrypted = f.read()
                    if encrypted:
                        decrypted = self.cipher.decrypt(encrypted)
                        df = pd.read_csv(pd.io.common.StringIO(decrypted.decode()))
                        self.memory = df.to_dict("records")
            except Exception:
                self.memory = []
        else:
            self._save_encrypted_df(pd.DataFrame(columns=[
                "timestamp","user","input","response","slot",
                "reward","feedback","fb_text"
            ]))

        # Load mood
        try:
            if os.path.exists(mood_file):
                with open(mood_file, "rb") as f:
                    self.mood_history = pickle.load(f)
        except Exception:
            self.mood_history = []

    def choose_slot(self):
        probs = self.w / self.w.sum()
        slot = np.random.choice(range(self.n_slots), p=probs)
        self.last_slot = slot
        return slot

    REPLIES = [
        ["Wow, fascinating!", "I'm intrigued!", "That's wild!"],
        ["I understand.", "That makes sense.", "Clear as day."],
        ["Tell me more!", "Keep going!", "Don't stop now!"],
        ["How do you feel about that?", "Why do you think so?", "That's deep."],
        ["Let's analyze this.", "Interesting angle.", "Break it down."]
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
        if self.cipher:
            self._save_encrypted_df(pd.DataFrame(self.memory))

    def _save_encrypted_df(self, df):
        try:
            csv = df.to_csv(index=False)
            encrypted = self.cipher.encrypt(csv.encode())
            with open(self.data_file, "wb") as f:
                f.write(encrypted)
        except Exception:
            pass

    def save_state(self):
        try:
            with open(self.brain_file, "wb") as f:
                pickle.dump({"w": self.w}, f)
            with open(self.mood_file, "wb") as f:
                pickle.dump(self.mood_history, f)
        except Exception:
            pass

    def update_mood(self, mood_value):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.mood_history.append({"timestamp": ts, "mood": mood_value})
        self.save_state()

    # ADMIN: Hidden reset command
    def admin_reset(self):
        self.memory = []
        if self.cipher:
            self._save_encrypted_df(pd.DataFrame(columns=[
                "timestamp","user","input","response","slot",
                "reward","feedback","fb_text"
            ]))

# ==============================================================
# 3. Streamlit UI – Clean & Production-Ready
# ==============================================================

st.set_page_config(page_title="Δ-Zero Chat", layout="centered")
st.title("Δ-Zero Chat")
st.markdown("<sub>by JCB</sub>", unsafe_allow_html=True)

agent = DeltaAgent()

# ---------- Sidebar ----------
st.sidebar.header("Mood")

# Mood slider
mood = st.sidebar.slider("Your mood", 0.0, 10.0, 5.0, 0.5, key="mood_slider")
if st.sidebar.button("Record"):
    agent.update_mood(mood)

# Emoji only
def mood_emoji(val):
    if val <= 2:   return "Very sad face"
    elif val <= 4: return "Sad face"
    elif val <= 6: return "Neutral face"
    elif val <= 8: return "Happy face"
    else:          return "Very happy face"

st.sidebar.markdown(
    f"<div style='text-align:center;font-size:60px;margin:10px 0'>{mood_emoji(mood)}</div>",
    unsafe_allow_html=True
)
st.sidebar.markdown(f"<p style='text-align:center;'><b>{mood:.1f}</b></p>", unsafe_allow_html=True)

st.sidebar.info(f"Messages: {len(agent.memory)}")

# ---------- Chat ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_bot_idx" not in st.session_state:
    st.session_state.last_bot_idx = -1
if "processed_inputs" not in st.session_state:
    st.session_state.processed_inputs = set()

def display_chat():
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["sender"] == "user":
            st.markdown(
                f"<div style='background:#D1E7DD;padding:10px;border-radius:8px;margin:5px 0'>"
                f"<
