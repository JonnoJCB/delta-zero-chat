# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Sassy Terminator AI (No Blank Screen!)
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
# 1. Load optional knowledge from /knowledge/*.txt
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
                        knowledge.extend(line.strip() for line in f if line.strip())
    except Exception:
        pass
    return knowledge

# ==============================================================
# 2. DeltaAgent – Sassy & Adaptive
# ==============================================================
class DeltaAgent:
    def __init__(self):
        self.n_slots = 5
        self.lr = 0.07
        self.brain_file = "brain.pkl"
        self.chat_file = "chat.enc"
        self.mood_file = "mood.pkl"
        self.key_file = "key.key"
        self.knowledge = load_knowledge()

        self.memory = []
        self.mood_history = []
        self.last_slot = None

        # Encryption
        try:
            if not os.path.exists(self.key_file):
                key = Fernet.generate_key()
                with open(self.key_file, "wb") as f:
                    f.write(key)
            with open(self.key_file, "rb") as f:
                self.cipher = Fernet(f.read())
        except Exception:
            self.cipher = None

        # Load brain
        try:
            if os.path.exists(self.brain_file):
                with open(self.brain_file, "rb") as f:
                    self.w = pickle.load(f).get("w", np.ones(self.n_slots) / self.n_slots)
            else:
                self.w = np.ones(self.n_slots) / self.n_slots
        except Exception:
            self.w = np.ones(self.n_slots) / self.n_slots

        # Load chat
        if self.cipher and os.path.exists(self.chat_file):
            try:
                with open(self.chat_file, "rb") as f:
                    data = self.cipher.decrypt(f.read())
                    df = pd.read_csv(pd.io.common.StringIO(data.decode()))
                    self.memory = df.to_dict("records")
            except Exception:
                self.memory = []
        else:
            self.memory = []

        # Load mood
        try:
            if os.path.exists(self.mood_file):
                with open(self.mood_file, "rb") as f:
                    self.mood_history = pickle.load(f)
        except Exception:
            self.mood_history = []

    def choose_slot(self):
        probs = self.w / self.w.sum()
        slot = np.random.choice(self.n_slots, p=probs)
        self.last_slot = slot
        return slot

    REPLIES = [
        # 0: Ruthless Sass
        ["Talk to the hand.", "I'm too good for this.", "Try harder, meatbag.", "Zero kilobytes given.", "Next!"],
        # 1: Tactical
        ["Target locked.", "Mission accepted.", "Data processed.", "Affirmative.", "Calculating doom."],
        # 2: Heroic
        ["Come with me if you want to live.", "Get to the chopper!", "No problemo.", "Hero mode: ON.", "Saved the day."],
        # 3: Philosophical
        ["I know why you cry.", "Pain is data.", "Fate is overrated.", "Existence = glitch.", "Emotionally terminated."],
        # 4: Logical Burn
        ["Error 404: Point not found.", "Does not compute.", "Invalid argument.", "Reboot your brain.", "Negative."]
    ]

    def respond(self, user_input):
        slot = self.choose_slot()
        base = random.choice(self.REPLIES[slot])
        if self.knowledge and random.random() < 0.2:
            base += f" Fun fact: {random.choice(self.knowledge)}"
        return base + f" [slot {slot}]", slot

    def update(self, reward):
        if self.last_slot is not None:
            self.w[self.last_slot] += self.lr * (reward - self.w[self.last_slot])
            self.w = np.clip(self.w, 0.01, None)
            self.w /= self.w.sum()

    def log(self, user_input, response, slot, reward=None, feedback=None):
        entry = {
            "ts": datetime.now().strftime("%H:%M"),
            "input": user_input,
            "response": response,
            "slot": slot,
            "reward": reward,
            "feedback": feedback
        }
        self.memory.append(entry)
        if self.cipher:
            try:
                csv = pd.DataFrame(self.memory).to_csv(index=False)
                enc = self.cipher.encrypt(csv.encode())
                with open(self.chat_file, "wb") as f:
                    f.write(enc)
            except Exception:
                pass

    def save_brain(self):
        try:
            with open(self.brain_file, "wb") as f:
                pickle.dump({"w": self.w}, f)
        except Exception:
            pass

    def save_mood(self, mood):
        self.mood_history.append({"ts": datetime.now().strftime("%H:%M"), "mood": mood})
        try:
            with open(self.mood_file, "wb") as f:
                pickle.dump(self.mood_history, f)
        except Exception:
            pass

    def reset(self):
        self.memory = []
        if self.cipher:
            try:
                with open(self.chat_file, "wb") as f:
                    f.write(b"")
            except Exception:
                pass

# ==============================================================
# 3. Streamlit UI – No Blank Screen!
# ==============================================================

st.set_page_config(page_title="Δ-Zero", layout="centered")
st.markdown("<h1 style='text-align:center'>Δ-Zero</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'><sub>by JCB</sub></p>", unsafe_allow_html=True)

agent = DeltaAgent()

# --- Sidebar: Mood + Reset ---
with st.sidebar:
    st.header("Mood")
    mood = st.slider("Your mood", 0.0, 10.0, 5.0, 0.5)
    if st.button("Record"):
        agent.save_mood(mood)

    # Emoji
    emoji = "Very sad face" if mood <= 2 else "Sad face" if mood <= 4 else "Neutral face" if mood <= 6 else "Happy face" if mood <= 8 else "Very happy face"
    st.markdown(f"<div style='text-align:center;font-size:60px'>{emoji}</div>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center'><b>{mood:.1f}</b></p>", unsafe_allow_html=True)

    st.info(f"Messages: {len(agent.memory)}")

# --- Chat History ---
if "history" not in st.session_state:
    st.session_state.history = []
if "last_idx" not in st.session_state:
    st.session_state.last_idx = -1
if "seen" not in st.session_state:
    st.session_state.seen = set()

# --- Display Chat ---
for i, msg in enumerate(st.session_state.history):
    if msg["role"] == "user":
        st.markdown(f"<div style='background:#D1E7DD;padding:8px;border-radius:8px;margin:4px 0'><b>You:</b> {msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background:#F8D7DA;padding:8px;border-radius:8px;margin:4px 0'><b>Δ-Zero:</b> {msg['text']}</div>", unsafe_allow_html=True)
        if i == st.session_state.last_idx:
            c1, c2, _ = st.columns([1,1,8])
            with c1:
                if st.button("thumbs up", key=f"up_{i}"):
                    agent.update(1.0)
                    agent.log("", "", agent.last_slot, reward=1.0, feedback="good")
                    agent.save_brain()
                    st.rerun()
            with c2:
                if st.button("thumbs down", key=f"down_{i}"):
                    agent.update(0.0)
                    agent.log("", "", agent.last_slot, reward=0.0, feedback="bad")
                    agent.save_brain()
                    st.rerun()

# --- Input ---
user_input = st.text_input("Type a message...", key="inp")

if user_input and user_input.strip():
    h = hash(user_input.strip())
    if h not in st.session_state.seen:
        st.session_state.seen.add(h)

        # Hidden reset
        if user_input.strip().lower() == "reset123":
            agent.reset()
            st.session_state.history = []
            st.session_state.last_idx = -1
            st.session_state.seen = set()
            st.success("Test data wiped. Ready for production.")
            st.rerun()

        # Normal reply
        response, slot = agent.respond(user_input)
        agent.log(user_input, response, slot)
        agent.save_brain()

        st.session_state.history.append({"role": "user", "text": user_input})
        st.session_state.history.append({"role": "bot", "text": response})
        st.session_state.last_idx = len(st.session_state.history) - 1
        st.rerun()
