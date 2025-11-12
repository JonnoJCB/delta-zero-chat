# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Smart AI with MOOD CHART, Learning & Encrypted Memory by JCB
# Enhanced with conversation UI, colors, and MSN sound
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

# --------------------------------------------------------------
# 1. DeltaAgent – Smart + Mood Tracking + Encrypted Learning
# --------------------------------------------------------------
class DeltaAgent:
    def __init__(
        self,
        n_slots=5,
        lr=0.07,
        brain_file="global_brain.pkl",
        data_file="chat_log.enc",
        mood_file="mood_history.pkl",
        key_file="secret.key"
    ):
        self.n_slots = n_slots
        self.lr = lr
        self.brain_file = brain_file
        self.data_file = data_file
        self.mood_file = mood_file
        self.key_file = key_file

        self.memory = []
        self.mood_history = []
        self.prev_vec = None
        self.last_slot = None

        # Encryption setup
        if not os.path.exists(self.key_file):
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(key)
        with open(self.key_file, "rb") as f:
            self.cipher = Fernet(f.read())

        # Load brain
        if os.path.exists(brain_file):
            with open(brain_file, "rb") as f:
                saved = pickle.load(f)
                self.w = saved.get("w", np.ones(n_slots) / n_slots)
        else:
            self.w = np.ones(n_slots) / n_slots

        # Load encrypted chat memory
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "rb") as f:
                    encrypted_data = f.read()
                    if encrypted_data:
                        decrypted = self.cipher.decrypt(encrypted_data)
                        df = pd.read_csv(pd.io.common.StringIO(decrypted.decode()))
                        self.memory = df.to_dict(orient="records")
                    else:
                        self.memory = []
            except Exception:
                st.warning("⚠️ Could not decrypt chat log. Starting fresh.")
                self.memory = []
        else:
            self._save_encrypted_df(pd.DataFrame(columns=["timestamp","user","input","response","slot","reward","feedback","fb_text"]))

        # Load mood history
        if os.path.exists(mood_file):
            with open(mood_file, "rb") as f:
                self.mood_history = pickle.load(f)
        else:
            self.mood_history = []

    def embed(self, text):
        vec = np.zeros(26)
        for c in text.lower():
            if c.isalpha():
                vec[ord(c)-97] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm>0 else vec

    def choose_slot(self):
        probs = self.w / self.w.sum()
        slot = np.random.choice(range(self.n_slots), p=probs)
        self.last_slot = slot
        return slot

    def generate_response(self, user_input, slot):
        replies = [
            "That's quite interesting!",
            "I see what you mean.",
            "Tell me more about that.",
            "How does that make you feel?",
            "Let's explore that thought."
        ]
        return random.choice(replies) + f" [slot {slot}]"

    def respond(self, user_input):
        vec = self.embed(user_input)
        slot = self.choose_slot()
        response = self.generate_response(user_input, slot)
        self.prev_vec = vec
        return response, slot

    def update(self, reward):
        if self.last_slot is not None:
            self.w[self.last_slot] += self.lr * (reward - self.w[self.last_slot])
            self.w = np.clip(self.w, 0.01, None)
            self.w /= self.w.sum()

    def log_interaction(self, user, user_input, response, slot, reward=None, feedback=None, fb_text=None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = {
            "timestamp": ts,
            "user": user,
            "input": user_input,
            "response": response,
            "slot": slot,
            "reward": reward,
            "feedback": feedback,
            "fb_text": fb_text
        }
        self.memory.append(new_entry)
        df = pd.DataFrame(self.memory)
        self._save_encrypted_df(df)

    def _save_encrypted_df(self, df):
        csv_data = df.to_csv(index=False)
        encrypted = self.cipher.encrypt(csv_data.encode())
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

# --------------------------------------------------------------
# 2. Streamlit Interface with Conversation Flow
# --------------------------------------------------------------
st.set_page_config(page_title="Δ-Zero Chat", layout="wide")
st.title("Δ-Zero Chat 🤖 – Encrypted Shared Learning AI")

agent = DeltaAgent()

# Sidebar: Mood Tracker
st.sidebar.header("🧠 Mood Tracker")
mood = st.sidebar.slider("Your current mood", 0.0, 10.0, 5.0, 0.5)
if st.sidebar.button("Record Mood"):
    agent.update_mood(mood)
    st.sidebar.success("Mood recorded!")

# Mood Chart
if agent.mood_history:
    mood_df = pd.DataFrame(agent.mood_history)
    fig = px.line(mood_df, x="timestamp", y="mood", title="Mood Over Time", markers=True)
    st.sidebar.plotly_chart(fig, use_container_width=True)

# Session state for conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages with colors
def display_chat():
    for entry in st.session_state.chat_history:
        if entry["sender"] == "user":
            st.markdown(f"<div style='background-color:#D1E7DD;padding:8px;border-radius:5px;margin-bottom:5px'><b>You:</b> {entry['message']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#F8D7DA;padding:8px;border-radius:5px;margin-bottom:5px'><b>Δ-Zero:</b> {entry['message']}</div>", unsafe_allow_html=True)

# Input area
user_input = st.text_input("Type your message here...", key="user_input")

if user_input:
    # Generate response
    response, slot = agent.respond(user_input)
    agent.log_interaction("user", user_input, response, slot)
    agent.save_state()

    # Append messages to chat
    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    st.session_state.chat_history.append({"sender": "bot", "message": response})

    # Play MSN-style ping sound (local file)
    try:
        with open("msn_ping.mp3", "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")
    except FileNotFoundError:
        st.warning("⚠️ MSN ping sound not found! Place 'msn_ping.mp3' in the app folder.")

    # Clear input
    st.session_state.user_input = ""

# Display chat history
display_chat()

# Optional: dropdown to reuse past messages
if st.checkbox("Use previous messages"):
    past_msgs = [entry["input"] for entry in agent.memory[-20:]]  # last 20 messages
    selected_msg = st.selectbox("Select a past message", [""] + past_msgs)
    if selected_msg:
        st.session_state.user_input = selected_msg

# Show total encrypted chat count
st.sidebar.info(f"Total encrypted chats stored: {len(agent.memory)}")
