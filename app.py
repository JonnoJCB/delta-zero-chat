# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Global-Learning Personality Bot (v2.0)
# Run locally:   py -3.11 -m streamlit run app.py
# Deploy:        Push to GitHub → Streamlit Community Cloud
# --------------------------------------------------------------

import streamlit as st
import numpy as np
import os
import pickle
import pandas as pd
from datetime import datetime

# ==============================================================
# 1. DeltaAgent – Smarter, Saves Data, No Slot Spam
# ==============================================================
class DeltaAgent:
    def __init__(self, n_slots=5, lr=0.05, brain_file="global_brain.pkl", data_file="chat_log.csv"):
        self.n_slots = n_slots
        self.lr = lr
        self.brain_file = brain_file
        self.data_file = data_file
        self.self_slot = None
        self.prev_vec = None

        # Load brain
        if os.path.exists(brain_file):
            with open(brain_file, "rb") as f:
                saved = pickle.load(f)
                self.w = saved.get("w", np.ones(n_slots) / n_slots)
        else:
            self.w = np.ones(n_slots) / n_slots

        # Init log
        if not os.path.exists(data_file):
            pd.DataFrame(columns=["timestamp", "user", "input", "response", "slot", "reward"]).to_csv(data_file, index=False)

    def embed(self, text):
        vec = np.zeros(26)
        for c in text.lower():
            if c.isalpha():
                vec[ord(c) - 97] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def choose_slot(self):
        probs = self.w / self.w.sum()
        return np.random.choice(range(self.n_slots), p=probs)

    def reinforce(self, slot, reward):
        self.w[slot] += self.lr * reward
        self.w = np.maximum(self.w, 1e-6)
        self.w /= self.w.sum()

    def respond(self, user_text, user_id="user"):
        vec = self.embed(user_text)

        if self.prev_vec is None:
            self.prev_vec = vec
            response = "Hey! I'm Δ-Zero. I learn from you. What's on your mind?"
            reward = 0.5
        else:
            slot = self.choose_slot()
            style = ["curious", "playful", "deep", "sarcastic", "poetic"][slot]

            # Smarter responses based on input length & keywords
            if len(user_text) < 10:
                response = random.choice([
                    "Short and sweet!", "Tell me more!", "Hmm?", "Go on..."
                ])
            elif any(word in user_text.lower() for word in ["i ", "me ", "my ", "myself"]):
                response = random.choice([
                    "That says a lot about you.", "I feel that.", "You're opening up!",
                    "Keep going — I’m listening."
                ])
                reward = 1.5
            else:
                responses = {
                    "curious": ["Really? How?", "What makes you say that?", "Interesting..."],
                    "playful": ["Ooooh!", "No way!", "You're fun!"],
                    "deep": ["The universe is wild.", "That’s profound.", "I wonder..."],
                    "sarcastic": ["Wow. Shocking.", "Tell me something new.", "As if I care... (jk!)"],
                    "poetic": ["Like whispers in the wind...", "Your words paint stars.", "A thought in the void."]
                }
                response = random.choice(responses[style])
                reward = 0.8

            self.reinforce(slot, reward)

        # Save to global brain
        with open(self.brain_file, "wb") as f:
            pickle.dump({"w": self.w}, f)

        # Log interaction
        log_df = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_id,
            "input": user_text,
            "response": response,
            "slot": slot if 'slot' in locals() else 0,
            "reward": reward if 'reward' in locals() else 0.5
        }])
        log_df.to_csv(self.data_file, mode='a', header=False, index=False)

        self.prev_vec = vec
        self.self_slot = int(np.argmax(self.w))
        return response


# ==============================================================
# 2. Streamlit UI – Clean, Guided, No Slot Boxes
# ==============================================================

st.set_page_config(page_title="Δ-Zero Chat", page_icon="robot", layout="centered")
st.title("Δ-Zero Chat")
st.caption("A bot that learns from *you*. Every chat makes it smarter.")

# --- User ID ---
user_id = st.text_input("Your Name/ID (so I remember you)", placeholder="e.g., Alex123", key="user_id_input")

if not user_id:
    st.info("Enter a name to start chatting!")
    st.stop()

# --- Init Agent ---
agent_key = f"agent_{user_id}"
if agent_key not in st.session_state:
    st.session_state[agent_key] = DeltaAgent(brain_file=f"brain_{user_id}.pkl", data_file="chat_log.csv")

agent = st.session_state[agent_key]

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User Input ---
if prompt := st.chat_input("Talk to Δ-Zero..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = agent.respond(prompt, user_id=user_id)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# --- Sidebar: Mood + Stats + Delta ---
with st.sidebar:
    st.header("Bot Status")
    if agent.self_slot is not None:
        mood = ["curious", "playful", "deep", "sarcastic", "poetic"][agent.self_slot]
        strength = agent.w[agent.self_slot]
        st.metric("Current Mood", mood.title(), f"{strength:.1%}")

    if st.button("Reset My Brain"):
        brain_path = f"brain_{user_id}.pkl"
        if os.path.exists(brain_path):
            os.remove(brain_path)
        st.session_state[agent_key] = DeltaAgent(brain_file=brain_path, data_file="chat_log.csv")
        st.session_state.messages = []
        st.success("Brain reset!")
        st.experimental_rerun()

    st.divider()
    st.caption("**Data Saved:** `chat_log.csv` (open in Excel)")

    if st.button("Download Chat Log"):
        df = pd.read_csv("chat_log.csv")
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "delta_chat_log.csv", "text/csv")