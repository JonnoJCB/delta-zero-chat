
# app.py
# Δ-Zero Chat v3.1 – by Jonno
# --------------------------------------------------------------

import streamlit as st
import numpy as np
import os
import pickle
import pandas as pd
import random
from datetime import datetime

# ==============================================================
# 1. DeltaAgent – Safer Logging, No Crashes
# ==============================================================
class DeltaAgent:
    def __init__(self, n_slots=5, lr=0.05, brain_file="global_brain.pkl", data_file="chat_log.csv"):
        self.n_slots = n_slots
        self.lr = lr
        self.brain_file = brain_file
        self.data_file = data_file
        self.self_slot = None
        self.prev_vec = None
        self.last_slot = None

        # Load brain
        if os.path.exists(brain_file):
            with open(brain_file, "rb") as f:
                saved = pickle.load(f)
                self.w = saved.get("w", np.ones(n_slots) / n_slots)
        else:
            self.w = np.ones(n_slots) / n_slots

        # Init log safely
        try:
            if not os.path.exists(data_file):
                pd.DataFrame(columns=["timestamp", "user", "input", "response", "slot", "reward", "feedback", "fb_text"]).to_csv(data_file, index=False)
        except Exception as e:
            st.error(f"Could not init log: {e}")
            self.data_file = None  # Disable logging if fails

    def embed(self, text):
        vec = np.zeros(26)
        for c in text.lower():
            if c.isalpha():
                vec[ord(c) - 97] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def choose_slot(self):
        probs = self.w / self.w.sum()
        slot = np.random.choice(range(self.n_slots), p=probs)
        self.last_slot = slot
        return slot

    def reinforce(self, slot, reward):
        self.w[slot] += self.lr * reward
        self.w = np.maximum(self.w, 1e-6)
        self.w /= self.w.sum()

    def log_interaction(self, user_id, user_text, response, slot, reward, feedback="", fb_text=""):
        if self.data_file is None:
            return  # Skip if logging disabled
        try:
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": user_id,
                "input": user_text,
                "response": response,
                "slot": slot,
                "reward": reward,
                "feedback": feedback,
                "fb_text": fb_text
            }
            pd.DataFrame([log_entry]).to_csv(self.data_file, mode='a', header=False, index=False)
        except Exception as e:
            st.error(f"Log failed: {e}")  # Show but don't crash

    def respond(self, user_text, user_id="user"):
        vec = self.embed(user_text)

        if self.prev_vec is None:
            self.prev_vec = vec
            response = "Hey! I'm Δ-Zero. I learn from *you*. Say anything."
            reward = 0.5
            slot = 0
        else:
            slot = self.choose_slot()
            style = ["curious", "playful", "deep", "sarcastic", "poetic"][slot]

            if len(user_text) < 8:
                response = random.choice(["Hmm?", "Tell me more!", "Go on..."])
                reward = 0.6
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

        # Save brain safely
        try:
            with open(self.brain_file, "wb") as f:
                pickle.dump({"w": self.w}, f)
        except Exception as e:
            st.error(f"Brain save failed: {e}")

        self.prev_vec = vec
        self.self_slot = int(np.argmax(self.w))

        # Log AFTER response
        self.log_interaction(user_id, user_text, response, slot, reward)

        return response, slot, reward


# ==============================================================
# 2. Streamlit UI – Feedback Outside Loop, No Crashes
# ==============================================================

st.set_page_config(page_title="Δ-Zero Chat", page_icon="robot", layout="centered")
st.title("Δ-Zero Chat")
st.caption("Talk → Bot learns → You give feedback → It gets smarter")

# --- User ID ---
user_id = st.text_input("Your Name/ID", placeholder="e.g., Alex123", key="user_id")

if not user_id:
    st.info("Enter a name to start!")
    st.stop()

# --- Init Agent ---
agent_key = f"agent_{user_id}"
if agent_key not in st.session_state:
    st.session_state[agent_key] = DeltaAgent(brain_file=f"brain_{user_id}.pkl", data_file="chat_log.csv")

agent = st.session_state[agent_key]

# --- Chat History (No Logging Here — Safe) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Feedback ONLY on LAST bot message (avoids loop issues)
        if msg["role"] == "assistant" and "feedback" not in msg:
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("👍 Good", key=f"good_{i}"):
                    agent.reinforce(agent.last_slot, 1.0)
                    msg["feedback"] = "good"  # Mark as done
                    st.success("Thanks! I’ll do more of that.")
                    st.rerun()  # Refresh to update
            with col1:
                if st.button("👎 Bad", key=f"bad_{i}"):
                    agent.reinforce(agent.last_slot, -1.0)
                    msg["feedback"] = "bad"
                    st.error("Got it. I’ll try less of that.")
                    st.rerun()
            with col2:
                fb_text = st.text_input("Why?", key=f"fbtext_{i}", placeholder="e.g., too sarcastic")
                if fb_text and st.button("Send Feedback", key=f"sendfb_{i}"):
                    msg["fb_text"] = fb_text
                    st.success("Feedback noted!")
                    st.rerun()

# --- User Input ---
if prompt := st.chat_input("Talk to Δ-Zero..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Δ-Zero is thinking..."):
            response, slot, reward = agent.respond(prompt, user_id=user_id)
            st.markdown(response)

    # Add to history AFTER display
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response, "feedback": ""})

# --- Sidebar: Mood + Delta + Export ---
with st.sidebar:
    st.header("Bot Brain")
    if agent.self_slot is not None:
        mood = ["curious", "playful", "deep", "sarcastic", "poetic"][agent.self_slot]
        st.metric("Mood", mood.title(), f"{agent.w[agent.self_slot]:.1%}")

    if st.button("Reset My Brain"):
        brain_path = f"brain_{user_id}.pkl"
        if os.path.exists(brain_path):
            os.remove(brain_path)
        st.session_state[agent_key] = DeltaAgent(brain_file=brain_path, data_file="chat_log.csv")
        st.session_state.messages = []
        st.success("Reset!")
        st.rerun()

    st.divider()
    st.subheader("Delta Analysis")
    try:
        if os.path.exists("chat_log.csv"):
            df = pd.read_csv("chat_log.csv")
            user_df = df[df["user"] == user_id]
            if len(user_df) > 1:
                delta_reward = user_df["reward"].diff().iloc[-1]
                st.metric("Reward Change (Δ)", f"{delta_reward:+.2f}")
    except Exception as e:
        st.error(f"Analysis failed: {e}")

    if st.button("Download Full Log"):
        try:
            if os.path.exists("chat_log.csv"):
                csv = pd.read_csv("chat_log.csv").to_csv(index=False).encode()
                st.download_button("Download CSV", csv, "delta_zero_log.csv", "text/csv")
            else:
                st.info("No data yet.")
        except Exception as e:
            st.error(f"Download failed: {e}")
