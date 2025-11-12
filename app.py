# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Adaptive AI with Short-Term Context
# by JCB (enhanced by ChatGPT)
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


# ============================================================== #
# 1. Load knowledge
# ============================================================== #
def load_knowledge():
    """Loads .txt files from /knowledge folder"""
    knowledge = []
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
    if os.path.exists(knowledge_dir):
        for filename in os.listdir(knowledge_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(knowledge_dir, filename), "r", encoding="utf-8") as f:
                    knowledge.extend([line.strip() for line in f if line.strip()])
    return knowledge


# ============================================================== #
# 2. DeltaAgent – Adaptive Chat Logic with Short-Term Memory
# ============================================================== #
class DeltaAgent:
    def __init__(
        self,
        n_slots=5,
        lr=0.07,
        brain_file="global_brain.pkl",
        data_file="chat_log.enc",
        key_file="secret.key",
        short_term_len=5,
    ):
        self.n_slots = n_slots
        self.lr = lr
        self.brain_file = brain_file
        self.data_file = data_file
        self.key_file = key_file
        self.knowledge = load_knowledge()
        self.memory = []  # Long-term memory
        self.short_term_len = short_term_len  # how many past messages to recall
        self.last_slot = None

        # Encryption setup
        if not os.path.exists(key_file):
            with open(key_file, "wb") as f:
                f.write(Fernet.generate_key())

        with open(key_file, "rb") as f:
            self.cipher = Fernet(f.read())

        # Load weights
        if os.path.exists(brain_file):
            with open(brain_file, "rb") as f:
                saved = pickle.load(f)
                self.w = saved.get("w", np.ones(n_slots) / n_slots)
        else:
            self.w = np.ones(n_slots) / n_slots

        # Load chat history (long-term)
        if os.path.exists(data_file):
            try:
                with open(data_file, "rb") as f:
                    encrypted = f.read()
                if encrypted:
                    decrypted = self.cipher.decrypt(encrypted)
                    df = pd.read_csv(pd.io.common.StringIO(decrypted.decode()))
                    self.memory = df.to_dict("records")
            except Exception:
                st.warning("Could not decrypt chat log. Starting fresh.")
                self.memory = []
        else:
            self._save_encrypted_df(pd.DataFrame(columns=[
                "timestamp", "user", "input", "response", "slot",
                "reward", "feedback", "fb_text"
            ]))

    # ========================== Core Logic ========================== #
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

    def get_short_term_context(self):
        """Get the last N user messages for short-term awareness"""
        recent = [e["input"] for e in self.memory[-self.short_term_len:] if e["input"]]
        if recent:
            return " | ".join(recent[-self.short_term_len:])
        return ""

    def generate_response(self, user_input, slot):
        """Generate a response influenced by style, context, and knowledge"""
        base = random.choice(self.REPLIES[slot])

        # Blend short-term context for more coherent replies
        context = self.get_short_term_context()
        if context and random.random() < 0.4:
            base += f" Considering what we discussed earlier ({context.split('|')[-1].strip()}), I'd say that's interesting."

        # Occasionally pull a random fact
        if self.knowledge and random.random() < 0.25:
            fact = random.choice(self.knowledge)
            base += f" Fun fact: {fact}"

        return base + f" [slot {slot}]"

    def respond(self, user_input):
        slot = self.choose_slot()
        response = self.generate_response(user_input, slot)
        return response, slot

    def update(self, reward):
        """Reinforce or weaken specific style slots"""
        if self.last_slot is not None:
            self.w[self.last_slot] += self.lr * (reward - self.w[self.last_slot])
            self.w = np.clip(self.w, 0.01, None)
            self.w /= self.w.sum()
            self.save_state()

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


# ============================================================== #
# 3. Streamlit Interface (Cleaned + Short-Term Aware)
# ============================================================== #

st.set_page_config(page_title="Δ-Zero Chat", layout="centered")
st.title("Δ-Zero Chat 🧠")
st.caption("Adaptive AI with short-term awareness & learning feedback")

agent = DeltaAgent()

# --- Sidebar info ---
st.sidebar.title("AI Info")
st.sidebar.metric("Stored Chats", len(agent.memory))
if agent.knowledge:
    st.sidebar.success(f"{len(agent.knowledge)} knowledge facts loaded")

weights = agent.w / agent.w.sum()
slot_labels = ["Curious", "Calm", "Engaging", "Empathetic", "Analytical"]
conf_df = pd.DataFrame({"Style": slot_labels, "Confidence": weights})
conf_fig = px.bar(conf_df, x="Style", y="Confidence",
                  title="AI Style Confidence", color="Confidence",
                  color_continuous_scale="Blues")
st.sidebar.plotly_chart(conf_fig, use_container_width=True)

# --- Chat State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_bot_idx" not in st.session_state:
    st.session_state.last_bot_idx = -1

# --- Display Chat ---
def display_chat():
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["sender"] == "user":
                st.markdown(
                    f"<div style='background:#D1E7DD;padding:10px;border-radius:8px;margin:5px 0'>"
                    f"<b>You:</b> {msg['message']}</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='background:#F8D7DA;padding:10px;border-radius:8px;margin:5px 0'>"
                    f"<b>Δ-Zero:</b> {msg['message']}</div>",
                    unsafe_allow_html=True)

                # Feedback buttons for the latest bot response only
                if i == st.session_state.last_bot_idx:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 Good", key=f"good_{i}"):
                            agent.update(1.0)
                            agent.log_interaction("user", "", "", agent.last_slot,
                                                  reward=1.0, feedback="good")
                            st.toast("Learning: favoring this style", icon="✅")
                            st.rerun()
                    with col2:
                        if st.button("👎 Bad", key=f"bad_{i}"):
                            agent.update(0.0)
                            agent.log_interaction("user", "", "", agent.last_slot,
                                                  reward=0.0, feedback="bad")
                            st.toast("Learning: avoiding this style", icon="⚠️")
                            st.rerun()

display_chat()

# --- Input Box ---
user_input = st.chat_input("Type your message...")

if user_input:
    response, slot = agent.respond(user_input)
    agent.log_interaction("user", user_input, response, slot)
    agent.save_state()

    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    st.session_state.chat_history.append({"sender": "bot", "message": response})
    st.session_state.last_bot_idx = len(st.session_state.chat_history) - 1
    st.rerun()

# --- Optional: Feedback Summary ---
with st.expander("Feedback Summary"):
    fb = [e for e in agent.memory if e["feedback"]]
    if fb:
        df = pd.DataFrame(fb)["feedback"].value_counts().reset_index()
        df.columns = ["Feedback", "Count"]
        fig = px.pie(df, names="Feedback", values="Count", title="Feedback Ratio")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feedback yet.")
