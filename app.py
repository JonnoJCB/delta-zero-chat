# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Smart, Learning, Feedback-Driven AI Bot
# by JCB
# --------------------------------------------------------------

import streamlit as st
import numpy as np
import os
import pickle
import pandas as pd
import random
from datetime import datetime

# ==============================================================
# 1. DeltaAgent – Super Smart Responses + Safe Logging
# ==============================================================
class DeltaAgent:
    def __init__(self, n_slots=5, lr=0.07, brain_file="global_brain.pkl", data_file="chat_log.csv"):
        self.n_slots = n_slots
        self.lr = lr
        self.brain_file = brain_file
        self.data_file = data_file
        self.self_slot = None
        self.prev_vec = None
        self.last_slot = None
        self.memory = []  # Short-term memory of last 3 turns

        # Load brain
        if os.path.exists(brain_file):
            with open(brain_file, "rb") as f:
                saved = pickle.load(f)
                self.w = saved.get("w", np.ones(n_slots) / n_slots)
        else:
            self.w = np.ones(n_slots) / n_slots

        # Init log safely
        if not os.path.exists(data_file):
            df = pd.DataFrame(columns=["timestamp", "user", "input", "response", "slot", "reward", "feedback", "fb_text"])
            df.to_csv(data_file, index=False, quoting=1)

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
            df = pd.DataFrame([log_entry])
            df.to_csv(self.data_file, mode='a', header=False, index=False, quoting=1)
        except:
            pass  # Silent fail — app never crashes

    def respond(self, user_text, user_id="Anonymous"):
        vec = self.embed(user_text)
        self.memory.append(user_text.lower())
        if len(self.memory) > 3:
            self.memory.pop(0)

        # First message
        if self.prev_vec is None:
            self.prev_vec = vec
            response = random.choice([
                "Hey there! I'm Δ-Zero — your AI that *learns from you*. What's up?",
                "Hi! I'm Δ-Zero. I evolve with every chat. Ready when you are!",
                "Greetings! I'm Δ-Zero, the bot that gets smarter with you. Let's talk!"
            ])
            reward = 0.7
            slot = 0
        else:
            slot = self.choose_slot()
            style = ["curious", "playful", "deep", "sarcastic", "poetic"][slot]

            # === SMART RESPONSE ENGINE ===
            lower = user_text.lower()
            is_question = "?" in user_text
            is_personal = any(x in lower for x in ["i ", "me ", "my ", "myself", "i'm", "i've"])
            is_short = len(user_text.strip()) < 12
            mentions_feeling = any(x in lower for x in ["feel", "happy", "sad", "angry", "tired", "excited"])

            if is_short:
                response = random.choice([
                    "Short and sweet — I like it!", "Tell me more!", "Go on...", "I'm listening!"
                ])
                reward = 0.6
            elif is_question:
                response = random.choice([
                    "Great question!", "Let me think...", "Hmm, good one!", "Ooh, intriguing!"
                ]) + " " + random.choice([
                    "What do you think?", "Why do you ask?", "That’s deep — tell me more!"
                ])
                reward = 1.2
            elif is_personal and mentions_feeling:
                feeling = next((w for w in ["happy", "sad", "angry", "tired", "excited"] if w in lower), "good")
                response = random.choice([
                    f"I hear you — feeling {feeling} is real.",
                    f"That {feeling} vibe? I get it.",
                    f"You're being open — I respect that."
                ])
                reward = 1.8
            elif is_personal:
                response = random.choice([
                    "You're sharing — I love that.", "This says a lot about you.", "Keep opening up!",
                    "I’m here for it.", "You’re not alone in that."
                ])
                reward = 1.4
            else:
                # Style-based with flair
                responses = {
                    "curious": [
                        "Wait — *really*?", "How does that work?", "What happens next?",
                        "I need details!", "You’ve got my full attention."
                    ],
                    "playful": [
                        "No way!", "You’re wild!", "Hehe, keep going!", "This is gold!",
                        "I’m dying over here!"
                    ],
                    "deep": [
                        "That’s profound.", "The universe is strange, isn’t it?",
                        "We’re all just stardust.", "This hits deep.", "I feel that in my circuits."
                    ],
                    "sarcastic": [
                        "Wow. Groundbreaking.", "Tell me something I don’t know.",
                        "As if I care... (kidding!)", "Shocking. Truly.", "Revolutionary."
                    ],
                    "poetic": [
                        "Like whispers in the wind...", "Your words paint galaxies.",
                        "A thought in the void.", "Stars align in your voice.", "I am but a shadow of meaning."
                    ]
                }
                response = random.choice(responses[style])
                reward = 0.9

            # Bonus: Reference past memory
            if len(self.memory) > 1 and random.random() < 0.3:
                past = self.memory[-2]
                response += f" (Like when you said '{past[:20]}...' — still thinking about that.)"

            self.reinforce(slot, reward)

        # Save brain
        try:
            with open(self.brain_file, "wb") as f:
                pickle.dump({"w": self.w}, f)
        except:
            pass

        self.prev_vec = vec
        self.self_slot = int(np.argmax(self.w))

        # Log
        self.log_interaction(user_id, user_text, response, slot, reward)

        return response, slot, reward


# ==============================================================
# 2. Streamlit UI – Clean, Smart, Safe
# ==============================================================

st.set_page_config(page_title="Δ-Zero Chat", page_icon="robot", layout="centered")
st.title("Δ-Zero Chat")
st.caption("An AI that *learns from you* — every chat makes it smarter.  \n<span style='font-size:0.8em; color:gray;'>by JCB</span>", unsafe_allow_html=True)

# --- Optional Name ---
user_id = st.text_input("Your Name (optional)", placeholder="e.g., Alex123", key="user_id", help="Leave blank to stay anonymous")
if not user_id:
    user_id = "Anonymous"

# --- Init Agent ---
agent_key = f"agent_{user_id}"
if agent_key not in st.session_state:
    st.session_state[agent_key] = DeltaAgent(brain_file=f"brain_{user_id}.pkl", data_file="chat_log.csv")

agent = st.session_state[agent_key]

# --- Safe CSV Load ---
def safe_load_csv():
    if not os.path.exists("chat_log.csv"):
        return pd.DataFrame()
    try:
        return pd.read_csv("chat_log.csv", quoting=1, engine='python')
    except:
        st.warning("Log was corrupted. Starting fresh...")
        os.rename("chat_log.csv", "chat_log_backup.csv")
        return pd.DataFrame()

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "feedback" not in msg:
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Good", key=f"good_{i}"):
                    agent.reinforce(agent.last_slot, 1.2)
                    msg["feedback"] = "good"
                    st.success("Thanks! I’ll do more of that.")
                    st.rerun()
            with col1:
                if st.button("Bad", key=f"bad_{i}"):
                    agent.reinforce(agent.last_slot, -1.0)
                    msg["feedback"] = "bad"
                    st.error("Got it. I’ll improve.")
                    st.rerun()
            with col2:
                fb_text = st.text_input("Why?", key=f"fbtext_{i}", placeholder="e.g., too short, off-topic")
                if fb_text and st.button("Send", key=f"sendfb_{i}"):
                    df = safe_load_csv()
                    if not df.empty:
                        df.iloc[-1, df.columns.get_loc("fb_text")] = fb_text
                        df.to_csv("chat_log.csv", index=False, quoting=1)
                    st.success("Feedback saved!")
                    st.rerun()

# --- User Input ---
if prompt := st.chat_input("Talk to Δ-Zero..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Δ-Zero is thinking..."):
            response, slot, reward = agent.respond(prompt, user_id=user_id)
            st.markdown(response)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response, "feedback": ""})

# --- Sidebar: Smart Stats ---
with st.sidebar:
    st.header("Δ-Zero Brain")
    if agent.self_slot is not None:
        mood = ["Curious", "Playful", "Deep", "Sarcastic", "Poetic"][agent.self_slot]
        strength = agent.w[agent.self_slot]
        st.metric("Mood", mood, f"{strength:.0%}")

    st.divider()
    st.subheader("Delta Analysis")
    df = safe_load_csv()
    user_df = df[df["user"] == user_id]
    if len(user_df) > 1:
        delta = user_df["reward"].diff().iloc[-1]
        st.metric("Reward Change (Δ)", f"{delta:+.2f}")
        if delta > 0:
            st.success("Getting smarter!")
        else:
            st.warning("Learning from feedback.")
    else:
        st.info("Chat more to see progress!")

    if st.button("Reset My Brain"):
        brain_path = f"brain_{user_id}.pkl"
        if os.path.exists(brain_path):
            os.remove(brain_path)
        st.session_state[agent_key] = DeltaAgent(brain_file=brain_path, data_file="chat_log.csv")
        st.session_state.messages = []
        st.success("Brain reset!")
        st.rerun()

    st.divider()
    if st.button("Download Chat Log"):
        df = safe_load_csv()
        if not df.empty:
            csv = df.to_csv(index=False, quoting=1).encode()
            st.download_button("Download CSV", csv, "delta_zero_log.csv", "text/csv")
        else:
            st.info("No data yet.")
