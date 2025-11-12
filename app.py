# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Adaptive AI with Feedback, Mood Chart & Knowledge
# by JCB – your personalized AI companion by JCB
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
from sentence_transformers import SentenceTransformer, util  # Semantic search

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
# 1b. Load conversations from /conversations/*.txt
# ==============================================================
def load_conversations():
    convs = []
    conv_dir = os.path.join(os.path.dirname(__file__), "conversations")
    if not os.path.exists(conv_dir):
        return convs
    for filename in os.listdir(conv_dir):
        if filename.endswith(".txt"):
            path = os.path.join(conv_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    for i in range(0, len(lines) - 1, 2):
                        user_msg = lines[i]
                        bot_msg = lines[i + 1]
                        if user_msg and bot_msg:
                            convs.append({"user": user_msg, "bot": bot_msg})
            except Exception as e:
                st.error(f"Could not read {filename}: {e}")
    if convs:
        print(f"[DEBUG] Loaded {len(convs)} movie conversation pairs")
    return convs

# ==============================================================
# 2. DeltaAgent – Semantic + Context + Mood-Tuned
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
        context_size=5,  # Remember last 5 turns
    ):
        self.n_slots = n_slots
        self.lr = lr
        self.brain_file = brain_file
        self.data_file = data_file
        self.mood_file = mood_file
        self.key_file = key_file
        self.context_size = context_size
        self.knowledge = load_knowledge()
        self.conversations = load_conversations()
        self.memory = []
        self.mood_history = []
        self.last_slot = None
        self.dynamic_convos = []
        self.context = []  # Short-term: last N turns
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Semantic embeddings

        # Encryption
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

        # Load chat log
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

        # Build dynamic convos
        self._update_dynamic_convos()

    def _update_dynamic_convos(self):
        self.dynamic_convos = []
        for i in range(0, len(self.memory) - 1, 2):
            user_msg = self.memory[i].get("input", "")
            bot_msg = self.memory[i + 1].get("response", "")
            if user_msg and bot_msg:
                self.dynamic_convos.append({"user": user_msg, "bot": bot_msg})

    def _apply_mood_boost(self, mood):
        """Boost slots based on mood"""
        boosted_w = self.w.copy()
        if mood <= 3:  # Low: Empathetic (slot 3)
            boosted_w[3] *= 1.4
        elif mood >= 7:  # High: Engaging (2) + Curious (0)
            boosted_w[2] *= 1.4
            boosted_w[0] *= 1.4
        boosted_w /= boosted_w.sum()
        return boosted_w

    def choose_slot(self, mood=None):
        probs = self.w
        if mood is not None:
            probs = self._apply_mood_boost(mood)
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

    def generate_response(self, user_input, slot, current_mood=None):
        # Build full context (last 5 turns + all convos)
        context_msgs = [turn["input"] + " " + turn["response"] for turn in self.context[-self.context_size*2:]]  # Last 5 user+bot
        all_convos = self.conversations + self.dynamic_convos
        all_texts = [c["user"] for c in all_convos] + context_msgs

        # Semantic search
        if all_texts:
            query_emb = self.model.encode(user_input)
            text_embs = self.model.encode(all_texts)
            scores = util.cos_sim(query_emb, text_embs)[0]
            best_idx = scores.argmax().item()
            if scores[best_idx] > 0.5:  # Semantic threshold
                if best_idx < len(all_convos):
                    reply = all_convos[best_idx]["bot"]
                else:
                    # From context (reuse recent)
                    reply = all_convos[0]["bot"]  # Fallback to first
                return reply + f" [slot {slot}]"

        # Fallback: Exact 3+ word match (backup)
        all_convos = self.conversations + self.dynamic_convos
        user_words = set(user_input.lower().split())
        if len(user_words) >= 3:
            candidates = []
            for conv in all_convos:
                conv_words = set(conv["user"].lower().split())
                overlap = len(user_words.intersection(conv_words))
                if overlap >= 3:
                    candidates.append((conv["bot"], overlap))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0] + f" [slot {slot}]"

        # Final fallback: Personality
        base = random.choice(self.REPLIES[slot])
        if self.knowledge and random.random() < 0.2:
            fact = random.choice(self.knowledge)
            base += f" Fun fact: {fact}"
        return base + f" [slot {slot}]"

    def respond(self, user_input, mood=None):
        slot = self.choose_slot(mood)
        response = self.generate_response(user_input, slot, mood)
        # Update context
        self.context.append({"input": user_input, "response": response})
        if len(self.context) > self.context_size * 2:
            self.context = self.context[-self.context_size * 2:]
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
        self._update_dynamic_convos()

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
# 3. Streamlit UI – Upgraded
# ==============================================================
st.set_page_config(page_title="Δ-Zero Chat", layout="wide")
st.title("Δ-Zero Chat – Adaptive AI")
st.markdown(
    "<sub>by JCB – your personalized AI companion by JCB</sub>",
    unsafe_allow_html=True
)

agent = DeltaAgent()

# ---------- Sidebar ----------
st.sidebar.header("Mood Tracker")
mood = st.sidebar.slider("Your mood (optional)", 0.0, 10.0, 5.0, 0.5)
if st.sidebar.button("Record Mood"):
    agent.update_mood(mood)
    st.sidebar.success("Saved!")

if agent.mood_history:
    df = pd.DataFrame(agent.mood_history)
    if not df.empty and "timestamp" in df.columns and "mood" in df.columns:
        fig = px.line(df, x="timestamp", y="mood", title="Mood Over Time", markers=True)
        st.sidebar.plotly_chart(fig, use_container_width=True)
    else:
        st.sidebar.info("No mood data yet.")
else:
    st.sidebar.info("No mood data yet.")

st.sidebar.info(f"Chats stored: {len(agent.memory)}")
if agent.knowledge:
    st.sidebar.success(f"Loaded {len(agent.knowledge)} facts")

# ---------- Slot Confidence ----------
weights = agent.w / agent.w.sum()
slot_labels = ["Curious", "Calm", "Engaging", "Empathetic", "Analytical"]
conf_df = pd.DataFrame({"Style": slot_labels, "Confidence": weights})
conf_fig = px.bar(
    conf_df, x="Style", y="Confidence", title="AI Personality Confidence",
    color="Confidence", color_continuous_scale="Blues", height=250
)
st.plotly_chart(conf_fig, use_container_width=True)

# ---------- Chat History ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_bot_idx" not in st.session_state:
    st.session_state.last_bot_idx = -1

chat_placeholder = st.empty()

def render_chat():
    with chat_placeholder.container():
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
                    with col2:
                        if st.button("Bad", key=f"bad_{i}"):
                            agent.update(0.0)
                            agent.log_interaction("user", "", "", agent.last_slot,
                                                  reward=0.0, feedback="bad")
                            st.error("Learning: avoiding this style")

# ---------- INPUT + SEND (Mood-Aware) ----------
input_col = st.container()

with input_col:
    if "msg_to_send" not in st.session_state:
        st.session_state.msg_to_send = ""

    def submit_on_enter():
        if st.session_state.msg_to_send.strip():
            st.session_state.pending_message = st.session_state.msg_to_send
            st.session_state.msg_to_send = ""

    user_input = st.text_input(
        "Type your message…",
        placeholder="Ask Δ-Zero anything…",
        key="msg_to_send",
        on_change=submit_on_enter
    )

    send_clicked = st.button("Send", type="primary")

    if send_clicked or getattr(st.session_state, "pending_message", None):
        msg = (st.session_state.pending_message
               if "pending_message" in st.session_state else user_input)

        if msg.strip():
            response, slot = agent.respond(msg, mood)  # Mood tunes slot
            agent.log_interaction("user", msg, response, slot)
            agent.save_state()

            st.session_state.chat_history.append({"sender": "user", "message": msg})
            st.session_state.chat_history.append({"sender": "bot", "message": response})
            st.session_state.last_bot_idx = len(st.session_state.chat_history) - 1

            if "pending_message" in st.session_state:
                del st.session_state.pending_message
            st.rerun()

# Render chat
render_chat()

# --------------------------------------------------------------
# Reuse past messages
# --------------------------------------------------------------
if st.checkbox("Reuse past messages"):
    past = [e["input"] for e in agent.memory[-20:] if e["input"]]
    sel = st.selectbox("Pick one", [""] + past)
    if sel:
        st.session_state.msg_to_send = sel
        st.rerun()

# --------------------------------------------------------------
# Learning summary
# --------------------------------------------------------------
if st.button("Show Feedback Summary"):
    fb = [e for e in agent.memory if e["feedback"]]
    if fb:
        df = pd.DataFrame(fb)["feedback"].value_counts().reset_index()
        df.columns = ["Feedback", "Count"]
        fig = px.pie(df, names="Feedback", values="Count", title="User Feedback")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feedback yet.")
