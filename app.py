# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Upgraded v2 (November 2025)
# by JCB + upgrades by your friend
# --------------------------------------------------------------

import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import random
from datetime import datetime
from cryptography.fernet import Fernet
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time

# ============================================================== #
# CONFIG PATHS
# ============================================================== #
BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
BRAIN_FILE = os.path.join(BASE_DIR, "global_brain.pkl")
DATA_FILE = os.path.join(BASE_DIR, "chat_log.enc")
MOOD_FILE = os.path.join(BASE_DIR, "mood_history.pkl")
KEY_FILE = os.path.join(BASE_DIR, "secret.key")
MAX_MEMORY = 600
SUMMARY_TRIGGER = 550
BOOTSTRAP_FILE = os.path.join(BASE_DIR, "bootstrapped.flag")

# ============================================================== #
# LOAD KNOWLEDGE
# ============================================================== #
def load_knowledge():
    knowledge = []
    if os.path.exists(KNOWLEDGE_DIR):
        for f in os.listdir(KNOWLEDGE_DIR):
            if f.endswith(".txt"):
                try:
                    with open(os.path.join(KNOWLEDGE_DIR, f), "r", encoding="utf-8") as file:
                        for line in file:
                            text = line.strip()
                            if text:
                                knowledge.append(text)
                except Exception as e:
                    print(f"Error loading {f}: {e}")
    return knowledge


# ============================================================== #
# Δ-ZERO AGENT
# ============================================================== #
class DeltaAgent:
    REPLIES = [
        ["Well it's nice to talk to someone - give me information?", "I'm intrigued! Tell me more?", "Really? What makes you say that?", "Ooh keep going!", "That's new to me - elaborate?"],
        ["I get that completely.", "Makes sense. I'm still learning though.", "Okay, noted. Anything else?", "Totally hear you.", "That's fair."],
        ["Tell me more about that! I'm all ears!", "Yes! Keep going!", "This is getting good.", "I love where this is going.", "Don't stop now!"],
        ["That sounds intense… how did it make you feel?", "I can imagine that wasn't easy.", "I'm really listening.", "That's heavy. Want to talk about it?", "Your feelings make sense."],
        ["Let's break this down.", "Interesting pattern here.", "Walk me through the logic?", "What caused that, do you think?", "I'm analyzing this as we speak."],
    ]

    def __init__(self, n_slots=5, lr=0.09):
        self.n_slots = n_slots
        self.lr = lr
        self.knowledge = load_knowledge()
        self.memory = []
        self.mood_history = []
        self.last_slot = None
        self.cipher = self._load_or_create_key()

        st.info("Loading sentence transformer (first start takes ~10 sec)...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_embeddings = None
        self.all_texts = []

        # Load saved state
        self.w = self._load_brain()
        self.memory = self._load_encrypted_log()
        self.mood_history = self._load_mood()

        self._refresh_knowledge_base()

    # ------------------- ENCRYPTION / FILE HANDLING -------------------
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
                enc = f.read()
                if not enc:
                    return []
                dec = self.cipher.decrypt(enc)
                df = pd.read_csv(pd.io.common.StringIO(dec.decode()))
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
        enc = self.cipher.encrypt(csv.encode())
        with open(DATA_FILE, "wb") as f:
            f.write(enc)

    # ------------------- SMART KNOWLEDGE + EMBEDDINGS -------------------
    def _refresh_knowledge_base(self):
        recent = self.memory[-MAX_MEMORY:]
        memory_texts = [f"{m.get('input','')} {m.get('response','')}" for m in recent if isinstance(m, dict)]
        self.all_texts = self.knowledge + memory_texts

        if self.all_texts:
            with st.spinner("Δ-Zero is updating its brain (embeddings)..."):
                self.knowledge_embeddings = self.sentence_model.encode(
                    self.all_texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
        else:
            self.knowledge_embeddings = None

    def refresh_knowledge(self):
        self.knowledge = load_knowledge()
        self._refresh_knowledge_base()

    def add_fact(self, text):
        text = text.strip()
        if not text:
            return False

        if self.knowledge_embeddings is not None and len(self.all_texts) > 0:
            emb = self.sentence_model.encode([text])
            sims = cosine_similarity(emb, self.knowledge_embeddings)[0]
            if sims.max() > 0.92:
                return False

        path = os.path.join(KNOWLEDGE_DIR, "learned_facts.txt")
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        self.knowledge.append(text)
        self._refresh_knowledge_base()
        return True

    def _summarize_old_memory(self):
        if len(self.memory) < SUMMARY_TRIGGER:
            return
        old_chunk = self.memory[:-400]
        summary_facts = [
            "User taught Δ-Zero many facts about movies, science, and history.",
            "We had deep conversations about emotions and personal experiences.",
            "User prefers curious and engaging responses most of the time.",
            "Several long discussions about AI, technology, and the future.",
            "User shared favorite books, games, and life stories."
        ]
        random.shuffle(summary_facts)
        for line in summary_facts[:3]:
            self.add_fact(line)
        self.memory = self.memory[-400:]
        df = pd.DataFrame(self.memory)
        self._save_encrypted_df(df)

    # ------------------- RESPONSE GENERATION -------------------
    def generate_response(self, user_input, slot, mood=None):
        response = ""

        if self.knowledge_embeddings is not None and len(user_input) > 8:
            query_emb = self.sentence_model.encode([user_input])
            sims = cosine_similarity(query_emb, self.knowledge_embeddings)[0]
            if sims.max() > 0.58:
                fact = self.all_texts[sims.argmax()]
                if fact.lower() not in user_input.lower():
                    response = random.choice([
                        f"I remember something related: {fact}",
                        f"That reminds me — {fact}",
                        f"By the way: {fact}",
                        fact
                    ])

        if not response:
            response = random.choice(self.REPLIES[slot])

        softeners = ["you know?", "right?", "haha", "just thinking out loud.", "don’t you think?"]
        if random.random() < 0.5:
            response += " " + random.choice(softeners)

        if self.knowledge and random.random() < 0.3:
            extra = random.choice(self.knowledge)
            if len(response + extra) < 280 and extra not in response:
                response += f" Oh and {extra.lower()}"

        return response + f" [slot {slot}]"

    def respond(self, user_input, mood=None):
        slot = self.choose_slot(mood)
        response = self.generate_response(user_input, slot, mood)
        return response, slot

    def update(self, reward):
        if self.last_slot is None:
            return
        self.w[self.last_slot] += self.lr * (reward - self.w[self.last_slot])
        for i in range(self.n_slots):
            if i != self.last_slot:
                self.w[i] += self.lr * 0.15 * (reward - self.w[i])
        self.w = np.clip(self.w, 0.01, None)
        self.w /= self.w.sum()

    def _apply_mood_boost(self, mood):
        w = self.w.copy()
        if mood <= 3:
            w[3] *= 1.5
        elif mood >= 7:
            w[0] *= 1.35
            w[2] *= 1.35
        return w / w.sum()

    def choose_slot(self, mood=None):
        probs = self._apply_mood_boost(mood) if mood is not None else self.w
        slot = np.random.choice(self.n_slots, p=probs)
        self.last_slot = slot
        return slot

    def log_interaction(self, user_input, response, slot, reward=None, feedback=None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {"timestamp": ts, "input": user_input, "response": response, "slot": slot,
                 "reward": reward, "feedback": feedback}
        self.memory.append(entry)
        df = pd.DataFrame(self.memory)
        self._save_encrypted_df(df)

        lower = user_input.lower()
        if any(phrase in lower for phrase in [" is ", " was ", " are ", " were ", " released ", " directed ", " stars ", " invented ", " created ", " discovered "]):
            self.add_fact(user_input)

        if len(self.memory) > SUMMARY_TRIGGER:
            self._summarize_old_memory()

        self._refresh_knowledge_base()

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
# STREAMLIT UI
# ============================================================== #
st.set_page_config(page_title="Δ-Zero Chat", layout="wide")
st.title("Δ-Zero Chat – Adaptive AI (v2)")
st.markdown("<sub>Now with real understanding • smarter • never forgets</sub>", unsafe_allow_html=True)

# Initialize agent
if "agent" not in st.session_state:
    with st.spinner("Waking up Δ-Zero (loading brain + embeddings)..."):
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
    fig = px.line(df_mood, x="timestamp", y="mood", title="Mood Over Time", markers=True)
    st.sidebar.plotly_chart(fig, use_container_width=True)

st.sidebar.info(f"Total chats: {len(agent.memory)}")
if agent.knowledge:
    st.sidebar.success(f"Knowledge base: {len(agent.knowledge)} facts")

# Personality bar
slot_labels = ["Curious", "Calm", "Engaging", "Empathetic", "Analytical"]
weights = (agent.w / agent.w.sum()).round(3)
conf_df = pd.DataFrame({"Style": slot_labels, "Confidence": weights})
conf_fig = px.bar(conf_df, x="Style", y="Confidence", color="Confidence",
                  title="Δ-Zero Personality", color_continuous_scale="Blues", height=250)
st.plotly_chart(conf_fig, use_container_width=True)

# Chat rendering
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_bot_idx" not in st.session_state:
    st.session_state.last_bot_idx = -1

def render_chat():
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["sender"] == "user":
            st.markdown(f"<div style='background:#D1E7DD;padding:12px;border-radius:10px;text-align:right;margin:5px 0'><b>You:</b> {msg['message']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#F8D7DA;padding:12px;border-radius:10px;margin:5px 0'><b>Δ-Zero:</b> {msg['message']}</div>", unsafe_allow_html=True)
            if i == st.session_state.last_bot_idx:
                c1, c2 = st.columns(2)
                if c1.button("Good", key=f"good_{i}"):
                    agent.update(1.0)
                    agent.log_interaction("", "", agent.last_slot, reward=1.0, feedback="good")
                    st.rerun()
                if c2.button("Bad", key=f"bad_{i}"):
                    agent.update(0.0)
                    agent.log_interaction("", "", agent.last_slot, reward=0.0, feedback="bad")
                    st.rerun()

render_chat()

# Regenerate button
if st.button("Regenerate last response"):
    if st.session_state.chat_history and len(st.session_state.chat_history) >= 2:
        last_user_msg = None
        for msg in reversed(st.session_state.chat_history):
            if msg["sender"] == "user":
                last_user_msg = msg["message"]
                break
        if last_user_msg:
            with st.spinner("Thinking again..."):
                response, slot = agent.respond(last_user_msg, mood)
            agent.log_interaction(last_user_msg, response, slot)
            agent.save_state()
            st.session_state.chat_history[-1]["message"] = response
            st.rerun()

# Main chat input
if user_input := st.chat_input("Talk to Δ-Zero..."):
    lines = [line.strip() for line in user_input.split("\n") if line.strip()]
    for i in range(0, len(lines), random.randint(1, 2)):
        chunk = "\n".join(lines[i:i+random.randint(1, 2)])
        agent.add_fact(chunk)

    response, slot = agent.respond(user_input, mood)
    agent.log_interaction(user_input, response, slot)
    agent.save_state()

    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    st.session_state.chat_history.append({"sender": "bot", "message": response})
    st.session_state.last_bot_idx = len(st.session_state.chat_history) - 1
    st.rerun()

# ============================================================== #
# ONE-TIME BOOTSTRAPPING
# ============================================================== #
def bootstrap_ai(agent, n_rounds=6):
    if os.path.exists(BOOTSTRAP_FILE):
        return
    st.info("Δ-Zero is warming up (self-talk bootstrapping)...")
    starter_facts = [
        "The Matrix was released in 1999.",
        "Inception is directed by Christopher Nolan.",
        "Star Wars Episode IV came out in 1977.",
        "Blade Runner is a cyberpunk classic.",
        "Interstellar explores relativity and love.",
        "The Godfather is considered one of the greatest films ever made."
    ]
    for _ in range(n_rounds):
        line = random.choice(starter_facts + ["tell me more", "what do you think?", "isn’t that cool?"])
        resp, slot = agent.respond(line)
        agent.log_interaction(line, resp, slot)
        agent.add_fact(line)
    agent.save_state()
    with open(BOOTSTRAP_FILE, "w") as f:
        f.write("done")
    st.success("Δ-Zero is now alive and ready!")

bootstrap_ai(agent)
