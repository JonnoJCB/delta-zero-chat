# app.py
# Δ-Zero Chat — Clean Final Version
# by JCB

import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import random
from datetime import datetime
from cryptography.fernet import Fernet
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================== #
# CONFIG
# ============================================================== #
BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
BRAIN_FILE = os.path.join(BASE_DIR, "global_brain.pkl")
DATA_FILE = os.path.join(BASE_DIR, "chat_log.enc")
MOOD_FILE = os.path.join(BASE_DIR, "mood_history.pkl")
KEY_FILE = os.path.join(BASE_DIR, "secret.key")
MAX_MEMORY = 300

st.set_page_config(page_title="Delta-Zero Chat", layout="centered")

# Dark mode + clean glassmorphism
st.markdown("""
<style>
    .main { background: #0a0a1a; color: #e0e0ff; }
    .stChatMessage { margin: 8px 0; padding: 12px; border-radius: 12px; }
    .user { background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; text-align: right; margin-left: 20%; }
    .bot { background: linear-gradient(135deg, #1e1b4b, #4c1d95); color: #e0e7ff; margin-right: 20%; }
    .glass { background: rgba(30, 30, 80, 0.35); backdrop-filter: blur(12px); border-radius: 16px; border: 1px solid rgba(100, 100, 255, 0.2); padding: 16px; }
    h1 { font-size: 3.5rem; margin: 0; letter-spacing: 2px; }
    .subtitle { color: #a0a0ff; font-size: 1.1em; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

# ============================================================== #
# TITLE — Exactly one Δ, as requested
# ============================================================== #
st.markdown("<h1 style='text-align:center;'>Δ-Zero Chat</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle' style='text-align:center;'>Adaptive • Contextual • Self-Learning AI</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#777; font-size:0.9em;'>by JCB — remembers everything • evolves with you</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================== #
# LOAD KNOWLEDGE
# ============================================================== #
def load_knowledge():
    knowledge = []
    if os.path.exists(KNOWLEDGE_DIR):
        for f in os.listdir(KNOWLEDGE_DIR):
            if f.endswith(".txt"):
                with open(os.path.join(KNOWLEDGE_DIR, f), "r", encoding="utf-8") as file:
                    for line in file:
                        text = line.strip()
                        if text and len(text) > 8:
                            knowledge.append(text)
    return knowledge

# ============================================================== #
# Δ-Zero AGENT
# ============================================================== #
class DeltaAgent:
    REPLIES = [
        ["Interesting! Tell me more?", "I'm curious — go on!", "What makes you say that?"],
        ["Got it. I'm still learning though.", "Okay, help me understand better.", "I hear you."],
        ["Yes! Keep going!", "This is great — tell me more!", "I love learning from you."],
        ["That sounds meaningful.", "How does that make you feel?", "I want to understand."],
        ["Let's break this down.", "There's a pattern here.", "I see the logic — explain more?"]
    ]

    def __init__(self):
        self.knowledge = load_knowledge()
        self.memory = []
        self.mood_history = []
        self.last_slot = None
        self.cipher = self._load_or_create_key()
        self.w = self._load_brain()
        self.memory = self._load_log()
        self.mood_history = self._load_mood()
        self._refresh_vectorizer()

    def _load_or_create_key(self):
        if not os.path.exists(KEY_FILE):
            key = Fernet.generate_key()
            with open(KEY_FILE, "wb") as f: f.write(key)
        with open(KEY_FILE, "rb") as f: return Fernet(f.read())

    def _load_brain(self):
        if os.path.exists(BRAIN_FILE):
            with open(BRAIN_FILE, "rb") as f:
                return pickle.load(f).get("w", np.ones(5)/5)
        return np.ones(5)/5

    def _load_log(self):
        if not os.path.exists(DATA_FILE): return []
        try:
            with open(DATA_FILE, "rb") as f:
                data = self.cipher.decrypt(f.read()).decode()
                return pd.read_csv(pd.io.common.StringIO(data)).to_dict("records")
        except: return []

    def _load_mood(self):
        if os.path.exists(MOOD_FILE):
            with open(MOOD_FILE, "rb") as f: return pickle.load(f)
        return []

    def _save_log(self):
        df = pd.DataFrame(self.memory)
        csv = df.to_csv(index=False)
        enc = self.cipher.encrypt(csv.encode())
        with open(DATA_FILE, "wb") as f: f.write(enc)

    def _refresh_vectorizer(self):
        recent = [f"{m['input']} {m['response']}" for m in self.memory[-MAX_MEMORY:] if 'input' in m]
        texts = self.knowledge + recent
        if texts:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            self.matrix = self.vectorizer.fit_transform(texts)
        else:
            self.matrix = None

    def add_fact(self, text):
        clean = text.strip()
        if len(clean) < 10 or clean in self.knowledge: return
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        with open(os.path.join(KNOWLEDGE_DIR, "learned_facts.txt"), "a", encoding="utf-8") as f:
            f.write(clean + "\n")
        self.knowledge = load_knowledge()
        self._refresh_vectorizer()

    def choose_slot(self, mood=None):
        w = self.w.copy()
        if mood is not None:
            if mood <= 3: w[3] *= 1.5
            if mood >= 7: w[0] *= 1.3; w[2] *= 1.3
        p = w / w.sum()
        slot = np.random.choice(5, p=p)
        self.last_slot = slot
        return slot

    def generate_response(self, user_input, slot):
        response = ""

        # Contextual recall
        if self.matrix is not None and len(user_input) > 5:
            try:
                q = self.vectorizer.transform([user_input])
                sims = cosine_similarity(q, self.matrix).flatten()
                if sims.max() > 0.18:
                    idx = sims.argmax()
                    fact = (self.knowledge + [m.get("input","") for m in self.memory[-100:]])[idx]
                    response = random.choice([f"Did you know? {fact}", f"Fun fact: {fact}", fact])
            except: pass

        if not response:
            response = random.choice(self.REPLIES[slot])

        # Soften
        if random.random() < 0.5:
            response += " " + random.choice(["right?", "you know?", "haha.", "don't you think?"])

        return response.strip()

    def respond(self, user_input, mood=None):
        slot = self.choose_slot(mood)
        response = self.generate_response(user_input, slot)
        return response, slot

    def update(self, reward):
        if self.last_slot is not None:
            self.w[self.last_slot] += 0.07 * (reward - self.w[self.last_slot])
            self.w = np.clip(self.w, 0.01, None)
            self.w /= self.w.sum()
            with open(BRAIN_FILE, "wb") as f:
                pickle.dump({"w": self.w}, f)

    def log(self, user_input, response, slot, reward=None):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": user_input,
            "response": response,
            "slot": slot,
            "reward": reward
        }
        self.memory.append(entry)
        if len(self.memory) > MAX_MEMORY * 2:
            self.memory = self.memory[-MAX_chat:]
        self._save_log()
        self._refresh_vectorizer()

# ============================================================== #
# INIT
# ============================================================== #
if "agent" not in st.session_state:
    with st.spinner("Starting Δ-Zero..."):
        st.session_state.agent = DeltaAgent()
agent = st.session_state.agent

if "history" not in st.session_state:
    st.session_state.history = []
if "last_idx" not in st.session_state:
    st.session_state.last_idx = -1

# ============================================================== #
# SIDEBAR
# ============================================================== #
with st.sidebar:
    st.header("Mood")
    mood = st.slider("How do you feel?", 0.0, 10.0, 5.0, 0.5)
    if st.button("Save Mood"):
        agent.mood_history.append({"timestamp": datetime.now(), "mood": mood})
        st.success("Mood recorded")

    st.divider()
    st.info(f"Known facts: {len(agent.knowledge)}\nTotal chats: {len(agent.memory)}")

# ============================================================== #
# CHAT
# ============================================================== #
for i, msg in enumerate(st.session_state.history):
    if msg["role"] == "user":
        st.markdown(f"<div class='stChatMessage user'><b>You:</b> {msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='stChatMessage bot'><b>Δ-Zero:</b> {msg['text']}</div>", unsafe_allow_html=True)
        if i == st.session_state.last_idx:
            c1, c2 = st.columns(2)
            if c1.button("Good", key=f"g{i}"):
                agent.update(1)
                st.rerun()
            if c2.button("Bad", key=f"b{i}"):
                agent.update(0)
                st.rerun()

# ============================================================== #
# INPUT
# ============================================================== #
if prompt := st.chat_input("Talk to Δ-Zero • Type 'teach: something' to make me remember permanently"):
    user_input = prompt.strip()

    if user_input.lower().startswith(("teach:", "remember:", "fact:")):
        fact = user_input.split(":", 1)[1].strip()
        agent.add_fact(fact)
        response = f"Learned: {fact}"
        slot = 0
    else:
        response, slot = agent.respond(user_input, mood)

    agent.log(user_input, response, slot)
    st.session_state.history.append({"role": "user", "text": user_input})
    st.session_state.history.append({"role": "bot", "text": response})
    st.session_state.last_idx = len(st.session_state.history) - 1
    st.rerun()
