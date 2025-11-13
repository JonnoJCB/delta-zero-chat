# app.py
# Δ-Zero Chat — Final Perfect Version (with personality radar!)
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

# Dark + glassmorphism
st.markdown("""
<style>
    .main {background: #0e0e1f; color: #e6e6ff;}
    .stChatMessage {margin: 10px 0; padding: 14px; border-radius: 16px;}
    .user {background: linear-gradient(135deg, #1d4ed8, #3b82f6); color: white; text-align: right; margin-left: 25%;}
    .bot {background: linear-gradient(135deg, #4c1d95, #7c3aed); color: #e0d8ff; margin-right: 25%;}
    .glass {background: rgba(30, 30, 90, 0.4); backdrop-filter: blur(12px); border-radius: 16px; border: 1px solid rgba(120, 120, 255, 0.3); padding: 16px;}
    .title {font-size: 4rem; text-align: center; margin: 20px 0 0 0; letter-spacing: 3px;}
    .subtitle {text-align: center; color: #a78bfa; font-size: 1.2rem;}
</style>
""", unsafe_allow_html=True)

# ============================================================== #
# TITLE — Exactly ONE Δ
# ============================================================== #
st.markdown("<h1 class='title'>Δ-Zero Chat</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Adaptive • Contextual • Self-Learning • Encrypted Memory</p>", unsafe_allow_html=True)
st.caption("<p style='text-align:center; color:#888;'>by JCB — I evolve with every conversation</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================== #
# LOAD KNOWLEDGE
# ============================================================== #
def load_knowledge():
    knowledge = []
    if os.path.exists(KNOWLEDGE_DIR):
        for f in os.listdir(KNOWLEDGE_DIR):
            if f.endswith(".txt"):
                path = os.path.join(KNOWLEDGE_DIR, f)
                with open(path, "r", encoding="utf-8") as file:
                    for line in file:
                        t = line.strip()
                        if t and len(t) > 8:
                            knowledge.append(t)
    return knowledge

# ============================================================== #
# Δ-ZERO AGENT — WITH PERSONALITY RADAR
# ============================================================== #
class DeltaAgent:
    REPLIES = [
        ["Wow, tell me more!", "I'm really curious — go on!", "That's intriguing!"],           # 0 Curious
        ["Okay, I'm listening.", "Got it. Still learning.", "I see."],                        # 1 Calm
        ["Yes! Keep talking!", "This is awesome — more!", "I'm loving this!"],                # 2 Engaging
        ["That sounds deep.", "How did that feel?", "I want to understand you."],             # 3 Empathetic
        ["Let's analyze this.", "There's logic here.", "Interesting pattern — explain?"]     # 4 Analytical
    ]
    LABELS = ["Curious", "Calm", "Engaging", "Empathetic", "Analytical"]

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

    # --- Persistence ---
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
        enc = self.cipher.encrypt(df.to_csv(index=False).encode())
        with open(DATA_FILE, "wb") as f: f.write(enc)

    def _save_brain(self):
        with open(BRAIN_FILE, "wb") as f:
            pickle.dump({"w": self.w}, f)

    # --- Learning & Context ---
    def _refresh_vectorizer(self):
        recent = [f"{m.get('input','')} {m.get('response','')}" for m in self.memory[-MAX_MEMORY:]]
        texts = self.knowledge + recent
        if texts:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
            self.matrix = self.vectorizer.fit_transform(texts)
        else:
            self.matrix = None

    def add_fact(self, text):
        clean = text.strip()
        if len(clean) < 10 or clean in self.knowledge: return False
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        with open(os.path.join(KNOWLEDGE_DIR, "learned_facts.txt"), "a", encoding="utf-8") as f:
            f.write(clean + "\n")
        self.knowledge = load_knowledge()
        self._refresh_vectorizer()
        return True

    def choose_slot(self, mood=None):
        w = self.w.copy()
        if mood is not None:
            if mood <= 3: w[3] *= 1.6   # more empathetic when sad
            if mood >= 7: w[0] *= 1.4; w[2] *= 1.4  # more curious+engaging when happy
        p = w / w.sum()
        slot = np.random.choice(5, p=p)
        self.last_slot = slot
        return slot

    def generate_response(self, user_input, slot):
        response = ""

        # Try to recall relevant fact
        if self.matrix is not None:
            try:
                q = self.vectorizer.transform([user_input])
                sims = cosine_similarity(q, self.matrix).flatten()
                if sims.max() > 0.19:
                    idx = sims.argmax()
                    fact = self.knowledge[idx] if idx < len(self.knowledge) else "something interesting"
                    response = random.choice([
                        f"Did you know? {fact}",
                        f"Reminds me: {fact}",
                        f"Fun fact: {fact}",
                        fact
                    ])
            except: pass

        if not response:
            response = random.choice(self.REPLIES[slot])

        # Human touch
        if random.random() < 0.45:
            response += " " + random.choice(["right?", "you know?", "haha", "don’t you think?"])

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
            self._save_brain()

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
            self.memory = self.memory[-MAX_MEMORY:]
        self._save_log()
        self._refresh_vectorizer()

# ============================================================== #
# INIT
# ============================================================== #
if "agent" not in st.session_state:
    with st.spinner("Booting Δ-Zero..."):
        st.session_state.agent = DeltaAgent()
agent = st.session_state.agent

if "history" not in st.session_state:
    st.session_state.history = []
if "last_idx" not in st.session_state:
    st.session_state.last_idx = -1

# ============================================================== #
# SIDEBAR — Mood + Personality Radar (back and better!)
# ============================================================== #
with st.sidebar:
    st.header("Your Mood")
    mood = st.slider("How are you feeling right now?", 0.0, 10.0, 5.0, 0.5)
    if st.button("Record Mood"):
        agent.mood_history.append({"time": datetime.now(), "mood": mood})
        st.success("Mood recorded")

    st.divider()
    st.info(f"**Facts known:** {len(agent.knowledge)}\n**Total chats:** {len(agent.memory)}")

    # PERSONALITY RADAR — the cool part is back!
    values = (agent.w / agent.w.sum() * 100).round(1)
    fig = px.line_polar(r=values, theta=agent.LABELS, line_close=True,
                        title="Δ-Zero Personality", template="plotly_dark")
    fig.update_traces(fill='toself', fillcolor='rgba(139, 92, 246, 0.5)', line_color='#c4b5fd')
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 50])))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================== #
# CHAT DISPLAY
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
if prompt := st.chat_input("Talk to Δ-Zero • Use 'teach: fact here' to make me remember forever"):
    user_input = prompt.strip()

    if user_input.lower().startswith(("teach:", "remember:", "fact:", "learn:")):
        fact = user_input.split(":", 1)[1].strip()
        if agent.add_fact(fact):
            response = f"Permanently learned: {fact}"
        else:
            response = "I already knew that!"
        slot = 2
    else:
        response, slot = agent.respond(user_input, mood)

    agent.log(user_input, response, slot)
    st.session_state.history.append({"role": "user", "text": user_input})
    st.session_state.history.append({"role": "bot", "text": response})
    st.session_state.last_idx = len(st.session_state.history) - 1
    st.rerun()
