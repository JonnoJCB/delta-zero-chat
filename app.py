# app.py
# Δ-Zero Chat v3 — LLM-Enhanced, Polya-Urn Personality Reinforced
# by JCB + GPT-5 mini upgrade

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
import openai

# ============================================================== #
# CONFIG
# ============================================================== #
BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
BRAIN_FILE = os.path.join(BASE_DIR, "global_brain.pkl")
DATA_FILE = os.path.join(BASE_DIR, "chat_log.enc")
KEY_FILE = os.path.join(BASE_DIR, "secret.key")
MAX_MEMORY = 300

st.set_page_config(page_title="Delta-Zero Chat v3", layout="centered")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Dark glass theme
st.markdown("""
<style>
    .main {background:#0a0a1a; color:#e6e6ff;}
    .stChatMessage {margin:10px 0; padding:14px; border-radius:16px;}
    .user {background:linear-gradient(135deg,#1d4ed8,#3b82f6); color:white; text-align:right; margin-left:25%;}
    .bot {background:linear-gradient(135deg,#4c1d95,#7c3aed); color:#e0d8ff; margin-right:25%;}
    .title {font-size:4rem; text-align:center; margin:20px 0 0 0; letter-spacing:3px;}
    .subtitle {text-align:center; color:#a78bfa; font-size:1.2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Δ-Zero Chat</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Adaptive • Contextual • Self-Learning • LLM-Enhanced</p>", unsafe_allow_html=True)
st.caption("<p style='text-align:center; color:#888;'>by JCB — now thinking with GPT power</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================== #
# KNOWLEDGE
# ============================================================== #
def load_knowledge():
    knowledge = []
    if os.path.exists(KNOWLEDGE_DIR):
        for f in os.listdir(KNOWLEDGE_DIR):
            if f.endswith(".txt"):
                try:
                    with open(os.path.join(KNOWLEDGE_DIR, f), "r", encoding="utf-8") as file:
                        for line in file:
                            t = line.strip()
                            if t and len(t) > 8:
                                knowledge.append(t)
                except: pass
    return knowledge

# ============================================================== #
# Δ-ZERO AGENT v3
# ============================================================== #
class DeltaAgent:
    REPLIES = [
        ["Tell me more!", "I'm curious — keep going!", "That's interesting!"],
        ["Got it.", "Okay, still learning.", "I see what you mean."],
        ["Yes! More!", "This is fun — continue!", "I'm loving this!"],
        ["That sounds meaningful.", "How does that make you feel?", "I want to understand."],
        ["Let's break it down.", "I see a pattern.", "Explain the logic?"]
    ]
    LABELS = ["Curious", "Calm", "Engaging", "Empathetic", "Analytical"]

    def __init__(self):
        self.knowledge = load_knowledge()
        self.memory = []
        self.last_slot = None
        self.cipher = self._load_or_create_key()
        self.w = self._load_brain()
        self.memory = self._load_log()
        self._refresh_vectorizer()

    # Persistence
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
                df = pd.read_csv(pd.io.common.StringIO(data))
                return df.to_dict("records")
        except: return []

    def _save_log(self):
        df = pd.DataFrame(self.memory)
        enc = self.cipher.encrypt(df.to_csv(index=False).encode())
        with open(DATA_FILE, "wb") as f: f.write(enc)

    def _save_brain(self):
        with open(BRAIN_FILE, "wb") as f:
            pickle.dump({"w": self.w}, f)

    # Vectorizer
    def _refresh_vectorizer(self):
        recent = [f"{m.get('input','')} {m.get('response','')}" for m in self.memory[-MAX_MEMORY:] if isinstance(m, dict)]
        texts = self.knowledge + recent
        if len(texts) > 5:
            try:
                self.vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
                self.matrix = self.vectorizer.fit_transform(texts)
                self.text_index = texts
            except:
                self.matrix = None
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
            if mood <= 3: w[3] *= 1.6
            if mood >= 7: w[0] *= 1.4; w[2] *= 1.4
        p = w / w.sum()
        slot = np.random.choice(5, p=p)
        self.last_slot = slot
        return slot

    # ======================= LLM INTEGRATION =======================
    def _llm_call(self, prompt):
        try:
            completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "You are Δ-Zero, an adaptive AI agent with memory, "
                        "polya-urn personality weights, encrypted recall, "
                        "and emotional tuning. "
                        "Your style is influenced by your personality weights: "
                        f"{self.w.tolist()}. "
                        "You speak briefly but intelligently."
                    )},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=220,
                temperature=0.84
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print("LLM error:", e)
            return ""

    def generate_response(self, user_input, slot):
        # Contextual recall
        recalled_fact = ""
        if self.matrix is not None and len(user_input) > 4:
            try:
                qv = self.vectorizer.transform([user_input])
                sims = cosine_similarity(qv, self.matrix).flatten()
                if sims.max() > 0.20:
                    recalled_fact = self.text_index[sims.argmax()]
            except:
                pass

        persona_label = self.LABELS[slot]

        system_prompt = f"""
The user said:
{user_input}

Your current personality slot: {persona_label}
Your personality weights: {self.w.tolist()}
Recalled memory (optional): {recalled_fact}

Respond in Δ-Zero's style:
- curious and alive
- short to medium length
- emotionally aware
- optionally reference the recalled fact
"""
        llm_reply = self._llm_call(system_prompt)

        # Fallback
        if not llm_reply or len(llm_reply.strip()) < 2:
            llm_reply = random.choice(self.REPLIES[slot])

        return llm_reply.strip()

    def respond(self, user_input, mood=None):
        slot = self.choose_slot(mood)
        response = self.generate_response(user_input, slot)
        return response, slot

    # ======================= REINFORCEMENT =======================
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
    with st.spinner("Booting Δ-Zero v3..."):
        st.session_state.agent = DeltaAgent()
agent = st.session_state.agent

if "history" not in st.session_state:
    st.session_state.history = []
if "last_idx" not in st.session_state:
    st.session_state.last_idx = -1

# ============================================================== #
# SIDEBAR — Mood + Personality Radar
# ============================================================== #
with st.sidebar:
    st.header("Your Mood")
    mood = st.slider("How are you feeling?", 0.0, 10.0, 5.0, 0.5)
    if st.button("Record"):
        st.success("Mood saved")

    st.divider()
    st.info(f"**Facts:** {len(agent.knowledge)}\n**Chats:** {len(agent.memory)}")

    values = (agent.w / agent.w.sum() * 100).round(1).tolist()
    fig = px.line_polar(r=values, theta=agent.LABELS, line_close=True,
                        title="Δ-Zero Personality", template="plotly_dark")
    fig.update_traces(fill='toself', fillcolor='rgba(139,92,246,0.5)', line_color='#c4b5fd')
    st.plotly_chart(fig, use_container_width=True)

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
                agent.update(1); st.rerun()
            if c2.button("Bad", key=f"b{i}"):
                agent.update(0); st.rerun()

# ============================================================== #
# INPUT
# ============================================================== #
if prompt := st.chat_input("Talk to Δ-Zero • teach: fact to remember forever"):
    user_input = prompt.strip()

    if user_input.lower().startswith(("teach:", "remember:", "fact:", "learn:")):
        fact = user_input.split(":", 1)[1].strip()
        success = agent.add_fact(fact)
        response = "Permanently learned!" if success else "I already knew that."
        slot = 2
    else:
        response, slot = agent.respond(user_input, mood)

    agent.log(user_input, response, slot)
    st.session_state.history.append({"role": "user", "text": user_input})
    st.session_state.history.append({"role": "bot", "text": response})
    st.session_state.last_idx = len(st.session_state.history) - 1
    st.rerun()
