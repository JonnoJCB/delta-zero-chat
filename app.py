# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Adaptive • Contextual • Self-Learning AI
# by JCB | Clean, Dark, Professional Edition
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================== #
# CONFIG & PATHS
# ============================================================== #
BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
BRAIN_FILE = os.path.join(BASE_DIR, "global_brain.pkl")
DATA_FILE = os.path.join(BASE_DIR, "chat_log.enc")
MOOD_FILE = os.path.join(BASE_DIR, "mood_history.pkl")
KEY_FILE = os.path.join(BASE_DIR, "secret.key")
MAX_MEMORY = 300

# ============================================================== #
# PAGE CONFIG & DARK MODE
# ============================================================== #
st.set_page_config(
    page_title="Δ-Zero Chat",
    page_icon="Δ",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Force dark mode + custom CSS
st.markdown("""
<style>
    .main { background-color: #0e0e1a; color: #e0e0ff; }
    .stChatMessage { border-radius: 12px; padding: 10px; margin: 8px 0; }
    .user-bubble { background: linear-gradient(135deg, #1e3a8a, #3b82f6); color: white; align-self: flex-end; }
    .bot-bubble { background: linear-gradient(135deg, #1e1b4b, #4c1d95); color: #e0e7ff; align-self: flex-start; }
    .glass { 
        background: rgba(30, 30, 70, 0.4); 
        backdrop-filter: blur(10px); 
        border-radius: 16px; 
        border: 1px solid rgba(100, 100, 255, 0.2); 
        padding: 16px; 
        margin: 10px 0;
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; letter-spacing: 1px; }
    .subtitle { color: #a0a0ff; font-size: 1.1em; }
</style>
""", unsafe_allow_html=True)

# ============================================================== #
# TITLE (Beautiful with real Δ symbol)
# ============================================================== #
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("<h1 style='font-size: 4rem; margin: 0;'>Δ</h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='margin-top: 20px;'>Δ-Zero Chat</h1>", unsafe_allow_html=True)

st.markdown("<p class='subtitle'>Adaptive • Contextual • Self-Learning • Encrypted Memory</p>", unsafe_allow_html=True)
st.markdown("<sub style='color:#777;'>by JCB • evolving with every conversation</sub><br><br>", unsafe_allow_html=True)

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
                            if text and len(text) > 10:
                                knowledge.append(text)
                except Exception as e:
                    st.error(f"Error loading {f}: {e}")
    return knowledge

# ============================================================== #
# Δ-Zero AGENT
# ============================================================== #
class DeltaAgent:
    REPLIES = [
        ["Well it's nice to talk to someone — give me information?", "I'm intrigued! Can you explain and tell me a fact?", "Really? What makes you say that?"],
        ["I get that. But I'm still learning. Give me another fact?", "Okay. Tell me more facts to help me understand.", "I hear you. But I need help understanding."],
        ["Tell me more about that! I'm learning from you!", "Give me more information!", "I love where this is going. Keep talking."],
        ["That sounds emotional.", "How did that make you feel?", "Interesting perspective. Tell me more?"],
        ["Let's analyze this a bit. Tell me five random facts? It'll help.", "Interesting pattern. Explain it?", "I like the logic behind that."]
    ]

    def __init__(self):
        self.knowledge = load_knowledge()
        self.memory = []
        self.mood_history = []
        self.last_slot = None
        self.cipher = self._load_or_create_key()
        self.w = self._load_brain()
        self.memory = self._load_encrypted_log()
        self.mood_history = self._load_mood()
        self._refresh_vectorizer()

    # --- Encryption & Persistence ---
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
                return data.get("w", np.ones(5) / 5)
        return np.ones(5) / 5

    def _load_encrypted_log(self):
        if not os.path.exists(DATA_FILE): return []
        try:
            with open(DATA_FILE, "rb") as f:
                enc = f.read()
                if not enc: return []
                dec = self.cipher.decrypt(enc)
                df = pd.read_csv(pd.io.common.StringIO(dec.decode()))
                return df.to_dict("records")
        except: return []

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

    # --- Context & Learning ---
    def _refresh_vectorizer(self):
        recent = [f"{m.get('input','')} {m.get('response','')}" for m in self.memory[-MAX_MEMORY:] if isinstance(m, dict)]
        texts = self.knowledge + recent
        if texts:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            self.knowledge_matrix = self.vectorizer.fit_transform(texts)
        else:
            self.knowledge_matrix = None

    def refresh_knowledge(self):
        self.knowledge = load_knowledge()
        self._refresh_vectorizer()

    def add_fact(self, text):
        if not text or len(text) < 10: return
        path = os.path.join(KNOWLEDGE_DIR, "learned_facts.txt")
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        known = set(self.knowledge)
        clean = text.strip()
        if clean not in known:
            with open(path, "a", encoding="utf-8") as f:
                f.write(clean + "\n")
            self.refresh_knowledge()

    def choose_slot(self, mood=None):
        w = self.w.copy()
        if mood is not None:
            if mood <= 3: w[3] *= 1.5   # Empathetic
            if mood >= 7: w[0] *= 1.3; w[2] *= 1.3  # Curious + Engaging
        probs = w / w.sum()
        slot = np.random.choice(5, p=probs)
        self.last_slot = slot
        return slot

    def generate_response(self, user_input, slot, mood=None):
        response = ""

        # Try contextual recall
        if self.knowledge_matrix is not None:
            try:
                q = self.vectorizer.transform([user_input])
                sims = cosine_similarity(q, self.knowledge_matrix).flatten()
                if sims.max() > 0.18:
                    fact = self.knowledge[sims.argmax()]
                    response = random.choice([
                        f"Did you know? {fact}",
                        f"Here's something I learned: {fact}",
                        f"{fact}",
                        f"Fun fact: {fact}"
                    ])
            except: pass

        if not response:
            response = random.choice(self.REPLIES[slot])

        # Humanize
        if random.random() < 0.5:
            soft = random.choice(["you know?", "right?", "don’t you think?", "haha.", "if that makes sense."])
            response += " " + soft

        if self.knowledge and random.random() < 0.35:
            extra = random.choice(self.knowledge)
            if extra not in response:
                response += f" By the way, {extra.lower()}"

        return response.strip()  # NO [slot X] anymore!

    def respond(self, user_input, mood=None):
        slot = self.choose_slot(mood)
        response = self.generate_response(user_input, slot, mood)
        return response, slot

    def update(self, reward):
        if self.last_slot is not None:
            self.w[self.last_slot] += 0.07 * (reward - self.w[self.last_slot])
            self.w = np.clip(self.w, 0.01, None)
            self.w /= self.w.sum()

    def log_interaction(self, user_input, response, slot, reward=None, feedback=None):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": user_input,
            "response": response,
            "slot": slot,
            "reward": reward,
            "feedback": feedback
        }
        self.memory.append(entry)
        if len(self.memory) > MAX_MEMORY * 2:
            self.memory = self.memory[-MAX_MEMORY:]
        df = pd.DataFrame(self.memory)
        self._save_encrypted_df(df)
        self._refresh_vectorizer()

    def save_state(self):
        with open(BRAIN_FILE, "wb") as f:
            pickle.dump({"w": self.w}, f)
        with open(MOOD_FILE, "wb") as f:
            pickle.dump(self.mood_history, f)

    def update_mood(self, mood_value):
        self.mood_history.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "mood": mood_value})
        self.save_state()

# ============================================================== #
# INITIALIZE AGENT
# ============================================================== #
if "agent" not in st.session_state:
    with st.spinner("Waking up Δ-Zero..."):
        st.session_state.agent = DeltaAgent()
agent = st.session_state.agent

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_bot_idx" not in st.session_state:
    st.session_state.last_bot_idx = -1

# ============================================================== #
# SIDEBAR – Mood + Stats
# ============================================================== #
with st.sidebar:
    st.header("Mood Control")
    mood = st.slider("How are you feeling?", 0.0, 10.0, 5.0, 0.5)
    if st.button("Record Mood"):
        agent.update_mood(mood)
        st.success("Mood saved!")

    if agent.mood_history:
        df = pd.DataFrame(agent.mood_history)
        fig = px.line(df, x="timestamp", y="mood", title="Your Mood Over Time", markers=True,
                      color_discrete_sequence=["#8b5cf6"])
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.info(f"**Chats:** {len(agent.memory)}\n\n**Facts Known:** {len(agent.knowledge)}")

    # Personality Radar
    labels = ["Curious", "Calm", "Engaging", "Empathetic", "Analytical"]
    values = (agent.w / agent.w.sum() * 100).round(1)
    fig2 = px.line_polar(r=values, theta=labels, line_close=True, title="Δ-Zero Personality")
    fig2.update_traces(fill='toself', fillcolor='rgba(139, 92, 246, 0.5)')
    fig2.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)'), template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================== #
# CHAT DISPLAY
# ============================================================== #
for i, msg in enumerate(st.session_state.chat_history):
    if msg["sender"] == "user":
        st.markdown(f"<div class='stChatMessage user-bubble'><b>You:</b> {msg['message']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='stChatMessage bot-bubble'><b>Δ-Zero:</b> {msg['message']}</div>", unsafe_allow_html=True)
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

# ============================================================== #
# CHAT INPUT + TEACH MODE
# ============================================================== #
prompt = st.chat_input("Talk to Δ-Zero • Use 'teach:' to add facts permanently...")

if prompt:
    user_input = prompt.strip()

    # Smart fact teaching
    if user_input.lower().startswith(("teach:", "remember:", "fact:", "learn:")):
        fact = user_input.split(":", 1)[1].strip()
        agent.add_fact(fact)
        response = f"I've learned: {fact}"
        slot = 2
    else:
        with st.spinner("Δ-Zero thinking..."):
            response, slot = agent.respond(user_input, mood)

    agent.log_interaction(user_input, response, slot)
    agent.save_state()

    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    st.session_state.chat_history.append({"sender": "bot", "message": response})
    st.session_state.last_bot_idx = len(st.session_state.chat_history) - 1
    st.rerun()

# ============================================================== #
# ONE-TIME BOOTSTRAP (only once ever)
# ============================================================== #
if len(agent.memory) < 3 and len(agent.knowledge) < 10:
    st.info("Δ-Zero is warming up with initial knowledge...")
    initial_facts = [
        "The Matrix was released in 1999.",
        "Interstellar was directed by Christopher Nolan.",
        "The speed of light is 299,792 km/s.",
        "Octopuses have three hearts.",
        "Honey never spoils."
    ]
    for fact in initial_facts:
        agent.add_fact(fact)
    st.success("Δ-Zero is ready!")
