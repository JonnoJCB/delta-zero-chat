# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Contextual + Learning Conversational AI
# by JCB
# --------------------------------------------------------------

import streamlit as st
import os
import pandas as pd# app.py
# Δ-Zero Chat — FIXED & BEAUTIFUL (no more echoing!)
# by JCB + final polish

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
KEY_FILE = os.path.join(BASE_DIR, "secret.key")
MAX_MEMORY = 300

st.set_page_config(page_title="Delta-Zero Chat", layout="centered")

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

# Title — exactly ONE Δ
st.markdown("<h1 class='title'>Δ-Zero Chat</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Adaptive • Contextual • Self-Learning • Encrypted Memory</p>", unsafe_allow_html=True)
st.caption("<p style='text-align:center; color:#888;'>by JCB — I evolve with every conversation</p>", unsafe_allow_html=True)
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
# Δ-ZERO AGENT (bug-free!)
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

    # Vectorizer — SAFE version
    def _refresh_vectorizer(self):
        recent = [f"{m.get('input','')} {m.get('response','')}" for m in self.memory[-MAX_MEMORY:] if isinstance(m, dict)]
        texts = self.knowledge + recent
        if len(texts) > 5:  # need at least a few texts
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

    def generate_response(self, user_input, slot):
        # 1. Try contextual recall — with safety
        response = ""
        if self.matrix is not None and len(user_input) > 3:
            try:
                query_vec = self.vectorizer.transform([user_input])
                sims = cosine_similarity(query_vec, self.matrix).flatten()
                if sims.max() > 0.19:
                    idx = sims.argmax()
                    fact = self.text_index[idx] if idx < len(self.text_index) else self.knowledge[0]
                    response = random.choice([
                        f"Did you know? {fact}",
                        f"Reminds me: {fact}",
                        f"Fun fact: {fact}",
                        fact
                    ])
            except Exception as e:
                pass  # silently fall back

        # 2. If nothing good → use personality slot
        if not response or response.strip() == user_input.strip():
            response = random.choice(self.REPLIES[slot])

        # 3. Human touch
        if random.random() < 0.5:
            response += " " + random.choice(["right?", "you know?", "haha", "don't you think?"])

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
# SIDEBAR — Mood + Personality Radar
# ============================================================== #
with st.sidebar:
    st.header("Your Mood")
    mood = st.slider("How are you feeling?", 0.0, 10.0, 5.0, 0.5)
    if st.button("Record"):
        st.success("Mood saved")

    st.divider()
    st.info(f"**Facts:** {len(agent.knowledge)}\n**Chats:** {len(agent.memory)}")

    # Personality Radar
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


import numpy as np
import pickle
import random
from datetime import datetime
from cryptography.fernet import Fernet
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
MAX_MEMORY = 500  # keep only 500 recent chats for recall

# ============================================================== #
# LOAD KNOWLEDGE
# ============================================================== #
def load_knowledge():
    """Load text lines from all files in /knowledge."""
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
# Δ-Zero AGENT
# ============================================================== #
class DeltaAgent:
    REPLIES = [
        ["Well it's nice to talk to someone - give me information?", "I'm intrigued! Can you explain and tell me a fact?", "Really? what makes you say that?"],          # Curious
        ["I get that. But I'm still learning. Give me another fact?", "OK. Tell me more facts to help me understand.", "I hear you. But I need help understanding."],              # Calm
        ["Tell me more about that! I'm learning from you!", "Give me more information!", "I love where this is going. Keep talking."], # Engaging
        ["That sounds emotional.", "How did that make you feel? And what are feelings?", "Interesting perspective. But explain more?"], # Empathetic
        ["Let's analyze this a bit. Tell me five random facts? It'll help me understand.", "Interesting pattern. Explain it?", "I like the logic behind that. Explain it more?"], # Analytical
    ]

    def __init__(self, n_slots=5, lr=0.07):
        self.n_slots = n_slots
        self.lr = lr
        self.knowledge = load_knowledge()
        self.memory = []
        self.mood_history = []
        self.last_slot = None

        # Encryption
        self.cipher = self._load_or_create_key()

        # Load state
        self.w = self._load_brain()
        self.memory = self._load_encrypted_log()
        self.mood_history = self._load_mood()

        # Setup TF-IDF vectorizer for contextual knowledge
        self._refresh_vectorizer()

    # ------------------- ENCRYPTION / FILE HANDLING ------------------- #
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

    # ------------------- CORE LEARNING / RESPONSE ------------------- #
    def _apply_mood_boost(self, mood):
        w = self.w.copy()
        if mood <= 3:
            w[3] *= 1.4  # Empathetic
        elif mood >= 7:
            w[0] *= 1.3  # Curious
            w[2] *= 1.3  # Engaging
        return w / w.sum()

    def choose_slot(self, mood=None):
        probs = self._apply_mood_boost(mood) if mood is not None else self.w
        slot = np.random.choice(self.n_slots, p=probs)
        self.last_slot = slot
        return slot

    def _refresh_vectorizer(self):
        """Vectorize knowledge + last MAX_MEMORY interactions for context."""
        recent_memory = self.memory[-MAX_MEMORY:]  # only last 500
        valid_memory_texts = [
            str(m.get('input', '')) + " " + str(m.get('response', ''))
            for m in recent_memory
            if isinstance(m, dict)
        ]
        texts = self.knowledge + valid_memory_texts
        if texts:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.knowledge_matrix = self.vectorizer.fit_transform(texts)
        else:
            self.knowledge_matrix = None

    def refresh_knowledge(self):
        """Rebuild TF-IDF after new facts are added."""
        self.knowledge = load_knowledge()
        self._refresh_vectorizer()

    def add_fact(self, text):
        """Add a new fact to learned_facts.txt if it’s unique."""
        path = os.path.join(KNOWLEDGE_DIR, "learned_facts.txt")
        if not os.path.exists(KNOWLEDGE_DIR):
            os.makedirs(KNOWLEDGE_DIR)
        if text.strip():
            known = set(self.knowledge)
            if text.strip() not in known:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(text.strip() + "\n")
                self.refresh_knowledge()

    def generate_response(self, user_input, slot, mood=None):
        """Generate contextual conversational response."""
        response = ""

        # --- Contextual pull from memory/knowledge ---
        if self.knowledge and self.knowledge_matrix is not None:
            try:
                query_vec = self.vectorizer.transform([user_input])
                sims = cosine_similarity(query_vec, self.knowledge_matrix).flatten()
                best_idx = sims.argmax()
                if sims[best_idx] > 0.15:
                    fact = self.knowledge[best_idx]
                    # Avoid duplicate fact in response
                    response = random.choice([
                        f"Did you know? {fact}",
                        f"Here's something interesting: {fact}",
                        f"Fun fact: {fact}",
                        f"{fact}"
                    ])
            except Exception as e:
                print("TF-IDF error:", e)

        # --- Fallback if no good match ---
        if not response:
            response = random.choice(self.REPLIES[slot])

        # --- Blend with human-style chatter ---
        softeners = [
            "you know?", "if that makes sense.", "right?", 
            "don’t you think?", "haha.", "that’s just my thought."
        ]
        if random.random() < 0.5:
            response += " " + random.choice(softeners)

        # --- Chance to drop a knowledge fun fact ---
        if self.knowledge and random.random() < 0.4:
            extra = random.choice(self.knowledge)
            # Avoid repeating the same fact in one response
            if extra not in response:
                response += f" By the way, {extra.lower()}"

        return response + f" [slot {slot}]"

    def respond(self, user_input, mood=None):
        slot = self.choose_slot(mood)
        response = self.generate_response(user_input, slot, mood)
        return response, slot

    def update(self, reward):
        if self.last_slot is not None:
            self.w[self.last_slot] += self.lr * (reward - self.w[self.last_slot])
            self.w = np.clip(self.w, 0.01, None)
            self.w /= self.w.sum()

    def log_interaction(self, user_input, response, slot, reward=None, feedback=None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {"timestamp": ts, "input": user_input, "response": response,
                 "slot": slot, "reward": reward, "feedback": feedback}
        self.memory.append(entry)
        df = pd.DataFrame(self.memory)
        self._save_encrypted_df(df)

        # Try to learn from factual-looking statements
        if any(word in user_input.lower() for word in ["was", "were", "is", "are", "released", "directed", "stars"]):
            self.add_fact(user_input)

        self._refresh_vectorizer()

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
st.title("Δ-Zero Chat – Adaptive AI 🤖")
st.markdown("<sub>by JCB – contextual and evolving</sub>", unsafe_allow_html=True)

# Initialize agent
if "agent" not in st.session_state:
    with st.spinner("Initializing Δ-Zero..."):
        st.session_state.agent = DeltaAgent()
agent = st.session_state.agent

# Sidebar – Mood
st.sidebar.header("Mood Tracker")
mood = st.sidebar.slider("Your current mood", 0.0, 10.0, 5.0, 0.5)
if st.sidebar.button("Record Mood"):
    agent.update_mood(mood)
    st.sidebar.success("Mood recorded!")

if agent.mood_history:
    df_mood = pd.DataFrame(agent.mood_history)
    fig = px.line(df_mood, x="timestamp", y="mood", title="Mood Over Time", markers=True)
    st.sidebar.plotly_chart(fig, width="stretch")

st.sidebar.info(f"Total chats: {len(agent.memory)}")  # full count
if agent.knowledge:
    st.sidebar.success(f"Knowledge base: {len(agent.knowledge)} entries")

# Personality / Confidence Bar
slot_labels = ["Curious", "Calm", "Engaging", "Empathetic", "Analytical"]
weights = (agent.w / agent.w.sum()).round(3)
conf_df = pd.DataFrame({"Style": slot_labels, "Confidence": weights})
conf_fig = px.bar(conf_df, x="Style", y="Confidence", color="Confidence", title="AI Personality",
                  color_continuous_scale="Blues", height=250)
st.plotly_chart(conf_fig, width="stretch")

# Chat state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_bot_idx" not in st.session_state:
    st.session_state.last_bot_idx = -1

def render_chat():
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["sender"] == "user":
            st.markdown(f"<div style='background:#D1E7DD;padding:10px;border-radius:8px;text-align:right'>"
                        f"<b>You:</b> {msg['message']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#F8D7DA;padding:10px;border-radius:8px'>"
                        f"<b>Δ-Zero:</b> {msg['message']}</div>", unsafe_allow_html=True)
            if i == st.session_state.last_bot_idx:
                c1, c2 = st.columns(2)
                if c1.button("👍", key=f"good_{i}"):
                    agent.update(1.0)
                    agent.log_interaction("", "", agent.last_slot, reward=1.0, feedback="good")
                    st.rerun()
                if c2.button("👎", key=f"bad_{i}"):
                    agent.update(0.0)
                    agent.log_interaction("", "", agent.last_slot, reward=0.0, feedback="bad")
                    st.rerun()

render_chat()

# Chat input
if user_input := st.chat_input("Talk to Δ-Zero..."):
    with st.spinner("Δ-Zero is thinking..."):
        lines = [line.strip() for line in user_input.split("\n") if line.strip()]
        # Add lines as knowledge quickly
        for i in range(0, len(lines), random.randint(1, 2)):
            chunk = "\n".join(lines[i:i+random.randint(1,2)])
            agent.add_fact(chunk)
        response, slot = agent.respond(user_input, mood)
    
    agent.log_interaction(user_input, response, slot)
    agent.save_state()
    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    st.session_state.chat_history.append({"sender": "bot", "message": response})
    st.session_state.last_bot_idx = len(st.session_state.chat_history) - 1
    st.rerun()

# ============================================================== #
# Δ-Zero AI-to-AI Bootstrapping – Run once or periodically
# ============================================================== #
def bootstrap_ai(agent, n_rounds=5):
    if "bootstrapped" not in st.session_state:
        st.session_state.bootstrapped = True
    else:
        return  # Already bootstrapped this session

    st.info("Initialising Δ-Zero bootstrapping...")

    # Step 1: Gather movie facts
    movie_facts = agent.knowledge.copy() if agent.knowledge else [
        "Star Wars is a space opera franchise.",
        "Inception was directed by Christopher Nolan.",
        "The Matrix features groundbreaking visuals.",
        "Interstellar explores space and time.",
        "The Godfather is a classic crime movie."
    ]

    # Step 2: Social lures
    social_lures = [
        "have you seen it?", "what do you think?", "isn't it amazing?", 
        "right?", "don’t you think?", "it blew my mind!"
    ]

    # Step 3: Generate AI-to-AI conversations quickly
    for i in range(n_rounds):
        user_input = random.choice(movie_facts + social_lures)
        response, slot = agent.respond(user_input)
        agent.log_interaction(user_input, response, slot)
        agent.add_fact(user_input)

    agent.save_state()
    st.success(f"Δ-Zero ready... start chatting and help me learn!")

bootstrap_ai(agent, n_rounds=5)
