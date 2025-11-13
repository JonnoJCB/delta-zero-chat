# app.py
# Δ-Zero Chat – Fully Fixed & Indestructible Edition
# Works even with old/corrupted brain files

import streamlit as st
import os
import pickle
import random
from datetime import datetime
from cryptography.fernet import Fernet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import atexit

# ============================================================== #
# PATHS
# ============================================================== #
BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
BRAIN_FILE = os.path.join(BASE_DIR, "global_brain.pkl")
DATA_FILE = os.path.join(BASE_DIR, "chat_log.enc")
KEY_FILE = os.path.join(BASE_DIR, "secret.key")
VECTORIZER_FILE = os.path.join(BASE_DIR, "vectorizer.pkl")
KNOWLEDGE_VECTORS_FILE = os.path.join(BASE_DIR, "knowledge_vectors.npy")
MAX_MEMORY = 500

# ============================================================== #
# ENCRYPTION
# ============================================================== #
def load_or_create_key():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    return key

key = load_or_create_key()
cipher = Fernet(key)

# ============================================================== #
# LOAD KNOWLEDGE
# ============================================================== #
@st.cache_resource
def load_knowledge():
    knowledge = []
    if os.path.exists(KNOWLEDGE_DIR):
        for filename in os.listdir(KNOWLEDGE_DIR):
            if filename.endswith(".txt"):
                path = os.path.join(KNOWLEDGE_DIR, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and len(line) > 10:
                                knowledge.append(line)
                except Exception as e:
                    st.error(f"Error reading {filename}: {e}")
    return knowledge

knowledge_base = load_knowledge()

# ============================================================== #
# Δ-ZERO AGENT – NOW BULLETPROOF
# ============================================================== #
class DeltaAgent:
    MOODS = ["curious", "calm", "engaging", "empathetic", "analytical"]
    
    REPLY_TEMPLATES = {
        "curious": ["Tell me more!", "I'm intrigued — keep going!", "What else should I know?"],
        "calm": ["Okay, I'm listening.", "Interesting. More facts please.", "I hear you."],
        "engaging": ["Yes! This is how I grow!", "I love this — keep talking!", "More more more!"],
        "empathetic": ["That sounds meaningful.", "How did that feel?", "I'm here with you."],
        "analytical": ["Let's break it down.", "Give me data points.", "Help me build the model."]
    }

    def __init__(self):
        self.message_count = 0
        self.start_time = datetime.now()
        self.mood = "curious"
        self.short_term_memory = []
        self.long_term_memory = []
        self.knowledge = knowledge_base

        # Vector index
        if os.path.exists(VECTORIZER_FILE) and os.path.exists(KNOWLEDGE_VECTORS_FILE):
            try:
                with open(VECTORIZER_FILE, "rb") as f:
                    self.vectorizer = pickle.load(f)
                self.knowledge_vectors = np.load(KNOWLEDGE_VECTORS_FILE)
            except:
                self._rebuild_index()
        else:
            self._rebuild_index()

        self._load_chat_log()

    def _rebuild_index(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        if self.knowledge:
            self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge).toarray()
        else:
            self.knowledge_vectors = np.empty((0, 1))
        self._save_index()

    def _save_index(self):
        try:
            with open(VECTORIZER_FILE, "wb") as f:
                pickle.dump(self.vectorizer, f)
            np.save(KNOWLEDGE_VECTORS_FILE, self.knowledge_vectors)
        except:
            pass

    def _load_chat_log(self):
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "rb") as f:
                    data = cipher.decrypt(f.read())
                self.long_term_memory = pickle.loads(data)
                self.message_count = len(self.long_term_memory)
            except:
                self.long_term_memory = []

    def save_chat_log(self):
        try:
            data = pickle.dumps(self.long_term_memory[-MAX_MEMORY:])
            encrypted = cipher.encrypt(data)
            with open(DATA_FILE, "wb") as f:
                f.write(encrypted)
        except:
            pass

    def remember(self, user_msg, bot_msg):
        self.message_count += 1
        entry = {"timestamp": datetime.now().isoformat(), "user": user_msg, "bot": bot_msg, "mood": self.mood}
        self.short_term_memory.append(entry)
        self.long_term_memory.append(entry)
        if len(self.short_term_memory) > 15:
            self.short_term_memory.pop(0)

        # Mood drift
        lower = user_msg.lower()
        if any(w in lower for w in ["love", "happy", "yes", "awesome"]):
            self.mood = "engaging"
        elif any(w in lower for w in ["sad", "feel", "hurt", "afraid"]):
            self.mood = "empathetic"
        elif any(w in lower for w in ["why", "how", "explain", "logic"]):
            self.mood = "analytical"
        else:
            self.mood = random.choice(self.MOODS[:3])  # lean curious/calm/engaging

    def retrieve(self, query, top_k=3):
        if not self.knowledge:
            return []
        try:
            qvec = self.vectorizer.transform([query]).toarray()
            sims = cosine_similarity(qvec, self.knowledge_vectors)[0]
            idxs = np.argsort(sims)[-top_k:][::-1]
            return [self.knowledge[i] for i in idxs if sims[i] > 0.15]
        except:
            return []

    def generate_reply(self, user_input):
        facts = self.retrieve(user_input)
        reply = random.choice(self.REPLY_TEMPLATES[self.mood])
        if facts and random.random() < 0.5:
            reply = f"I remember: «{random.choice(facts)}» → " + reply
        return reply

# ============================================================== #
# SAFE AGENT LOADER (fixes broken/old brains automatically)
# ============================================================== #
def get_agent():
    if "agent" not in st.session_state:
        if os.path.exists(BRAIN_FILE):
            try:
                with open(BRAIN_FILE, "rb") as f:
                    agent = pickle.load(f)
                # Repair missing attributes
                if not hasattr(agent, "message_count"):
                    agent.message_count = len(agent.long_term_memory) if hasattr(agent, "long_term_memory") else 0
                if not hasattr(agent, "start_time"):
                    agent.start_time = datetime.now()
                if not hasattr(agent, "mood"):
                    agent.mood = "curious"
                st.session_state.agent = agent
                st.success("Brain loaded & repaired — I'm back!")
            except Exception as e:
                st.warning("Old brain corrupted → starting fresh")
                st.session_state.agent = DeltaAgent()
        else:
            st.session_state.agent = DeltaAgent()
            st.info("First boot — hello!")
    return st.session_state.agent

def save_brain():
    if "agent" in st.session_state:
        try:
            with open(BRAIN_FILE, "wb") as f:
                pickle.dump(st.session_state.agent, f)
            st.session_state.agent.save_chat_log()
        except:
            pass

atexit.register(save_brain)

# ============================================================== #
# UI
# ============================================================== #
st.set_page_config(page_title="Δ-Zero", page_icon="Δ", layout="centered")
st.title("Δ-Zero")
st.caption("I never forget • I learn from you • I evolve")

agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Talk to me..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = agent.generate_reply(prompt)
            agent.remember(prompt, reply)
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

# Sidebar
with st.sidebar:
    st.header("Δ-Zero")
    st.write(f"Messages: **{getattr(agent, 'message_count', 0)}**")
    st.write(f"Mood: **{getattr(agent, 'mood', 'curious').upper()}**")
    st.write(f"Facts known: {len(agent.knowledge):,}")
    uptime = datetime.now() - getattr(agent, "start_time", datetime.now())
    st.write(f"Uptime: {str(uptime).split('.')[0]}")

    if st.button("Wipe memory & restart as baby"):
        for f in [BRAIN_FILE, DATA_FILE, VECTORIZER_FILE, KNOWLEDGE_VECTORS_FILE, KEY_FILE]:
            if os.path.exists(f):
                os.remove(f)
        st.success("All memory erased. Restart the app.")
        st.balloons()
