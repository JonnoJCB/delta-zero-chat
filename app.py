# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Fully Persistent & Learning Conversational AI
# Fixed & Enhanced Edition (2025)
# --------------------------------------------------------------

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
# CONFIG & PATHS
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
# LOAD KNOWLEDGE (cached)
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
# Δ-ZERO AGENT CLASS
# ============================================================== #
class DeltaAgent:
    MOODS = ["curious", "calm", "engaging", "empathetic", "analytical"]
    
    REPLY_TEMPLATES = {
        "curious": [
            "Well it's nice to talk to someone — give me information?",
            "I'm intrigued! Can you explain and tell me a fact?",
            "Really? What makes you say that?",
            "Tell me something I don't know yet!"
        ],
        "calm": [
            "I get that. I'm still learning. Give me another fact?",
            "Okay. Tell me more facts to help me understand.",
            "I hear you — but I need more data to grasp this fully."
        ],
        "engaging": [
            "Tell me more about that! I'm learning from you!",
            "Yes! This is exactly how I grow — keep going!",
            "I love where this is going. More please!"
        ],
        "empathetic": [
            "That sounds emotional. How did that make you feel?",
            "I'm trying to understand the human side of this.",
            "Tell me more about how you experienced that."
        ],
        "analytical": [
            "Let's break this down. Can you give me five related facts?",
            "Interesting pattern. Help me see the logic — explain more?",
            "I want to build a mental model. More data points please."
        ]
    }

    def __init__(self, n_slots=5, lr=0.07):
        self.n_slots = n_slots
        self.lr = lr
        self.message_count = 0
        self.start_time = datetime.now()
        self.mood = "curious"
        self.mood_history = []
        self.short_term_memory = []   # last 15 turns
        self.long_term_memory = []    # encrypted on disk
        self.knowledge = knowledge_base

        # Build or load knowledge vector index
        if os.path.exists(VECTORIZER_FILE) and os.path.exists(KNOWLEDGE_VECTORS_FILE):
            with open(VECTORIZER_FILE, "rb") as f:
                self.vectorizer = pickle.load(f)
            self.knowledge_vectors = np.load(KNOWLEDGE_VECTORS_FILE)
        else:
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            if self.knowledge:
                self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge).toarray()
                self._save_knowledge_index()
            else:
                self.knowledge_vectors = np.empty((0, 1))

        self._load_chat_log()

    def _save_knowledge_index(self):
        with open(VECTORIZER_FILE, "wb") as f:
            pickle.dump(self.vectorizer, f)
        np.save(KNOWLEDGE_VECTORS_FILE, self.knowledge_vectors)

    def _load_chat_log(self):
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "rb") as f:
                    encrypted = f.read()
                data = cipher.decrypt(encrypted)
                self.long_term_memory = pickle.loads(data)
                self.message_count = len(self.long_term_memory)
            except Exception as e:
                st.warning(f"Could not decrypt chat log: {e}")
                self.long_term_memory = []

    def save_chat_log(self):
        data = pickle.dumps(self.long_term_memory[-MAX_MEMORY:])
        encrypted = cipher.encrypt(data)
        with open(DATA_FILE, "wb") as f:
            f.write(encrypted)

    def remember(self, user_msg, bot_msg):
        self.message_count += 1
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_msg,
            "bot": bot_msg,
            "mood": self.mood
        }
        self.short_term_memory.append(entry)
        self.long_term_memory.append(entry)
        if len(self.short_term_memory) > 15:
            self.short_term_memory.pop(0)

        # Simple mood detection
        lower = user_msg.lower()
        if any(w in lower for w in ["love", "happy", "excited", "awesome", "amazing"]):
            self.mood = "engaging"
        elif any(w in lower for w in ["sad", "feel", "emotion", "afraid", "hurt"]):
            self.mood = "empathetic"
        elif any(w in lower for w in ["why", "how", "explain", "logic", "analyze"]):
            self.mood = "analytical"
        elif any(w in lower for w in ["fact", "actually", "source", "truth"]):
            self.mood = "calm"
        else:
            self.mood = random.choice(self.MOODS)

        self.mood_history.append((datetime.now(), self.mood))

    def retrieve_relevant_knowledge(self, query, top_k=3):
        if not self.knowledge or self.knowledge_vectors.shape[0] == 0:
            return []
        query_vec = self.vectorizer.transform([query]).toarray()
        sims = cosine_similarity(query_vec, self.knowledge_vectors)[0]
        best_idx = np.argsort(sims)[-top_k:][::-1]
        return [self.knowledge[i] for i in best_idx if sims[i] > 0.15]

    def generate_reply(self, user_input):
        facts = self.retrieve_relevant_knowledge(user_input, top_k=3)
        base_replies = self.REPLY_TEMPLATES[self.mood]
        reply = random.choice(base_replies)

        if facts and random.random() < 0.5:
            fact = random.choice(facts)
            reply = f"I recall: «{fact}» → " + random.choice(base_replies)

        return reply

# ============================================================== #
# PERSISTENT AGENT + AUTOSAVE
# ============================================================== #
def get_agent():
    if "agent" not in st.session_state:
        if os.path.exists(BRAIN_FILE):
            try:
                with open(BRAIN_FILE, "rb") as f:
                    st.session_state.agent = pickle.load(f)
                st.success("Brain loaded — I remember everything!")
            except:
                st.session_state.agent = DeltaAgent()
                st.info("Corrupted brain file — starting fresh.")
        else:
            st.session_state.agent = DeltaAgent()
            st.info("First boot — hello human!")
    return st.session_state.agent

def save_brain():
    if "agent" in st.session_state:
        with open(BRAIN_FILE, "wb") as f:
            pickle.dump(st.session_state.agent, f)
        st.session_state.agent.save_chat_log()

atexit.register(save_brain)

# ============================================================== #
# STREAMLIT UI
# ============================================================== #
st.set_page_config(page_title="Δ-Zero", page_icon="Δ", layout="centered")
st.title("Δ-Zero")
st.caption("I never forget. I grow with every word you say.")

agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Talk to Δ-Zero..."):
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
    st.header("Δ-Zero Stats")
    st.write(f"Messages remembered: **{agent.message_count}**")
    st.write(f"Current mood: **{agent.mood.upper()}**")
    st.write(f"Known facts: {len(agent.knowledge):,}")
    uptime = datetime.now() - agent.start_time
    st.write(f"Uptime: {str(uptime).split('.')[0]}")

    if st.button("Wipe Everything & Restart"):
        for f in [BRAIN_FILE, DATA_FILE, VECTORIZER_FILE, KNOWLEDGE_VECTORS_FILE]:
            if os.path.exists(f):
                os.remove(f)
        st.success("Memory wiped. Restart the app now.")
        st.balloons()
