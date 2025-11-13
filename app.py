# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Persistent, Learning, Mood-Aware Conversational AI
# by JCB + upgrades by Grok (2025 edition)
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
import time
import atexit

# ============================================================== #
# CONFIG PATHS
# ============================================================== #
BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
BRAIN_FILE = os.path.join(BASE_DIR, "global_brain.pkl")
DATA_FILE = os.path.join(BASE_DIR, "chat_log.enc")
MOOD_FILE = os.path.join(BASE_DIR, "mood_history.pkl")
KEY_FILE = os.path.join(BASE_DIR, "secret.key")
VECTORIZER_FILE = os.path.join(BASE_DIR, "vectorizer.pkl")
KNOWLEDGE_VECTORS_FILE = os.path.join(BASE_DIR, "knowledge_vectors.npy")
MAX_MEMORY = 500

# ============================================================== #
# ENCRYPTION SETUP
# ============================================================== #
def load_or_create_key():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        return key

key = load_or_create_key()
cipher = Fernet(key)

# ============================================================== #
# LOAD KNOWLEDGE ONCE
# ============================================================== #
@st.cache_resource
def load_knowledge():
    knowledge = []
    if os.path.exists(KNOWLEDGE_DIR):
        for f in os.listdir(KNOWLEDGE_DIR):
            if f.endswith(".txt"):
                try:
                    with open(os.path.join(KNOWLEDGE_DIR, f), "r", encoding="utf-8") as file:
                        for line in file:
                            text = line.strip()
                            if text and len(text) > 10:  # filter garbage
                                knowledge.append(text)
                except Exception as e:
                    st.error(f"Error loading {f}: {e}")
    return knowledge

knowledge_base = load_knowledge()

# ============================================================== #
# Δ-ZERO AGENT – NOW FULLY PERSISTENT & SMARTER
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
            "I get that. But I'm still learning. Give me another fact?",
            "OK. Tell me more facts to help me understand.",
            "I hear you. But I need help understanding.",
            "That's interesting. Keep feeding me knowledge."
        ],
        "engaging": [
            "Tell me more about that! I'm learning from you!",
            "Give me more information!",
            "I love where this is going. Keep talking.",
            "Yes! This is exactly how I grow."
        ],
        "empathetic": [
            "That sounds emotional.",
            "How did that make you feel? And what are feelings?",
            "Interesting perspective. But explain more?",
            "I want to understand the human side of this."
        ],
        "analytical": [
            "Let's analyze this a bit. Tell me five random facts? It'll help me understand.",
            "Interesting pattern. Explain it?",
            "I like the logic behind that. Explain it more?",
            "Help me build a mental model — give me data points."
        ]
    }

    def __init__(self, n_slots=5, lr=0.07):
        self.n_slots = n_slots
        self.lr = lr
        self.message_count = 0
        self.start_time = datetime.now()
        self.mood = "curious"
        self.mood_history = []
        self.short_term_memory = []  # last 15 exchanges
        self.long_term_memory = []   # encrypted on disk
        self.knowledge = knowledge_base

        # Load or build knowledge index
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
                self.knowledge_vectors = np.zeros((0, 1))

        # Load encrypted chat log
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
            except:
                self.long_term_memory = []

    def save_chat_log(self):
        data = pickle.dumps(self.long_term_memory[-MAX_MEMORY:])
        encrypted = cipher.encrypt(data)
        with open(DATA_FILE, "wb") as f:
            f.write(encrypted)

    def remember(self, user_msg, bot_msg):
        self.message_count += 1
        entry = {
            "timestamp": datetime.now(),
            "user": user_msg,
            "bot": bot_msg,
            "mood": self.mood
        }
        self.short_term_memory.append(entry)
        self.long_term_memory.append(entry)
        if len(self.short_term_memory) > 15:
            self.short_term_memory.pop(0)

        # Mood drift based on keywords (simple but effective)
        lower = user_msg.lower()
        if any(w in lower for w in ["love", "amazing", "wow", "happy", "excited"]):
            self.mood = "engaging"
        elif any(w in lower for w in ["sad", "feel", "emotion", "hurt", "afraid"]):
            self.mood = "empathetic"
        elif any(w in lower for w in ["why", "how", "explain", "logic", "because"]):
            self.mood = "analytical"
        elif any(w in lower for w in ["fact", "did you know", "actually", "source"]):
            self.mood = "calm"
        else:
            self.mood = random.choice(self.MOODS)  # curiosity is default

        self.mood_history.append((datetime.now(), self.mood))

    def retrieve_relevant_knowledge(self, query, top_k=3):
        if len(self.knowledge) == 0 or self.knowledge_vectors.shape[0] == 0:
            return []
        query_vec = self.vectorizer.transform([query]).toarray()
        sims = cosine_similarity(query_vec, self.knowledge_vectors)[0]
        best_idx = np.argsort(sims)[-top_k:][::-1]
        return [self.knowledge[i] for i in best_idx if sims[i] > 0.2]

    def generate_reply(self, user_input):
        facts = self.retrieve_relevant_knowledge(user_input)
        context = "\n".join([f"User said: {m['user']}" for m in self.short_term_memory[-5:]])
        
        prompt = f"Recent context:\n{context}\nRelevant facts I know:\n" + "\n".join(facts[:2]) + f"\nCurrent mood: {self.mood}\nUser says: {user_input}\nΔ-Zero replies (curious, learning, asks for more info):"

        # Mood-weighted reply selection
        candidates = self.REPLY_TEMPLATES[self.mood]
        reply = random.choice(candidates)

        # Inject real retrieved fact sometimes
        if facts and random.random() < 0.4:
            reply = f"I recall: \"{random.choice(facts)}\" → " + random.choice(candidates)

        return reply

# ============================================================== #
# PERSISTENT AGENT LOADER (THE MAGIC THAT STOPS COUNTER RESET)
# ============================================================== #
def get_agent():
    if "agent" not in st.session_state:
        if os.path.exists(BRAIN_FILE):
            try:
                with open(BRAIN_FILE, "rb") as f:
                    st.session_state.agent = pickle.load(f)
                st.success("🧠 Brain restored — I remember you perfectly.")
            except:
                st.session_state.agent = DeltaAgent()
                st.info("🧠 New brain initialized — teach me everything!")
        else:
            st.session_state.agent = DeltaAgent()
            st.info("🧠 First time booting up — hello, human!")
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
st.title("Δ-Zero — Your Personal Learning AI")
st.caption("I never forget. I evolve with every word you say.")

agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if prompt := st.chat_input("Talk to Δ-Zero..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = agent.generate_reply(prompt)
            agent.remember(prompt, reply)
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

# Sidebar stats
with st.sidebar:
    st.header("Δ-Zero Stats")
    st.write(f"Messages learned: {agent.message_count}")
    st.write(f"Current mood: **{agent.mood.upper()}**")
    st.write(f"Knowledge facts: {len(agent.knowledge)}")
    st.write(f"Session uptime: {(datetime.now() - agent.start_time).strftime('%H:%M:%S')}")

    if st.button("Wipe memory & start fresh"):
        for f in [BRAIN_FILE, DATA_FILE, MOOD_FILE, VECTORIZER_FILE, KNOWLEDGE_VECTORS_FILE]:
            if os.path.exists(f):
                os.remove(f)
        st.success("Memory wiped. Restarting as baby Δ-Zero...")
        st.experimental_rerun()        self.lr = lr
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


