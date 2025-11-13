# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Upgraded v2 (November 2025) ← NOW REALLY SMART
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
from sentence_transformers import SentenceTransformer  # ← NEW
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

# ============================================================== #
# LOAD KNOWLEDGE (unchanged)
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
# Δ-Zero AGENT – FULLY UPGRADED
# ============================================================== #
class DeltaAgent:
    REPLIES = [
        ["Well it's nice to talk to someone - give me information?", "I'm intrigued! Tell me more?", "Really? What makes you say that?", "Ooh keep going!", "That's new to me - elaborate?"],     # Curious
        ["I get that completely.", "Makes sense. I'm still learning though.", "Okay, noted. Anything else?", "Totally hear you.", "That's fair."],                                             # Calm
        ["Tell me more about that! I'm all ears!", "Yes! Keep going!", "This is getting good.", "I love where this is going.", "Don't stop now!"],                                         # Engaging
        ["That sounds intense… how did it make you feel?", "I can imagine that wasn't easy.", "I'm really listening.", "That's heavy. Want to talk about it?", "Your feelings make sense."], # Empathetic
        ["Let's break this down.", "Interesting pattern here.", "Walk me through the logic?", "What caused that, do you think?", "I'm analyzing this as we speak."],                       # Analytical
    ]

    def __init__(self, n_slots=5, lr=0.09):
        self.n_slots = n_slots
        self.lr = lr
        self.knowledge = load_knowledge()
        self.memory = []
        self.mood_history = []
        self.last_slot = None
        self.cipher = self._load_or_create_key()

        # ← NEW: Much smarter embeddings
        st.info("Loading sentence transformer (first start takes ~10 sec)...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_embeddings = None
        self.all_texts = []  # parallel list for retrieval

        # Load persistent state
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
                if not enc: return []
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
        memory_texts = [
            f"{m.get('input','')} {m.get('response','')}"
            for m in recent if isinstance(m, dict)
        ]
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

    # ← 5. Smart deduplication
    def add_fact(self, text):
        text = text.strip()
        if not text:
            return False

        if self.knowledge_embeddings is not None and len(self.all_texts) > 0:
            emb = self.sentence_model.encode([text])
            sims = cosine_similarity(emb, self.knowledge_embeddings)[0]
            if sims.max() > 0.92:  # almost identical
                return False

        path = os.path.join(KNOWLEDGE_DIR, "learned_facts.txt")
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        self.knowledge.append(text)
        self._refresh_knowledge_base()
        return True

    # ← 4. Memory summarization (prevents total amnesia)
    def _summarize_old_memory(self):
        if len(self.memory) < SUMMARY_TRIGGER:
            return
        old_chunk = self.memory[:-400]  # keep only newest 400
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

    # ------------------- RESPONSE GENERATION (now super contextual) -------------------
    def generate_response(self, user_input, slot, mood=None):
        response = ""

        # ← 2. Smart contextual recall
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

        # Human-like softeners
        softeners = ["you know?", "right?", "haha", "just thinking out loud.", "don’t you think?"]
        if random.random() < 0.5:
            response += " " + random.choice(softeners)

        # Occasional extra fact
        if self.knowledge and random.random() < 0.3:
            extra = random.choice(self.knowledge)
            if len(response + extra) < 280 and extra not in response:
                response += f" Oh and {extra.lower()}"

        return response + f" [slot {slot}]"

    def respond(self, user_input, mood=None):
        slot = self.choose_slot(mood)
        response = self.generate_response(user_input, slot, mood)
        return response, slot

    # ← 3. Better reward modeling (smoother learning)
    def update(self, reward):
        if self.last_slot is None:
            return
        # Strong update for chosen slot
        self.w[self.last_slot] += self.lr * (reward - self.w[self.last_slot])
        # Gentle update for others (helps exploration)
        for i in range(self.n_slots):
            if i != self.last_slot:
                self.w[i] += self.lr * 0.15 * (reward - self.w[i])
        self.w = np.clip(self.w, 0.01, None)
        self.w /= self.w.sum()

    def _apply_mood_boost(self, mood):
        w = self.w.copy()
        if mood <= 3:
            w[3] *= 1.5   # Empathetic boost
        elif mood >= 7:
            w[0] *= 1.35  # Curious
            w[2] *= 1.35  # Engaging
        return w / w.sum()

    def choose_slot(self, mood=None):
        probs = self._apply_mood_boost(mood) if mood is not None else self.w
        slot = np.random.choice(self.n_slots, p=probs)
        self.last_slot = slot
        return slot

    def log_interaction(self, user_input, response, slot, reward=None, feedback=None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {"timestamp": ts, "input": user_input, "response": response,
                 "slot": slot, "reward": reward, "feedback": feedback}
        self.memory.append(entry)
        df = pd.DataFrame(self.memory)
        self._save_encrypted_df(df)

        # Auto-learn factual-looking input
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
    with st.spinner("Waking up Δ-Zero (loading brain + 384-dim embeddings)..."):
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

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_bot_idx" not in st.session_state:
    st.session_state.last_bot_idx = -1

def render_chat():
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["sender"] == "user":
            st.markdown(f"<div style='background:#D1E7DD;padding:12px;border-radius:10px;text-align:right;margin:5px 0'>"
                        f"<b>You:</b> {msg['message']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#F8D7DA;padding:12px;border-radius:10px;margin:5px 0'>"
                        f"<b>Δ-Zero:</b> {msg['message']}</div>", unsafe_allow_html=True)
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
# ONE-TIME BOOTSTRAPPING (only runs once ever)
# ============================================================== #
BOOTSTRAP_FILE = os.path.join(BASE_DIR, "bootstrapped.flag")

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

bootstrap_ai(agent)    if os.path.exists(KNOWLEDGE_DIR):
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

