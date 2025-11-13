@@ -21,8 +21,6 @@
# CONFIG PATHS
# ============================================================== #
BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGO_FILE = os.path.join(ASSETS_DIR, "logo.jpg")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
BRAIN_FILE = os.path.join(BASE_DIR, "global_brain.pkl")
DATA_FILE = os.path.join(BASE_DIR, "chat_log.enc")
@@ -34,13 +32,17 @@
# LOAD KNOWLEDGE
# ============================================================== #
def load_knowledge():
    """Load text lines from all files in /knowledge."""# --------------------------------------------------------------
# Δ-Zero Chat – Contextual + Learning Conversational AI
# by JCB
# Improved UI + Fast Knowledge Learning
# --------------------------------------------------------------

import streamlit as st
@@ -13,6 +12,7 @@
import random
from datetime import datetime
from cryptography.fernet import Fernet
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
@@ -26,83 +26,13 @@
DATA_FILE = os.path.join(BASE_DIR, "chat_log.enc")
MOOD_FILE = os.path.join(BASE_DIR, "mood_history.pkl")
KEY_FILE = os.path.join(BASE_DIR, "secret.key")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGO_FILE = os.path.join(ASSETS_DIR, "logo.jpg")
MAX_MEMORY = 500

# ============================================================== #
# STREAMLIT PAGE CONFIG
# ============================================================== #
st.set_page_config(page_title="Δ-Zero Chat", layout="wide", page_icon=LOGO_FILE)

# ============================================================== #
# CUSTOM CSS FOR DARK/COOL THEME
# ============================================================== #
st.markdown("""
<style>
/* Body background gradient */
body {
    background: linear-gradient(135deg, #1f1c2c, #3e3a59);
    color: #e0e0e0;
}
/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #2b2a3a;
    color: #f0f0f0;
}
/* Sidebar headers and text readability */
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #ffffff;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
    color: #e0e0e0;
}
/* Header logo */
.header-logo {
    width: 60px;
    margin-right: 15px;
    vertical-align: middle;
}
/* Chat bubbles */
.user-bubble {
    background-color: #4a4c6b;
    color: #ffffff;
    padding: 12px;
    border-radius: 12px;
    margin: 5px 0px;
    text-align: right;
}
.bot-bubble {
    background-color: #6c5ce7;
    color: #ffffff;
    padding: 12px;
    border-radius: 12px;
    margin: 5px 0px;
    text-align: left;
}
/* Buttons */
.stButton>button {
    background-color: #6a5acd;
    color: white;
    border-radius: 8px;
    padding: 5px 10px;
}
.stButton>button:hover {
    background-color: #836fff;
}
/* Chat input */
div.stTextInput>div>input {
    background-color: #3a3a55;
    color: #ffffff;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
MAX_MEMORY = 500  # keep only 500 recent chats for recall

# ============================================================== #
# LOAD KNOWLEDGE
# ============================================================== #
def load_knowledge():
    """Load text lines from all files in /knowledge."""
    knowledge = []
    if os.path.exists(KNOWLEDGE_DIR):
        for f in os.listdir(KNOWLEDGE_DIR):
@@ -123,9 +53,9 @@ def load_knowledge():
class DeltaAgent:
    REPLIES = [
        ["Well it's nice to talk to someone - give me information?", "I'm intrigued! Can you explain and tell me a fact?", "Really? what makes you say that?"],          # Curious
        ["I get that. But I'm still learning. Give me another fact?", "OK. Tell me more facts to help me understand.", "I hear you. But I need help understanding."],    # Calm
        ["Tell me more about that! I'm learning from you!", "Give me more information!", "I love where this is going. Keep talking."],                                   # Engaging
        ["That sounds emotional.", "How did that make you feel? And what are feelings?", "Interesting perspective. But explain more?"],                                # Empathetic
        ["I get that. But I'm still learning. Give me another fact?", "OK. Tell me more facts to help me understand.", "I hear you. But I need help understanding."],              # Calm
        ["Tell me more about that! I'm learning from you!", "Give me more information!", "I love where this is going. Keep talking."], # Engaging
        ["That sounds emotional.", "How did that make you feel? And what are feelings?", "Interesting perspective. But explain more?"], # Empathetic
        ["Let's analyze this a bit. Tell me five random facts? It'll help me understand.", "Interesting pattern. Explain it?", "I like the logic behind that. Explain it more?"], # Analytical
    ]

@@ -207,7 +137,8 @@ def choose_slot(self, mood=None):
        return slot

    def _refresh_vectorizer(self):
        recent_memory = self.memory[-MAX_MEMORY:]
        """Vectorize knowledge + last MAX_MEMORY interactions for context."""
        recent_memory = self.memory[-MAX_MEMORY:]  # only last 500
        valid_memory_texts = [
            str(m.get('input', '')) + " " + str(m.get('response', ''))
            for m in recent_memory
@@ -221,53 +152,88 @@ def _refresh_vectorizer(self):
            self.knowledge_matrix = None

    def refresh_knowledge(self):
        """Rebuild TF-IDF after new facts are added."""
        self.knowledge = load_knowledge()
        self._refresh_vectorizer()

    # ------------------- FAST FACT ADDITION ------------------- #
    def add_fact(self, text):
        """Add a new fact to learned_facts.txt if it’s unique."""
        path = os.path.join(KNOWLEDGE_DIR, "learned_facts.txt")
        if not os.path.exists(KNOWLEDGE_DIR):
            os.makedirs(KNOWLEDGE_DIR)
        if text.strip() and text.strip() not in self.knowledge:
            # Append to multiple fact files if desired
            path = os.path.join(KNOWLEDGE_DIR, "learned_facts.txt")
            with open(path, "a", encoding="utf-8") as f:
                f.write(text.strip() + "\n")
            # Immediately refresh vectorizer to allow fast responses
            self.refresh_knowledge()
        if text.strip():
            known = set(self.knowledge)
            if text.strip() not in known:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(text.strip() + "\n")
                self.refresh_knowledge()

    def generate_response(self, user_input, slot, mood=None):
        """Generate response quickly using TF-IDF and fallback replies."""
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
                    response = f"{fact}"
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
        # Auto-learn new facts if sentence contains key information
        if any(word in user_input.lower() for word in ["was", "is", "are", "released", "directed", "stars"]):

        # Try to learn from factual-looking statements
        if any(word in user_input.lower() for word in ["was", "were", "is", "are", "released", "directed", "stars"]):
            self.add_fact(user_input)

        self._refresh_vectorizer()

    def save_state(self):
@@ -282,27 +248,19 @@ def update_mood(self, mood_value):
        self.save_state()

# ============================================================== #
# INITIALIZE AGENT
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

# ============================================================== #
# HEADER WITH LOGO
# ============================================================== #
st.markdown(f"""
<div style="display:flex;align-items:center">
    <img class="header-logo" src="assets/logo.jpg">
    <h1 style="margin:0;color:#ffffff">Δ-Zero Chat – Adaptive AI</h1>
</div>
""", unsafe_allow_html=True)
st.markdown("<sub style='color:#c0c0c0'>by JCB – contextual and evolving</sub>", unsafe_allow_html=True)

# ============================================================== #
# SIDEBAR
# ============================================================== #
# Sidebar – Mood
st.sidebar.header("Mood Tracker")
mood = st.sidebar.slider("Your current mood", 0.0, 10.0, 5.0, 0.5)
if st.sidebar.button("Record Mood"):
@@ -311,31 +269,99 @@ def update_mood(self, mood_value):

if agent.mood_history:
    df_mood = pd.DataFrame(agent.mood_history)
    st.sidebar.line_chart(df_mood.set_index("timestamp")["mood"])
    fig = px.line(df_mood, x="timestamp", y="mood", title="Mood Over Time", markers=True)
    st.sidebar.plotly_chart(fig, width="stretch")

st.sidebar.info(f"Total chats: {len(agent.memory)}")
st.sidebar.info(f"Total chats: {len(agent.memory)}")  # full count
if agent.knowledge:
    st.sidebar.success(f"Knowledge base: {len(agent.knowledge)} entries")

# ============================================================== #
# CHAT INTERFACE
# ============================================================== #
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
    for msg in st.session_state.chat_history:
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["sender"] == "user":
            st.markdown(f"<div class='user-bubble'><b>You:</b> {msg['message']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background:#D1E7DD;padding:10px;border-radius:8px;text-align:right'>"
                        f"<b>You:</b> {msg['message']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'><b>Δ-Zero:</b> {msg['message']}</div>", unsafe_allow_html=True)
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
    response, slot = agent.respond(user_input, mood)
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
    knowledge = []
    if os.path.exists(KNOWLEDGE_DIR):
        for f in os.listdir(KNOWLEDGE_DIR):
            if f.endswith(".txt"):
                try:
                    with open(os.path.join(KNOWLEDGE_DIR, f), "r", encoding="utf-8") as file:
                        knowledge.extend([line.strip() for line in file if line.strip()])
                        for line in file:
                            text = line.strip()
                            if text:
                                knowledge.append(text)
                except Exception as e:
                    print(f"Error loading {f}: {e}")
    return knowledge
@@ -65,10 +67,15 @@ def __init__(self, n_slots=5, lr=0.07):
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
@@ -130,59 +137,84 @@ def choose_slot(self, mood=None):
        return slot

    def _refresh_vectorizer(self):
        recent_memory = self.memory[-MAX_MEMORY:]
        texts = self.knowledge + [
            str(m.get("input", "")) + " " + str(m.get("response", ""))
            for m in recent_memory if isinstance(m, dict)
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
        if text.strip() and text.strip() not in set(self.knowledge):
            with open(path, "a", encoding="utf-8") as f:
                f.write(text.strip() + "\n")
            self.refresh_knowledge()
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
                    response = self.knowledge[best_idx]
            except Exception:
                pass

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

        # Human-style chatter
        softeners = ["you know?", "if that makes sense.", "right?", "don’t you think?", "haha.", "that’s just my thought."]
        # --- Blend with human-style chatter ---
        softeners = [
            "you know?", "if that makes sense.", "right?", 
            "don’t you think?", "haha.", "that’s just my thought."
        ]
        if random.random() < 0.5:
            response += " " + random.choice(softeners)

        # Drop extra fact occasionally
        if self.knowledge and random.random() < 0.25:
            response += f" By the way, {random.choice(self.knowledge).lower()}"
        # --- Chance to drop a knowledge fun fact ---
        if self.knowledge and random.random() < 0.4:
            extra = random.choice(self.knowledge)
            # Avoid repeating the same fact in one response
            if extra not in response:
                response += f" By the way, {extra.lower()}"

        return response + f" [slot {slot}]"

    def respond(self, user_input, mood=None):
        slot = self.choose_slot(mood)
        return self.generate_response(user_input, slot, mood), slot
        response = self.generate_response(user_input, slot, mood)
        return response, slot

    def update(self, reward):
        if self.last_slot is not None:
@@ -198,6 +230,7 @@ def log_interaction(self, user_input, response, slot, reward=None, feedback=None
        df = pd.DataFrame(self.memory)
        self._save_encrypted_df(df)

        # Try to learn from factual-looking statements
        if any(word in user_input.lower() for word in ["was", "were", "is", "are", "released", "directed", "stars"]):
            self.add_fact(user_input)

@@ -218,22 +251,7 @@ def update_mood(self, mood_value):
# STREAMLIT UI
# ============================================================== #
st.set_page_config(page_title="Δ-Zero Chat", layout="wide")

# ----- Inject dark theme & colors -----
st.markdown(
    """
    <style>
    body {background-color:#1E1E2F; color:#FFFFFF;}
    .css-1d391kg {background-color:#2C2C3E; color:#FFFFFF;}
    div.stMarkdown div p {color:#FFFFFF;}
    </style>
    """, unsafe_allow_html=True
)

# ----- Logo & Title -----
if os.path.exists(LOGO_FILE):
    st.image(LOGO_FILE, width=80)
st.title("Δ-Zero Chat – Adaptive AI")
st.title("Δ-Zero Chat – Adaptive AI 🤖")
st.markdown("<sub>by JCB – contextual and evolving</sub>", unsafe_allow_html=True)

# Initialize agent
@@ -254,7 +272,7 @@ def update_mood(self, mood_value):
    fig = px.line(df_mood, x="timestamp", y="mood", title="Mood Over Time", markers=True)
    st.sidebar.plotly_chart(fig, width="stretch")

st.sidebar.info(f"Total chats: {len(agent.memory)}")
st.sidebar.info(f"Total chats: {len(agent.memory)}")  # full count
if agent.knowledge:
    st.sidebar.success(f"Knowledge base: {len(agent.knowledge)} entries")

@@ -297,6 +315,7 @@ def render_chat():
if user_input := st.chat_input("Talk to Δ-Zero..."):
    with st.spinner("Δ-Zero is thinking..."):
        lines = [line.strip() for line in user_input.split("\n") if line.strip()]
        # Add lines as knowledge quickly
        for i in range(0, len(lines), random.randint(1, 2)):
            chunk = "\n".join(lines[i:i+random.randint(1,2)])
            agent.add_fact(chunk)
@@ -310,16 +329,17 @@ def render_chat():
    st.rerun()

# ============================================================== #
# Δ-Zero AI-to-AI Bootstrapping
# Δ-Zero AI-to-AI Bootstrapping – Run once or periodically
# ============================================================== #
def bootstrap_ai(agent, n_rounds=20):
def bootstrap_ai(agent, n_rounds=5):
    if "bootstrapped" not in st.session_state:
        st.session_state.bootstrapped = True
    else:
        return
        return  # Already bootstrapped this session

    st.info("Initialising Δ-Zero AI-to-AI bootstrapping...")
    st.info("Initialising Δ-Zero bootstrapping...")

    # Step 1: Gather movie facts
    movie_facts = agent.knowledge.copy() if agent.knowledge else [
        "Star Wars is a space opera franchise.",
        "Inception was directed by Christopher Nolan.",
@@ -328,18 +348,21 @@ def bootstrap_ai(agent, n_rounds=20):
        "The Godfather is a classic crime movie."
    ]

    # Step 2: Social lures
    social_lures = [
        "have you seen it?", "what do you think?", "isn't it amazing?", 
        "right?", "don’t you think?", "it blew my mind!"
    ]

    for _ in range(n_rounds):
    # Step 3: Generate AI-to-AI conversations quickly
    for i in range(n_rounds):
        user_input = random.choice(movie_facts + social_lures)
        response, slot = agent.respond(user_input)
        agent.log_interaction(user_input, response, slot)
        agent.add_fact(user_input)

    agent.save_state()
    st.success("Δ-Zero is ready! Give feedback to help me learn.")
    st.success(f"Δ-Zero ready... start chatting and help me learn!")

bootstrap_ai(agent, n_rounds=5)

bootstrap_ai(agent, n_rounds=20)

