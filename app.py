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
    """Load text lines from all files in /knowledge."""
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
