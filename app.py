# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Contextual + Learning Conversational AI
# by JCB
# Improved UI + Fast Knowledge Learning
# --------------------------------------------------------------

import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import random
from datetime import datetime
from cryptography.fernet import Fernet
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
        ["I get that. But I'm still learning. Give me another fact?", "OK. Tell me more facts to help me understand.", "I hear you. But I need help understanding."],    # Calm
        ["Tell me more about that! I'm learning from you!", "Give me more information!", "I love where this is going. Keep talking."],                                   # Engaging
        ["That sounds emotional.", "How did that make you feel? And what are feelings?", "Interesting perspective. But explain more?"],                                # Empathetic
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
            w[3] *= 1.4  # Empat*
