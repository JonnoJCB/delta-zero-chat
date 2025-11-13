# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Adaptive AI with Feedback, Mood Chart & Knowledge
# by JCB – fixed responsive contextual version
# --------------------------------------------------------------

import streamlit as st
import numpy as np
import os
import pickle
import pandas as pd
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
MOOD_FILE = os.path.join(BASE_DIR, "mood_history.pkl")
KEY_FILE = os.path.join(BASE_DIR, "secret.key")

# ============================================================== #
# LOAD KNOWLEDGE
# ============================================================== #
def load_knowledge():
    knowledge = []
    if os.path.exists(KNOWLEDGE_DIR):
        for f in os.listdir(KNOWLEDGE_DIR):
            if f.endswith(".txt"):
                path = os.path.join(KNOWLEDGE_DIR, f)
                with open(path, "r", encoding="utf-8") as file:
                    for line in file:
                        line = line.strip()
                        if line:
                            knowledge.append(line)
    return knowledge

# ============================================================== #
# Δ-Zero AGENT
# ============================================================== #
class DeltaAgent:
    REPLIES = [
        ["Wow, fascinating!", "I'm intrigued!", "That's wild!"],          # Curious
        ["I understand.", "That makes sense.", "Clear as day."],          # Calm
        ["Tell me more!", "Keep going!", "Don't stop now!"],              # Engaging
        ["How do you feel about that?", "Why do you think so?", "Deep."], # Empathetic
        ["Let's analyze this.", "Interesting angle.", "Break it down."],  # Analytical
    ]

    def __init__(self, n_slots=5, lr=0.07, context_size=5):
        self.n_slots = n_slots
        self.lr = lr
        self.context_size = context_size
        self.knowledge = load_knowledge()
        self.memory = []
        self.mood_history = []
        self.context = []
        self.last_slot = None

        # Encryption
        self.cipher = self._load_or_create_key()

        # Load persistent state
        self.w = self._load_brain()
        self.memory = self._load_encrypted_log()
        self.mood_history = self._load_mood()

        # Vectorizer for contextual retrieval
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._fit_vectorizer()

    # ------------------- internal ------------------- #
    def _fit_vectorizer(self):
        if self.knowledge:
            self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge)
        else:
            self.knowledge_vectors = None

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
                encrypted = f.read()
                if not encrypted:
                    return []
                decrypted = self.cipher.decrypt(encrypted)
                df = pd.read_csv(pd.io.common.StringIO(decrypted.decode()))
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
        encrypted = self.cipher.encrypt(csv.encode())
        with open(DATA_FILE, "wb") as f:
            f.write(encrypted)

    # ------------------- mood weighting ------------------- #
    def _apply_mood_boost(self, mood):
        w = self.w.copy()
        if mood <= 3:
            w[3] *= 1.4  # empathetic
        elif mood >= 7:
            w[0] *= 1.3  # curious
            w[2] *= 1.3  # engaging
        return w / w.sum()

    def choose_slot(self, mood=None):
        probs = self._apply_mood_boost(mood) if mood is not None else self.w
        slot = np.random.choice(self.n_slots, p=probs)
        self.last_slot = slot
        return slot

    # ------------------- response logic ------------------- #
    def generate_response(self, user_input, slot, mood=None):
        if not user_input:
            return random.choice(self.REPLIES[slot])

        # Try contextual match from movie.txt
        if self.knowledge and self.knowledge_vectors is not None:
            try:
                query_vec = self.vectorizer.transform([user_input])
                sims = cosine_similarity(query_vec, self.knowledge_vectors)[0]
                best_idx = int(np.argmax(sims))
                if sims[best_idx] > 0.1:
                    base = self.knowledge[best_idx]
                    return base + f" [slot {slot}]"
            except Exce
