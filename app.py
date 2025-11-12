# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Adaptive AI with Feedback, Mood Chart & MSN Ping
# by JCB (enhanced)
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
import time
import base64
import streamlit.components.v1 as components

# ==============================================================
# 1. DeltaAgent – Adaptive with Feedback + Per-Slot Responses
# ==============================================================

# ---- Optional: Load knowledge from .txt files in /knowledge/ ----
def load_knowledge():
    knowledge = []
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
    if os.path.exists(knowledge_dir):
        for file in os.listdir(knowledge_dir):
            if file.endswith(".txt"):
                with open(os.path.join(knowledge_dir, file), "r", encoding="utf-8") as f:
                    knowledge.extend([line.strip() for line in f if line.strip()])
    return knowledge

class DeltaAgent:
    def __init__(
        self,
        n_slots=5,
        lr=0.07,
        brain_file="global_brain.pkl",
        data_file="chat_log.enc",
        mood_file="mood_history.pkl",
        key_file="secret.key"
    ):
        self.n_slots = n_slots
        self.lr = lr
        self.brain_file = brain_file
        self.data_file = data_file
        self.mood_file = mood_file
        self.key_file = key_file
        self.knowledge = load_knowledge()  # <-- NEW: optional facts

        self.memory = []
        self.mood_history = []
        self.last_slot = None

        # Encryption
        if not os.path.exists(self.key_file):
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(key)
        with open(self.key_file, "rb") as f:
            self.cipher = Fernet(f.read())

        # Load brain
        if os.path.exists(brain_file):
            with open(brain_file, "rb") as f:
                saved = pickle.load(f)
                self.w = saved.get("w", np.ones(n_slots) / n_slots)
        else:
            self.w = np.ones(n_slots) / n_slots

        # Load encrypted chat
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "rb") as f:
                    encrypted = f.read()
                    if encrypted:
                        decrypted = self.cipher.decrypt(encrypted)
                        df = pd.read_csv(pd.io.common.StringIO(decrypted.decode()))
                        self.memory = df.to_dict("records")
                    else:
                        self.memory = []
            except Exception:
                st.warning("Warning: Could not decrypt chat log. Starting fresh.")
                self.memory = []
        else:
            self._save_encrypted_df(pd.DataFrame(columns=[
                "timestamp","user","input","response","slot",
                "reward","feedback","fb_text"
            ]))

        # Load mood
        if os.path.exists(mood_file):
            with open(mood_file, "rb") as f:
                self.mood_history = pickle.load(f)
        else:
            self.mood_history = []

    def embed(self, text):
        vec = np.zeros(26)
        for c in text.lower():
            if c.isalpha():
                vec[ord(c) - 97] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def choose_slot(self):
        probs = self.w / self.w.sum()
        slot = np.random.choice(range(self.n_slots), p=probs)
        self.last_slot = slot
        return slot

    # ---- Per-slot response styles ----
    REPLIES = [
        ["Wow, fascinating!", "I'm intrigued!", "That's wild!"],                    #
