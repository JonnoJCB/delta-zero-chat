# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Smart AI with MOOD CHART, Encrypted Memory & MSN Ping
# by JCB
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
import base64                                 # <-- NEW import for autoplay

# ==============================================================
# 1. DeltaAgent – Smart + Mood Tracking + Encrypted Learning
# ==============================================================
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

        self.memory = []
        self.mood_history = []
        self.prev_vec = None
        self.last_slot = None

        # Encryption setup
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

        # Load encrypted chat memory
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "rb") as f:
                    encrypted_data = f.read()
                    if encrypted_data:
                        decrypted = self.cipher.decrypt(encrypted_data)
                        df = pd.read_csv(pd.io.common.StringIO(decrypted.decode()))
                        self.memory = df.to_dict(orient="records")
                    else:
                        self.memory = []
            except Exception:
                st.warning("Warning: Could not decrypt chat log. Starting fresh.")
                self.memory = []
        else:
            self._save_encrypted_df(pd.DataFrame(columns=["timestamp","user","input","response","slot","reward","feedback","fb_text"]))

        # Load mood history
        if os.path.exists(mood_file):
            with open(mood_file, "rb") as f:
                self.mood_history = pickle.load(f)
        else:
            self.mood_history = []

    def embed(self, text):
        vec = np.zeros(26)
        for c in text.lower():
            if c.isalpha():
                vec[ord(c)-97] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm>0 else vec

    def choose_slot(self):
        probs = self.w / self.w.sum()
        slot = np.random.choice(range(self.n_slots), p=probs)
        self.last_slot = slot
        return slot

    def generate_response(self, user_input, slot):
        replies = [
            "That's quite interesting!",
            "I see what you mean.",
            "Tell me more about that.",
            "How does that make you feel?",
            "Let's explore that thought."
        ]
        return random.choice(replies) + f" [slot {slot}]"

    def respond(self, user_input):
        vec = self
