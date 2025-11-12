# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Sassy Terminator AI with Hidden Reset
# --------------------------------------------------------------

import streamlit as st
import numpy as np
import os
import pickle
import pandas as pd
import random
from datetime import datetime
from cryptography.fernet import Fernet

# ==============================================================
# 1. Load knowledge from /knowledge/*.txt
# ==============================================================
def load_knowledge():
    knowledge = []
    try:
        knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
        if os.path.exists(knowledge_dir):
            for filename in os.listdir(knowledge_dir):
                if filename.endswith(".txt"):
                    path = os.path.join(knowledge_dir, filename)
                    with open(path, "r", encoding="utf-8") as f:
                        knowledge.extend(line.strip() for line in f if line.strip())
    except Exception:
        pass
    return knowledge

# ==============================================================
# 2. DeltaAgent – Sassy Terminator Replies
# ==============================================================
class DeltaAgent:
    def __init__(
        self,
        n_slots=5,
        lr=0.07,
        brain_file="global_brain.pkl",
        data_file="chat_log.enc",
        mood_file="mood_history.pkl",
        key_file="secret.key",
    ):
        self.n_slots = n_slots
        self.lr = lr
        self.brain_file = brain_file
        self.data_file = data_file
        self.mood_file = mood_file
        self.key_file = key_file
        self.knowledge = load_knowledge()

        self.memory = []
        self.mood_history = []
        self.last_slot = None

        # Encryption
        try:
            if not os.path.exists(key_file):
                key = Fernet.generate_key()
                with open(key_file, "wb") as f:
                    f.write(key)
            with open(key_file, "rb") as f:
                self.cipher = Fernet(f.read())
        except Exception:
            self.cipher = None

        # Load brain
        try:
            if os.path.exists(brain_file):
                with open(brain_file, "rb") as f:
                    saved = pickle.load(f)
                    self.w = saved.get("w", np.ones(n_slots) / n_slots)
            else:
                self.w = np.ones(n_slots) / n_slots
        except Exception:
            self.w = np.ones(n_slots) / n_slots

        # Load encrypted chat
        if self.cipher and os.path.exists(data_file):
            try:
                with open(data_file, "rb") as f:
                    encrypted = f.read()
                    if encrypted:
                        decrypted = self.cipher.decrypt(encrypted)
                        df = pd.read_csv(pd.io.common.StringIO(decrypted.decode()))
                        self.memory = df.to_dict("records")
            except Exception:
                self.memory = []
        else:
            self._save_encrypted_df(pd.DataFrame(columns=[
                "timestamp","user","input","response","slot",
                "reward","feedback","fb_text"
            ]))

        # Load mood
        try:
            if os.path.exists(mood_file):
                with open(mood_file, "rb") as f:
                    self.mood_history = pickle.load(f)
        except Exception:
            self.mood_history = []

    def choose_slot(self):
        probs
