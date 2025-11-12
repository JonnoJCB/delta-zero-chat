# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Sassy Terminator AI (No Blank Screen!)
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
# 1. Load optional knowledge from /knowledge/*.txt
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
# 2. DeltaAgent – Sassy & Adaptive
# ==============================================================
class DeltaAgent:
    def __init__(self):
        self.n_slots = 5
        self.lr = 0.07
        self.brain_file = "brain.pkl"
        self.chat_file = "chat.enc"
        self.mood_file = "mood.pkl"
        self.key_file = "key.key"
        self.knowledge = load_knowledge()

        self.memory = []
        self.mood_history = []
        self.last_slot = None

        # Encryption
        try:
            if not os.path.exists(self.key_file):
                key = Fernet.generate_key()
                with open(self.key_file, "wb") as f:
                    f.write(key)
            with open(self.key_file, "rb")
