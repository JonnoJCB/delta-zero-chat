# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Contextual + Learning Conversational AI
# by JCB
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

# ============================================================== #
# CONFIG PATHS
# ============================================================== #
BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
BRAIN_FILE = os.path.join(BASE_DIR, "global_brain.pkl")
DATA_FILE = os.path.join(BASE_DIR, "chat_log.enc")
MOOD_FILE = os.path.join(BASE_DIR, "mood_history.pkl")
KEY_FILE = os.path.join(BASE_DIR, "secret.key")
MAX_MEMORY = 500  # keep only 500 recent chats for recall

# ============================================================== #
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
