# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Adaptive AI with Feedback, Knowledge & Learning Stats
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

# ==============================================================
# 1. Helper: Load knowledge from /knowledge/*.txt
# ==============================================================
def load_knowledge():
    knowledge = []
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
    if os.path.exists(knowledge_dir):
        for filename in os.listdir(knowledge_dir):
            if filename.endswith(".txt"):
                path = os.path.join(knowledge_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    knowledge.extend([line.strip() for line in f if line.strip()])
    return knowledge


# ==============================================================
# 2. DeltaAgent – Adaptive with per-slot replies & knowledge
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
        self.knowledge = load_knowledge()          # <-- optional facts

       
