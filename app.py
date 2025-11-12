# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Adaptive AI with Feedback, Knowledge & Stats
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
# 1. Load knowledge from /knowledge/*.txt
# ==============================================================
def load_knowledge():
    knowledge = []
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
    if os.path.exists(knowledge_dir):
        for filename in os.listdir(knowledge_dir):
            if filename.endswith(".txt"):
                path = os.path.join(knowledge_dir, filename)   # <-- Fixed: moved outside 'with'
                with open(path, "r", encoding="utf-8") as f:
                    knowledge.extend([line.strip() for line in f if line.strip()])
    return knowledge

# ==============================================================
# 2. DeltaAgent – Adaptive with per-slot replies
# ==============================================================
class DeltaAgent:
    def __init__(
        self,
        n_slots=5,
        lr
