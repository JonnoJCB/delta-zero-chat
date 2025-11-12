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
                    knowledge.extend([line.strip() for line in
