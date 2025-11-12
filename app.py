# app.py
# Δ-Zero Chat v3.2 – FIXED CSV CORRUPTION + DELTA ANALYSIS
# --------------------------------------------------------------

import streamlit as st
import numpy as np
import os
import pickle
import pandas as pd
import random
from datetime import datetime

# ==============================================================
# 1. DeltaAgent – Bulletproof CSV Logging
# ==============================================================
class DeltaAgent:
    def __init__(self, n_slots=5, lr=0.05, brain_file="global_brain.pkl", data_file="chat_log.csv"):
        self.n_slots = n_slots
        self.lr = lr
        self.brain_file = brain_file
        self.data_file = data_file
        self.self_slot = None
        self.prev_vec = None
        self.last_slot = None

        # Load brain
        if os.path.exists(brain_file):
            with open(brain_file, "rb") as f:
                saved = pickle.load(f)
                self.w = saved.get("w", np.ones(n_slots) / n_slots)
        else:
            self.w = np.ones(n_slots) / n_slots

        # Init log with
