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
                path =
