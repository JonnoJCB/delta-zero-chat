# app.py
# --------------------------------------------------------------
# Δ-Zero Chat – Contextual + Learning Conversational AI
# Full Streamlit app
# --------------------------------------------------------------

import streamlit as st
import os
import pandas as pd
from datetime import datetime
from delta_agent import DeltaAgent  # assumes the class from previous code is saved in delta_agent.py

# ============================================================== #
# APP CONFIG
# ============================================================== #
st.set_page_config(page_title="Δ-Zero Chat", page_icon="🤖", layout="wide")

# ============================================================== #
# SESSION STATE INIT
# ============================================================== #
if "agent" not in st.session_state:
    st.session_state.agent = DeltaAgent()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================== #
# HEADER
# ============================================================== #
st.title("Δ-Zero Chat 🤖")
st.markdown(
    """
    Δ-Zero Chat is an adaptive AI chatbot that learns from your inputs.  
    It maintains memory of the last 500 chats while tracking total conversation count.  
    Mood-aware and contextually intelligent. 
    """
)

# ============================================================== #
# SIDEBAR - Mood
# ============================================================== #
st.sidebar.header("Set Mood")
mood = st.sidebar.slider("Your current mood (1 = sad, 10 = happy)", min_value=1, max_value=10, value=5)

# ============================================================== #
# CHAT INTERFACE
# ============================================================== #
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input.strip():
    agent = st.session_state.agent

    # Generate response
    response, slot = agent.respond(user_input, mood)
    agent.log_interaction(user_input, response, slot)
    agent.update(user_input=user_input, response=response)
    agent.update_mood(mood)
    agent.save_state()

    # Append to chat history
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    st.session_state.chat_history.append({"role": "agent", "text": response})

# ============================================================== #
# DISPLAY CHAT HISTORY
# ============================================================== #
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['text']}")
    else:
        st.markdown(f"**Δ-Zero:** {chat['text']}")

# ============================================================== #
# OPTIONAL: Total Chats & Agent Stats
# ============================================================== #
st.sidebar.markdown("---")
st.sidebar.subheader("Agent Stats")
st.sidebar.text(f"Total chats recorded: {st.session_state.agent.total_chats}")
st.sidebar.text(f"Memory retained: {len(st.session_state.agent.memory)}")
st.sidebar.text(f"Slot probabilities: {st.session_state.agent.w.round(2)}")
