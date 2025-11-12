# --------------------------------------------------------------
#  INPUT + SEND (works with Enter AND with a Send button)
# --------------------------------------------------------------
with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([5, 1])               # 5 parts text, 1 part button
    with cols[0]:
        user_input = st.text_input(
            "Type your message...",
            placeholder="Ask Δ-Zero anything…",
            label_visibility="collapsed"   # hide the label (we already have a placeholder)
        )
    with cols[1]:
        submit_btn = st.form_submit_button("Send")

# ----- Process the message (runs when Enter OR Send is pressed) -----
if submit_btn and user_input.strip():
    response, slot = agent.respond(user_input)
    agent.log_interaction("user", user_input, response, slot)
    agent.save_state()

    # add to the visual chat
    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    st.session_state.chat_history.append({"sender": "bot",  "message": response})
    st.session_state.last_bot_idx = len(st.session_state.chat_history) - 1

    st.rerun()          # refresh the page so the new messages appear instantly
