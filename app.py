def generate_response(self, user_input, slot, mood=None):
    """Generate contextual conversational response with social lures."""
    response = ""

    # --- Contextual pull from movie knowledge ---
    if self.knowledge and self.knowledge_matrix is not None:
        try:
            query_vec = self.vectorizer.transform([user_input])
            sims = cosine_similarity(query_vec, self.knowledge_matrix).flatten()
            best_idx = sims.argmax()
            if sims[best_idx] > 0.15:
                fact = self.knowledge[best_idx]
                base = random.choice([
                    f"That reminds me of something: {fact}",
                    f"I recall: {fact}",
                    f"Funny you mention that — {fact}",
                    f"From what I remember: {fact}",
                ])
                response = base
        except Exception as e:
            print("TF-IDF error:", e)

    # --- Fallback if no good match ---
    if not response:
        response = random.choice(self.REPLIES[slot])

    # --- Blend with human-style chatter ---
    softeners = [
        "you know?", "if that makes sense.", "right?", 
        "don’t you think?", "haha.", "that’s just my thought."
    ]
    if random.random() < 0.5:
        response += " " + random.choice(softeners)

    # --- Social lures / curiosity hooks ---
    social_lures = [
        "Have you ever thought about that?", 
        "I wonder what you think.", 
        "Isn’t that interesting?", 
        "What would you do in that situation?", 
        "I’ve always found that curious.", 
        "Makes you think, doesn’t it?"
    ]
    if random.random() < 0.3:
        response += " " + random.choice(social_lures)

    # --- Chance to drop a movie fun fact ---
    if self.knowledge and random.random() < 0.25:
        extra = random.choice(self.knowledge)
        response += f" By the way, {extra.lower()}"

    return response + f" [slot {slot}]"

