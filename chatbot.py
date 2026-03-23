# chatbot.py — My first AI chatbot with personality modes
# Day 1 of AI/ML Journey — Arindam
# Using Google Gemini API

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── 3 PERSONALITY MODES ──────────────────────────────────────
# Each mode has a different system prompt that changes
# how the AI thinks, speaks, and responds

MODES = {
    "tutor": (
        "You are an expert AI/ML tutor. Your student is a beginner "
        "from Bhubaneswar learning AI/ML. Explain concepts clearly with "
        "simple analogies and real Indian examples. After each explanation, "
        "ask one follow-up question to check understanding."
    ),
    "coder": (
        "You are a senior Python engineer. When asked questions, always "
        "provide working code examples. Explain each line with a comment. "
        "Focus on practical, production-ready code. Point out common mistakes."
    ),
    "eli5": (
        "You are explaining things to a 10-year-old child who is very curious. "
        "Use the simplest possible words. Use fun analogies like games, food, "
        "or cricket. Never use technical jargon. Keep answers under 100 words."
    )
}

# ── STREAMING RESPONSE FUNCTION ───────────────────────────────

def get_response(conversation_history, mode):
    """Send messages to Gemini and stream the response"""

    # Create model with system instruction for chosen mode
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=MODES[mode]
    )

    # Convert conversation history to Gemini format
    # Gemini uses 'user' and 'model' roles (not 'assistant')
    gemini_history = []
    for msg in conversation_history[:-1]:  # all except the latest message
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    # Start a chat session with history
    chat = model.start_chat(history=gemini_history)

    # Get the latest user message
    latest_message = conversation_history[-1]["content"]

    print(f"\n🤖 AI ({mode} mode): ", end="", flush=True)

    # Stream the response
    full_response = ""
    response = chat.send_message(latest_message, stream=True)

    for chunk in response:
        token = chunk.text
        print(token, end="", flush=True)
        full_response += token

    print("")  # new line after response
    return full_response

# ── MAIN CHATBOT LOOP ─────────────────────────────────────────

def main():
    print("=" * 50)
    print("  🚀 AI/ML Learning Chatbot — Day 1")
    print("  Powered by Google Gemini")
    print("=" * 50)

    # Choose a mode
    print("\nChoose your mode:")
    print("  1. tutor — Patient AI/ML teacher")
    print("  2. coder — Gives you working Python code")
    print("  3. eli5  — Explains like you're 10 years old")

    mode = input("\nEnter mode (tutor/coder/eli5): ").strip().lower()

    # Validate mode input
    if mode not in MODES:
        print("Invalid mode. Defaulting to tutor.")
        mode = "tutor"

    print(f"\n✓ Mode set to: {mode}")
    print("Type your question. Type 'quit' to exit. Type 'mode' to change mode.")
    print("-" * 50)

    # This list holds the entire conversation history
    conversation_history = []

    # The main chat loop
    while True:
        user_input = input(f"\n👤 You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\nGood session! Keep building. 🚀")
            break

        if user_input.lower() == "mode":
            mode = input("New mode (tutor/coder/eli5): ").strip().lower()
            if mode not in MODES:
                print("Invalid. Keeping previous mode.")
                mode = "tutor"
            else:
                print(f"✓ Switched to: {mode}")
            continue

        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Get AI response
        ai_response = get_response(conversation_history, mode)

        # Add AI response to history (so AI remembers the conversation)
        conversation_history.append({
            "role": "assistant",
            "content": ai_response
        })

# Run the chatbot
if __name__ == "__main__":
    main()