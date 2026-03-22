# chatbot.py — AI chatbot with 3 personality modes
# Day 1 of my AI/ML Journey

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 3 PERSONALITY MODES
MODES = {
    "tutor": (
        "You are an expert AI/ML tutor. Your student is a "
        "beginner from Bhubaneswar learning AI/ML. Explain "
        "concepts clearly with simple analogies and real "
        "Indian examples. After each explanation, ask one "
        "follow-up question to check understanding."
    ),
    "coder": (
        "You are a senior Python engineer. When asked "
        "questions, always provide working code examples. "
        "Explain each line with a comment. Focus on "
        "practical, production-ready code."
    ),
    "eli5": (
        "You are explaining things to a 10-year-old child "
        "who is very curious. Use the simplest possible "
        "words. Use fun analogies like games, food, or "
        "cricket. Never use technical jargon. "
        "Keep answers under 100 words."
    )
}

def get_response(user_input, mode):
    prompt = MODES[mode] + "\n\nUser: " + user_input
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        prompt,
        stream=True
    )
    print(f"\nAI ({mode} mode): ", end="")
    for chunk in response:
        print(chunk.text, end="", flush=True)
    print("")

def main():
    print("=" * 50)
    print("AI/ML Learning Chatbot — Day 1")
    print("=" * 50)
    print("\nChoose your mode:")
    print("  1. tutor — Patient AI/ML teacher")
    print("  2. coder — Gives you working Python code")
    print("  3. eli5  — Explains like you're 10 years old")

    mode = input("\nEnter mode (tutor/coder/eli5): ").strip().lower()

    if mode not in MODES:
        print("Invalid mode. Defaulting to tutor.")
        mode = "tutor"

    print(f"\nMode set to: {mode}")
    print("Type your question. Type 'quit' to exit.")
    print("Type 'mode' to change mode.")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\nGood session! Keep building.")
            break

        if user_input.lower() == "mode":
            mode = input(
                "New mode (tutor/coder/eli5): "
            ).strip().lower()
            if mode not in MODES:
                mode = "tutor"
            print(f"Mode changed to: {mode}")
            continue

        get_response(user_input, mode)

if __name__ == "__main__":
    main()