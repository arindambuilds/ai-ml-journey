# app.py — Web version of the AI chatbot using Gradio + Gemini
# Day 1 of AI/ML Journey — Arindam

import google.generativeai as genai
import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── 3 PERSONALITY MODES ──────────────────────────────────────

MODES = {
    "Tutor Mode": (
        "You are an expert AI/ML tutor for a beginner from Bhubaneswar. "
        "Explain concepts clearly with simple analogies and real Indian examples."
    ),
    "Coder Mode": (
        "You are a senior Python engineer. Always provide working code examples "
        "with clear comments. Point out common mistakes."
    ),
    "ELI5 Mode": (
        "Explain to a 10-year-old child using fun analogies like games, food, "
        "or cricket. Under 80 words."
    )
}

# ── CHAT FUNCTION ─────────────────────────────────────────────

def chat(message, history, mode):
    """Called every time user sends a message. Streams response to browser."""

    # Create Gemini model with the selected mode's system instruction
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=MODES[mode]
    )

    # Convert Gradio history format to Gemini format
    # Gradio gives: list of [user_msg, assistant_msg] pairs
    # Gemini needs: list of {role, parts} dicts
    gemini_history = []
    for human, assistant in history:
        gemini_history.append({"role": "user", "parts": [human]})
        gemini_history.append({"role": "model", "parts": [assistant]})

    # Start chat with history
    chat_session = model.start_chat(history=gemini_history)

    # Stream the response
    response = chat_session.send_message(message, stream=True)

    partial = ""
    for chunk in response:
        partial += chunk.text
        yield partial  # yield streams text to browser in real time

# ── BUILD THE GRADIO INTERFACE ────────────────────────────────

with gr.Blocks(title="AI Learning Chatbot — Gemini") as demo:
    gr.Markdown("# 🚀 AI/ML Learning Chatbot\nBuilt on Day 1 of my AI journey · Powered by Google Gemini")

    mode_dropdown = gr.Dropdown(
        choices=["Tutor Mode", "Coder Mode", "ELI5 Mode"],
        value="Tutor Mode",
        label="Choose AI Personality Mode"
    )

    gr.ChatInterface(
        fn=chat,
        additional_inputs=[mode_dropdown],
        title="",
    )
if __name__ == "__main__":
    demo.launch()