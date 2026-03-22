# app.py — Web chatbot using Gradio + Gemini
# Day 1 of my AI/ML Journey

from google import genai
from google.genai import types
from dotenv import load_dotenv
import gradio as gr
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODES = {
    "Tutor Mode": (
        "You are an expert AI/ML tutor for a beginner "
        "from Bhubaneswar. Explain concepts clearly "
        "with simple analogies and Indian examples."
    ),
    "Coder Mode": (
        "You are a senior Python engineer. Always "
        "provide working code examples with clear comments."
    ),
    "ELI5 Mode": (
        "Explain to a 10-year-old child using fun "
        "analogies like games, food, or cricket. "
        "Keep it under 80 words."
    )
}

def chat(message, history, mode):
    full_prompt = MODES[mode] + "\n\nUser: " + message
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=full_prompt
    )
    return response.text

with gr.Blocks(title="AI Learning Chatbot") as demo:
    gr.Markdown(
        "# AI/ML Learning Chatbot 🚀\n"
        "**Built on Day 1 — Bhubaneswar to Bangalore**"
    )
    mode_dropdown = gr.Dropdown(
        choices=["Tutor Mode", "Coder Mode", "ELI5 Mode"],
        value="Tutor Mode",
        label="Choose AI Personality Mode"
    )
    gr.ChatInterface(
        fn=chat,
        additional_inputs=[mode_dropdown]
    )

if __name__ == "__main__":
    demo.launch()