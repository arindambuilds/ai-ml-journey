# streaming_call.py — AI responses appear word by word

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

print("AI is thinking...\n")

# stream=True makes response come word by word
response = model.generate_content(
    "Explain what RAG means in AI in simple terms",
    stream=True
)

for chunk in response:
    print(chunk.text, end="", flush=True)

print("\n\nDone!")