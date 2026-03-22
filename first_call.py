import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Get the key
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# Use the model
model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content(
    "I am a 28 year old beginner from Bhubaneswar "
    "starting my AI/ML journey today. "
    "Give me one powerful motivational message "
    "and 3 specific things I should focus on first."
)

print("=" * 50)
print("MESSAGE FROM YOUR FIRST AI CALL:")
print("=" * 50)
print(response.text)