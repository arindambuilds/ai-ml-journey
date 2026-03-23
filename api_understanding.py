# api_understanding.py
# Day 1 Consolidation - Understanding API calls deeply

# STEP 1: Import the library
# In my own words: This opens the Google AI Toolbox so python can use it
import google.generativeai as genai
from dotenv import load_dotenv
import os

# STEP 2: Load API Key
# In my own words: This reads my secret key from the .env file into memory
load_dotenv()

# STEP 3: Show our identity to Google
# In my own words: This tells Google who I am using my API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# STEP 4: Choose which AI brain to use
# In my own words: This picks which Gemini model I want to talk to
model= genai.GenerativeModel("gemini-2.0-flash")

# STEP 5: The actual API call - message travels to Google and back
# In my own words: This sends my message to Google and waits for the answer
response = model.generate_content("Explain what an API call is in 2 sentences")

# STEP 6: Extract just the text from the response
# In my own words: The response has many parts - .text gets only the answer
print(response.text)

# STEP 7: Check what else is inside the response object
print("\n---Other things inside the response ---")
print(type(response))
print(dir(respionse)) 

# This shows ALL atributes available