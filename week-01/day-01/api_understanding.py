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

# numpy_basics.py - Day 2 of AI\ML Journey
# Topic: Numpy Fundamentalsfor ML

import numpy as np

# 1.Creating arrays
scores = np.array[85, 91, 65, 29, 89, 75, 63, 74]
print("Scores array:", scores)
print("Shape", scores.shape)
print("Data type", scores.dtype)
print("Number of Dimensions", scores.ndim)


# 2. Basic operations — these work element-by-element
print("\n--- Array Operations ---")
print("Mean score:", np.mean(scores))
print("Max score:", np.max(scores))
print("Min score:", np.min(scores))
print("Standard deviation:", np.std(scores))
print("Scores above 85:", scores[scores > 85])

# 3. 2D arrays — this is what your data looks like in ML
student_data = np.array([
    [85, 92, 78],   # student 1: math, science, english
    [95, 88, 91],   # student 2
    [76, 83, 70],   # student 3
    [91, 95, 88]    # student 4
])
print("\n--- 2D Array ---")
print("Shape:", student_data.shape)         # (4, 3) — 4 students, 3 subjects
print("All math scores:", student_data[:, 0])     # first column
print("Student 1 scores:", student_data[0, :])    # first row
print("Average per student:", np.mean(student_data, axis=1))
print("Average per subject:", np.mean(student_data, axis=0))
