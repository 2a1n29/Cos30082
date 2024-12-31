# This is my personal API key
# AIzaSyAjslHKjjvqmxgvtdrxMecDmVvu7EsgbwU
import google.generativeai as genai
import os
os.environ["API_KEY"] = "AIzaSyAjslHKjjvqmxgvtdrxMecDmVvu7EsgbwU"
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")
while True:
    user_input = input("Enter a text: ")
    response = model.generate_content(user_input)
    print(response.text)