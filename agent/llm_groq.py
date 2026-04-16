import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()  # load .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

URL = "https://api.groq.com/openai/v1/chat/completions"

def call_llm(messages):
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY not found. Check .env")
    
    # 🔥 WAIT BEFORE EVERY CALL
    print("⏳ Waiting 30 seconds before API call...")
    time.sleep(30)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.2
    }

    response = requests.post(URL, headers=headers, json=payload)

    if response.status_code != 200:
        print("ERROR:", response.text)
        raise Exception(response.text)

    return response.json()["choices"][0]["message"]["content"]