import os
import requests
import json
from dotenv import load_dotenv
import textwrap

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def ask_openrouter(prompt: str, width: int = 80) -> str:
    """Query OpenRouter API and return a clean, wrapped response regardless of JSON structure."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "LangGraph Chatbot Playground",
    }

    data = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json()

    # Debug
    # print(json.dumps(result, indent=2))

    if "choices" in result and len(result["choices"]) > 0:
        content = result["choices"][0]["message"]["content"]
    elif "completion" in result:
        content = result["completion"]
    elif "response" in result:
        content = result["response"]
    else:
        # fallback
        content = str(result)

    clean_content = textwrap.fill(content.strip(), width=width)

    return clean_content
print(ask_openrouter(str(input("What Is Your Question: "))))