# quake_talk/gpt.py

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Initialize the OpenAI client once
CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask(question: str, context: str) -> str:
    """
    Sends the concatenated context + question to GPT and returns the response.
    """
    resp = CLIENT.chat.completions.create(
        model=os.getenv("GPT_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
        ],
        temperature=float(os.getenv("GPT_TEMPERATURE", 0.4)),
        max_tokens=int(os.getenv("GPT_MAX_TOKENS", 800))
    )
    return resp.choices[0].message.content.strip()
