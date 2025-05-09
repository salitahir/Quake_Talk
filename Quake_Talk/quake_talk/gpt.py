# quake_talk/gpt.py
import os
from dotenv import load_dotenv

load_dotenv()   # <— this will read .env file into os.environ

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask(
    question: str,
    context: str,
    temperature: float | None = None,
    max_tokens: int | None = None
) -> str:
    """
    Query GPT-4o using only the provided context.
    """
    system_prompt = (
        "You are a research assistant analyzing tweets about the 2023 Turkey-Syria earthquake. "
        "Your response must be rooted in the provided context. "
        "If there is not enough information in the context to answer the question, say so. "
        "If you add any external knowledge, preface it with 'Beyond the provided data…'."
    )
    return _call_gpt(
        system=system_prompt,
        user=f"Context:\n{context}\n\nQuestion:\n{question}",
        temperature=temperature or float(os.getenv("GPT_TEMPERATURE", 0.4)),
        max_tokens=max_tokens or int(os.getenv("GPT_MAX_TOKENS", 1000)),
    )

def summarize_with_gpt4o(text_chunk: str) -> str:
    """
    Summarize the key concerns, themes, and sentiments in the tweets.
    Uses its own system prompt, higher creativity, and larger token budget.
    """
    system_prompt = (
        "You are a helpful assistant summarizing tweets related to the Turkey-Syria earthquake."
    )
    user_prompt = (
        f"{text_chunk}\n\n"
        "Please summarize the key concerns, themes, and sentiments expressed in these tweets."
    )
    return _call_gpt(
        system=system_prompt,
        user=user_prompt,
        temperature=float(os.getenv("SUMMARY_TEMPERATURE", 0.7)),
        max_tokens=int(os.getenv("SUMMARY_MAX_TOKENS", 4000)),
    )

def _call_gpt(system: str, user: str, temperature: float, max_tokens: int) -> str:
    try:
        resp = client.chat.completions.create(
            model=os.getenv("GPT_MODEL", "gpt-4o"),
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"❗ GPT API error: {e}"