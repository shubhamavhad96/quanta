import os
import requests
OLLAMA_URL = os.environ.get("QUANTA_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("QUANTA_OLLAMA_MODEL", "llama3.1:8b")
MAX_TOKENS = int(os.environ.get("QUANTA_OLLAMA_TOKENS", "220"))
TEMPERATURE = float(os.environ.get("QUANTA_OLLAMA_TEMP", "0.1"))


def generate(prompt: str) -> str:
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}
    }, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()
