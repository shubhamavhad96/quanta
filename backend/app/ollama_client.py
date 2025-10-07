import os
import requests

OLLAMA_URL = os.environ.get("QUANTA_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("QUANTA_OLLAMA_MODEL", "gemma3:4b")
TOKENS = int(os.environ.get("QUANTA_OLLAMA_TOKENS", "220"))
TEMP = float(os.environ.get("QUANTA_OLLAMA_TEMP", "0.05"))


def _generate(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": TOKENS, "temperature": TEMP},
        },
        timeout=120,
    )
    if r.status_code == 404:
        raise FileNotFoundError("generate endpoint not available")
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _chat(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": TOKENS, "temperature": TEMP},
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    # newer API returns {message: {content: ...}}; older may return response
    if isinstance(data, dict):
        msg = data.get("message") or {}
        content = msg.get("content")
        if content:
            return content.strip()
        if "response" in data:
            return str(data["response"]).strip()
    return ""


def ollama_generate(prompt: str) -> str:
    try:
        return _generate(prompt)
    except FileNotFoundError:
        # fallback to chat API
        return _chat(prompt)
    except requests.RequestException as e:
        # try chat as a fallback for other HTTP errors
        try:
            return _chat(prompt)
        except Exception as e2:
            raise RuntimeError(f"Ollama error: {e2}") from e2
