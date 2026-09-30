from typing import List, Dict, Optional

import requests

from config import (
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    OLLAMA_TIMEOUT,
)


def check_ollama_status() -> Dict[str, object]:
    """Check whether Ollama is running and both required models exist."""
    try:
        response = requests.get(
            f"{OLLAMA_API_BASE}/tags",
            timeout=10,
        )
        response.raise_for_status()

        payload = response.json()
        model_names = {
            item.get("name", "")
            for item in payload.get("models", [])
        }

        generator_ready = any(
            name == OLLAMA_MODEL
            or name.startswith(f"{OLLAMA_MODEL}:")
            for name in model_names
        )

        embedding_ready = any(
            name == EMBEDDING_MODEL
            or name.startswith(f"{EMBEDDING_MODEL}:")
            for name in model_names
        )

        if not generator_ready and not embedding_ready:
            message = (
                "Ollama is running, but both models are missing. "
                f"Run: ollama pull {OLLAMA_MODEL} "
                f"and ollama pull {EMBEDDING_MODEL}"
            )
        elif not generator_ready:
            message = (
                f"Ollama is running, but {OLLAMA_MODEL} is missing. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            )
        elif not embedding_ready:
            message = (
                f"Ollama is running, but {EMBEDDING_MODEL} is missing. "
                f"Run: ollama pull {EMBEDDING_MODEL}"
            )
        else:
            return {
                "ok": True,
                "generator": OLLAMA_MODEL,
                "embedding": EMBEDDING_MODEL,
                "message": "Ollama ready",
            }

        return {
            "ok": False,
            "generator": OLLAMA_MODEL,
            "embedding": EMBEDDING_MODEL,
            "message": message,
        }

    except requests.RequestException as exc:
        return {
            "ok": False,
            "generator": OLLAMA_MODEL,
            "embedding": EMBEDDING_MODEL,
            "message": (
                "Cannot connect to Ollama. Start Ollama and try again. "
                f"Details: {exc}"
            ),
        }


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Create local embeddings using Ollama's /api/embed endpoint."""
    if not texts:
        return []

    response = requests.post(
        f"{OLLAMA_API_BASE}/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        },
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()

    payload = response.json()

    if "embeddings" not in payload:
        raise RuntimeError(
            f"Unexpected embedding response: {payload}"
        )

    return payload["embeddings"]


def chat_with_ollama(
    system_prompt: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
) -> str:
    """Send a non-streaming chat request to Qwen through Ollama."""
    request_messages = [
        {"role": "system", "content": system_prompt},
        *messages,
    ]

    response = requests.post(
        f"{OLLAMA_API_BASE}/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": request_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        },
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()

    payload = response.json()

    try:
        return payload["message"]["content"].strip()
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected chat response: {payload}"
        ) from exc
