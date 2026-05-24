"""
Thin wrapper around the Ollama Python SDK.

Provides a chat interface (multi-turn with message history) and
a one-shot generate interface used for testing / debugging.
"""
from __future__ import annotations

import logging
from typing import List

import ollama

import config

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Wraps the Ollama client with sane defaults pulled from config.py.

    Usage::

        client = OllamaClient()
        reply = client.chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "Summarise attention mechanisms."},
        ])
    """

    def __init__(
        self,
        model: str = config.OLLAMA_CHAT_MODEL,
        base_url: str = config.OLLAMA_BASE_URL,
    ) -> None:
        self.model = model
        self._client = ollama.Client(host=base_url)
        logger.info("OllamaClient ready  model=%s  url=%s", model, base_url)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def chat(self, messages: List[dict]) -> str:
        """
        Send a list of chat messages and return the assistant's reply.

        Args:
            messages: List of {"role": …, "content": …} dicts.
                      Roles: "system" | "user" | "assistant"

        Returns:
            The assistant's reply as a plain string.
        """
        logger.debug("Sending %d message(s) to %s", len(messages), self.model)
        response = self._client.chat(model=self.model, messages=messages)
        return response.message.content

    def generate(self, prompt: str) -> str:
        """
        One-shot text generation (no message history).

        Args:
            prompt: Raw prompt string.

        Returns:
            Generated text as a plain string.
        """
        response = self._client.generate(model=self.model, prompt=prompt)
        return response.response

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable and the model exists."""
        try:
            models = self._client.list()
            names = [m.model for m in models.models]
            # Accept both exact match and prefix match (e.g. "llama3.1" matches "llama3.1:latest")
            return any(self.model in n or n.startswith(self.model) for n in names)
        except Exception as exc:
            logger.warning("Ollama availability check failed: %s", exc)
            return False
