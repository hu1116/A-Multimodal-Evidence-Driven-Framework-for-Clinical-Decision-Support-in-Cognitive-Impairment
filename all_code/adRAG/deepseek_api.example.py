"""Example DeepSeek client module for the RAG-LLM pipeline.

Copy this file to ``deepseek_api.py`` (which is git-ignored) and adapt it to
your environment. ``ReportToCsv.py`` imports ``DeepSeekClient`` from here and
calls ``client.chat(prompt, model=..., temperature=..., max_tokens=...)``.

    cp adRAG/deepseek_api.example.py adRAG/deepseek_api.py

This template wraps the OpenAI-compatible DeepSeek endpoint. It is provided for
reproducibility only; review and adapt before use. NEVER commit real keys.
"""

from openai import OpenAI


class DeepSeekClient:
    """Minimal OpenAI-compatible wrapper around the DeepSeek chat API."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, prompt: str, model: str = "deepseek-chat",
             temperature: float = 0.0, max_tokens: int = 200) -> str:
        """Return the assistant text for a single-prompt chat completion."""
        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
