"""Optional explanation-only OpenRouter call; never part of numerical optimizer."""
from __future__ import annotations
import json
import os
from urllib.request import Request, urlopen

def explain(record: dict) -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return "LLM disabled: set OPENROUTER_API_KEY to request an explanation."
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    payload = {"model": model, "temperature": 0.1, "max_tokens": 450,
               "messages": [
                   {"role": "system", "content":
                    "Explain only the supplied synthetic simulation metrics. "
                    "Do not claim real-world predictions, guarantees or causal proof. "
                    "State limitations and acknowledge missing evidence. "
                    "Do not emit commands or recommend actions on external systems."},
                   {"role": "user", "content": json.dumps(record, ensure_ascii=False)}]}
    request = Request("https://openrouter.ai/api/v1/chat/completions",
                      data=json.dumps(payload).encode("utf-8"),
                      headers={"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json", "HTTP-Referer": "https://localhost"})
    try:
        with urlopen(request, timeout=20) as response:
            result = json.load(response)
        return str(result["choices"][0]["message"]["content"])
    except (OSError, ValueError, KeyError, IndexError) as exc:
        return f"LLM explanation unavailable ({type(exc).__name__}); numerical results remain accessible."
