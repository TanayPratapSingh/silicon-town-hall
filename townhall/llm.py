"""One place every agent talks to a language model.

`get_llm()` returns the OpenAI client when OPENAI_API_KEY is set, and a FakeLLM
otherwise, so tests and offline runs never need a key or spend money.
"""

import base64
import json
from pathlib import Path

from townhall import config


class OpenAILLM:
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def chat(self, messages, model=None, temperature=0.7, json_mode=False):
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = self.client.chat.completions.create(
            model=model or config.RESIDENT_MODEL, messages=messages, temperature=temperature, **kwargs
        )
        return response.choices[0].message.content

    def look(self, prompt, image_path, model=None):
        """Ask a vision model about one page image; returns its text reply."""
        data = base64.b64encode(Path(image_path).read_bytes()).decode()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
        ]}]
        return self.chat(messages, model=model or config.VISION_MODEL, temperature=0)


class FakeLLM:
    """Deterministic stand in: echoes a canned reply, or one built by `responder(messages)`."""

    def __init__(self, reply="SIMULATED", responder=None):
        self.reply, self.responder, self.calls = reply, responder, []

    def chat(self, messages, model=None, temperature=0.7, json_mode=False):
        self.calls.append(messages)
        out = self.responder(messages) if self.responder else self.reply
        return json.dumps(out) if json_mode and not isinstance(out, str) else out

    def look(self, prompt, image_path, model=None):
        self.calls.append([{"role": "user", "content": prompt, "image": str(image_path)}])
        return self.responder(prompt) if self.responder else self.reply


def get_llm():
    return OpenAILLM() if config.OPENAI_API_KEY else FakeLLM()
