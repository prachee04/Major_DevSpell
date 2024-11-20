from langchain.llms.base import LLM
from typing import Optional
import os
import requests

class ChatGroq(LLM):
    api_key: str
    model: str = "llama-3.1-70b-versatile"
    api_url: str = "https://api.groq.ai/v1/completions"  # Replace with actual endpoint

    def _call(self, prompt: str, stop: Optional[str] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 2000,
            "temperature": 0.7,
        }
        response = requests.post(self.api_url, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"ChatGroq API call failed: {response.text}")
        
        return response.json().get("choices")[0].get("text", "").strip()

    @property
    def _llm_type(self) -> str:
        return "chat_groq"
