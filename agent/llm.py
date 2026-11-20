"""Optional LLM layer (any OpenAI-compatible endpoint).

When LLM_API_KEY is set, the agent asks the model to rewrite the grounded template reply
so it sounds natural, while keeping every fact (IDs, dates, amounts) unchanged. If the call
fails or times out, the template reply is returned, so the agent never blocks on the LLM.
"""
import os

import httpx

SYSTEM = ("You are a concise customer-support assistant. Rewrite the draft reply to sound natural and friendly, "
          "in the same language as the customer. Keep every fact, ID, date and number exactly. Max 2 sentences.")


class LLM:
    def __init__(self):
        self.key = os.getenv("LLM_API_KEY")
        self.base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "4"))

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    def polish(self, customer_msg: str, draft: str) -> str:
        if not self.enabled:
            return draft
        try:
            r = httpx.post(f"{self.base}/chat/completions", timeout=self.timeout,
                           headers={"Authorization": f"Bearer {self.key}"},
                           json={"model": self.model, "temperature": 0.2, "messages": [
                               {"role": "system", "content": SYSTEM},
                               {"role": "user", "content": f"Customer: {customer_msg}\nDraft reply: {draft}"}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip() or draft
        except Exception:
            return draft
