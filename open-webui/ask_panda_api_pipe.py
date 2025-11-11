"""
AskPanDA API Pipe for Open WebUI

This pipe calls the ask-panda HTTP API instead of importing Python code directly.
Much simpler and avoids dependency conflicts!
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field


def _resolve_default_api_url() -> str:
    """Prefer explicit env vars but fall back to the Docker network hostname."""
    explicit = os.getenv("ASK_PANDA_API_URL")
    if explicit:
        return explicit
    base = os.getenv("ASK_PANDA_BASE_URL")
    if base:
        return f"{base.rstrip('/')}/agent_ask"
    return "http://ask-panda:8000/agent_ask"


DEFAULT_ASK_PANDA_URL = _resolve_default_api_url()


class Pipe:
    class Valves(BaseModel):
        ask_panda_url: str = Field(
            default=DEFAULT_ASK_PANDA_URL,
            description="Ask PanDA endpoint",
        )
        model: str = Field(
            default="gpt-oss:20b",
            description="LLM model",
        )

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Emit progress updates back to Open WebUI"
        )

    def __init__(self):
        self.name = "Ask PanDA"
        self.valves = self.Valves()

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Any] = None,
        __event_call__: Optional[str] = None,
    ) -> str | Dict[str, Any]:
        if __event_call__ in {"follow_ups", "followups", "title", "notes"}:
            return {"follow_ups": []}

        prompt = self._extract_last_user_prompt(body.get("messages") or [])
        if not prompt:
            return "No user prompt provided."

        user_valves_data = (__user__ or {}).get("valves") or {}
        try:
            user_valves = self.UserValves(**user_valves_data)
        except Exception:
            user_valves = self.UserValves()

        if user_valves.show_status and __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Contacting Ask PanDA…", "done": False},
                }
            )

        payload = {"question": prompt, "model": self.valves.model}
        try:
            data = await asyncio.to_thread(
                self._call_ask_panda, self.valves.ask_panda_url, payload
            )
        except Exception as exc:  # noqa: BLE001
            if user_valves.show_status and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Ask PanDA error", "done": True},
                    }
                )
            return f"Error contacting Ask PanDA: {exc}"

        if user_valves.show_status and __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Ask PanDA response ready", "done": True},
                }
            )

        return data.get("answer", "No answer provided.")

    @staticmethod
    def _extract_last_user_prompt(messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
        return ""

    @staticmethod
    def _call_ask_panda(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()
