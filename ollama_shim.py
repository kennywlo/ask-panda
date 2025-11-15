#!/usr/bin/env python3
"""
ollama_shim.py
──────────────────────────────────────────────────────────────────────
Minimal Ollama-compatible façade for Ask-PanDA.

This FastAPI application exposes the handful of endpoints that Open WebUI
expects from an Ollama server. Instead of talking to a local Ollama daemon,
each request is forwarded to the Ask-PanDA HTTP API, which in turn uses the
Mistral API (or any other LLM provider configured inside Ask-PanDA).

Default Configuration:
    - Backend: Ask-PanDA agent endpoint (`/agent_ask`) for chat
    - LLM: Failover via Ask-PanDA (`auto` → Mistral then gpt-oss:20b)
    - Model exposed: askpanda-auto:latest
    - Default port: 11434 (11435 in Docker to avoid conflicts)

Endpoints implemented:
    GET  /api/tags      → model listing with :latest tag support
    GET  /api/ps        → running models list
    GET  /api/version   → version information
    POST /api/chat      → translate chat requests to Ask-PanDA RAG
    POST /api/generate  → translate generate requests to Ask-PanDA RAG

Environment Variables:
    ASK_PANDA_BASE_URL        - Ask-PanDA server URL (default: http://localhost:8000)
    OLLAMA_SHIM_MODEL         - Legacy single backend model name (default: auto)
    OLLAMA_SHIM_MODEL_DISPLAY - Legacy single display name (default: askpanda-auto)
    OLLAMA_SHIM_MODELS        - Optional JSON list of {"display": "...", "backend": "..."} entries
    OLLAMA_SHIM_PORT          - Port to listen on (default: 11434)
    OLLAMA_SHIM_VERSION       - Version string (default: v0.0-shim)
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Ask-PanDA Ollama Shim", version="0.1.0")

ASK_PANDA_BASE_URL = os.getenv("ASK_PANDA_BASE_URL", "http://localhost:8000")
# Use agent_ask for chat endpoint to get proper task/job routing
ASK_PANDA_AGENT_ENDPOINT = f"{ASK_PANDA_BASE_URL.rstrip('/')}/agent_ask"
ASK_PANDA_RAG_ENDPOINT = f"{ASK_PANDA_BASE_URL.rstrip('/')}/rag_ask"

OLLAMA_SHIM_MODEL = os.getenv("OLLAMA_SHIM_MODEL", "auto")
OLLAMA_SHIM_MODEL_DISPLAY = os.getenv(
    "OLLAMA_SHIM_MODEL_DISPLAY", f"{OLLAMA_SHIM_MODEL}-proxy"
)
OLLAMA_VERSION = os.getenv("OLLAMA_SHIM_VERSION", "v0.0-shim")


def _utcnow_iso() -> str:
    return _dt.datetime.utcnow().isoformat() + "Z"


def _tagged_display(name: str) -> str:
    """Append :latest to a display name if no explicit tag is present."""
    return name if ":" in name else f"{name}:latest"


def _normalize_backend_name(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _augment_with_priority_defaults(registry: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Ensure every model declared in ASK_PANDA_MODEL_PRIORITY is routable
    through the shim so Open WebUI can request them explicitly.
    """
    seen = {_normalize_backend_name(entry["backend"]) for entry in registry}
    priority_raw = os.getenv("ASK_PANDA_MODEL_PRIORITY", "")
    for item in priority_raw.split(","):
        backend = item.strip()
        if not backend:
            continue
        key = _normalize_backend_name(backend)
        if key in seen:
            continue
        display = backend if ":" in backend else f"{backend}-proxy"
        registry.append({"display": display, "backend": backend})
        seen.add(key)
    return registry


def _load_model_registry() -> List[Dict[str, str]]:
    """
    Create a registry of {display, backend} pairs.

    Supports legacy single-model env vars as well as a JSON blob supplied via
    OLLAMA_SHIM_MODELS. Example value:
        export OLLAMA_SHIM_MODELS='[
          {"display": "Mistral-Proxy", "backend": "mistral"},
          {"display": "Gemini-Proxy", "backend": "gemini"}
        ]'
    """
    default = [{"display": OLLAMA_SHIM_MODEL_DISPLAY, "backend": OLLAMA_SHIM_MODEL}]
    raw = os.getenv("OLLAMA_SHIM_MODELS")
    if not raw:
        return _augment_with_priority_defaults(default)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse OLLAMA_SHIM_MODELS (%s); falling back to default", exc)
        return _augment_with_priority_defaults(default)

    registry: List[Dict[str, str]] = []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        logger.warning("OLLAMA_SHIM_MODELS must be a list/dict; falling back to default")
        return _augment_with_priority_defaults(default)

    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        display = entry.get("display") or entry.get("name")
        backend = entry.get("backend") or entry.get("model")
        if display and backend:
            registry.append({"display": str(display), "backend": str(backend)})

    if not registry:
        logger.warning("OLLAMA_SHIM_MODELS produced no valid entries; using default")
        return _augment_with_priority_defaults(default)

    return _augment_with_priority_defaults(registry)


MODEL_REGISTRY = _load_model_registry()


def _strip_tag_lower(name: Optional[str]) -> Optional[str]:
    return name.split(":")[0].lower() if name else None


def _resolve_model(model_name: Optional[str]) -> tuple[str, str]:
    """
    Return (backend_model, display_name_with_tag) for the requested model.
    Falls back to the first configured model if no match is found.
    """
    default = MODEL_REGISTRY[0]
    if not model_name:
        return default["backend"], _tagged_display(default["display"])

    requested = _strip_tag_lower(model_name)
    for entry in MODEL_REGISTRY:
        display_base = _strip_tag_lower(entry["display"])
        backend_base = _strip_tag_lower(entry["backend"])
        if requested in {display_base, backend_base}:
            return entry["backend"], _tagged_display(entry["display"])

    logger.info("Unknown model '%s', defaulting to '%s'", model_name, default["display"])
    return default["backend"], _tagged_display(default["display"])


def _extract_user_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    Collapse an Ollama `messages` list into a single text prompt.

    The simplest workable strategy (sufficient for the shim) is to concatenate
    the conversation using `role:` prefixes. The last user message is the one
    we send to Ask-PanDA, prefixed with short context for continuity.
    """
    if not messages:
        return ""

    history_lines = [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}".strip()
                     for msg in messages[:-1]]
    last_msg = messages[-1]
    last_user = last_msg.get("content", "")

    history_prefix = "\n".join(history_lines).strip()
    if history_prefix:
        combined = f"{history_prefix}\nuser: {last_user}"
    else:
        combined = last_user
    return combined.strip()


async def _call_ask_panda(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            logger.debug("Posting to %s with payload %s", endpoint, payload)
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        msg = f"Ask-PanDA returned {exc.response.status_code}: {exc.response.text}"
        logger.error(msg)
        raise HTTPException(status_code=exc.response.status_code, detail=msg)
    except httpx.RequestError as exc:
        msg = f"Failed to reach Ask-PanDA: {exc}"
        logger.error(msg)
        raise HTTPException(status_code=502, detail=msg)

    try:
        return response.json()
    except ValueError:
        msg = "Ask-PanDA response was not valid JSON"
        logger.error(msg)
        raise HTTPException(status_code=502, detail=msg)


@app.get("/api/tags")
async def list_models() -> Dict[str, Any]:
    """
    Mirror Ollama's /api/tags to satisfy Open WebUI model discovery.
    """
    models = []
    for entry in MODEL_REGISTRY:
        model_name = _tagged_display(entry["display"])
        models.append(
            {
                "name": model_name,
                "model": model_name,
                "modified_at": _utcnow_iso(),
            }
        )
    return {"models": models}


@app.get("/api/ps")
async def list_running_models() -> Dict[str, Any]:
    """
    Report a fake running model list. This is primarily for UI niceties.
    """
    models = []
    for entry in MODEL_REGISTRY:
        model_name = _tagged_display(entry["display"])
        models.append(
            {
                "name": model_name,
                "model": model_name,
                "digest": "shim",
                "size": 0,
                "details": {},
                "expires_at": None,
            }
        )
    return {"models": models}


@app.get("/api/version")
async def version() -> Dict[str, Any]:
    """
    Provide a synthetic version string.
    """
    return {"version": OLLAMA_VERSION}


@app.post("/api/chat")
async def chat(request: Request) -> JSONResponse:
    """
    Emulate Ollama's /api/chat endpoint by forwarding the last user prompt to
    Ask-PanDA's agent endpoint.
    """
    data: Dict[str, Any] = await request.json()

    messages: Optional[List[Dict[str, Any]]] = data.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    prompt = _extract_user_prompt(messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="Could not extract user prompt")

    # Normalize model name (strip :latest tag, map to backend model)
    requested_model = data.get("model")
    model_id, resolved_display = _resolve_model(requested_model)

    ask_payload = {"question": prompt, "model": model_id}
    ask_response = await _call_ask_panda(ASK_PANDA_AGENT_ENDPOINT, ask_payload)

    answer = ask_response.get("answer", "")
    response_body = {
        "model": requested_model or resolved_display,
        "created_at": _utcnow_iso(),
        "message": {"role": "assistant", "content": answer},
    }

    return JSONResponse(response_body)


@app.post("/api/generate")
async def generate(request: Request) -> JSONResponse:
    """
    Minimal support for Ollama's /api/generate endpoint (prompt-based).
    """
    data: Dict[str, Any] = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    # Normalize model name (strip :latest tag, map to backend model)
    requested_model = data.get("model")
    model_id, resolved_display = _resolve_model(requested_model)

    ask_payload = {"question": prompt, "model": model_id}
    ask_response = await _call_ask_panda(ASK_PANDA_RAG_ENDPOINT, ask_payload)

    answer = ask_response.get("answer", "")
    response_body = {
        "model": requested_model or resolved_display,
        "created_at": _utcnow_iso(),
        "response": answer,
        "done": True,
    }
    return JSONResponse(response_body)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("OLLAMA_SHIM_PORT", "11434"))
    uvicorn.run("ollama_shim:app", host="0.0.0.0", port=port, reload=False)
