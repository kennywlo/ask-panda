# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Authors:
# - Paul Nilsson, paul.nilsson@cern.ch, 2025

"""Utility functions for server management."""

import logging
import os
import requests

from tools.errorcodes import EC_OK, EC_SERVERNOTRUNNING, EC_CONNECTIONPROBLEM, EC_TIMEOUT, EC_UNKNOWN_ERROR

# MCP server IP and env vars
MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://ask-panda:8000")
MISTRAL_API_URL: str = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
LLAMA_API_URL: str = os.getenv("LLAMA_API_URL", "http://192.168.100.97:11434/api/generate")
LLAMA_MODEL: str = os.getenv("LLAMA_MODEL", "gpt-oss:20b")
AUTO_MODEL_ALIASES = {"auto", "default", "failover", "hybrid"}
SUPPORTED_MODELS = {
    "mistral",
    "anthropic",
    "openai",
    "llama",
    "gpt-oss:20b",
    "gemini",
}


def _parse_model_priority() -> list[str]:
    raw = os.getenv("ASK_PANDA_MODEL_PRIORITY", "mistral,gpt-oss:20b")
    priority = []
    for entry in raw.split(","):
        model = entry.strip().lower()
        if not model:
            continue
        if model not in SUPPORTED_MODELS:
            logger.warning("Ignoring unsupported model '%s' in ASK_PANDA_MODEL_PRIORITY", model)
            continue
        priority.append(model)
    if not priority:
        priority = ["mistral", "gpt-oss:20b"]
    return priority


MODEL_PRIORITY = _parse_model_priority()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("server_utils.log"),
        logging.StreamHandler()  # Optional: keeps logs visible in console too
    ]
)
logger = logging.getLogger(__name__)


def call_mistral_direct(prompt: str, timeout: int = 60) -> str:
    """
    Call the Mistral API directly using the configured API key.

    Args:
        prompt: The prompt to send to Mistral.
        timeout: Timeout in seconds for the HTTP request.

    Returns:
        str: The text content returned by Mistral, or an error message prefixed with "Error:".
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return "Error: MISTRAL_API_KEY not set in environment."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    try:
        response = requests.post(
            MISTRAL_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return f"Error: Mistral API request failed - {exc}"

    if not response.ok:
        return f"Error: Mistral API responded with status {response.status_code} - {response.text}"

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as exc:
        return f"Error: Unexpected response format from Mistral API - {exc}"


def call_ollama_direct(prompt: str, timeout: int = 120) -> str:
    """
    Call the configured Ollama endpoint directly.

    Args:
        prompt: Prompt text to send to Ollama.
        timeout: HTTP timeout in seconds.

    Returns:
        str: Model response text or an error string prefixed with 'Error:'.
    """
    payload = {
        "model": LLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(
            LLAMA_API_URL,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return f"Error: Ollama request failed - {exc}"

    if not response.ok:
        return f"Error: Ollama responded with status {response.status_code} - {response.text}"

    try:
        data = response.json()
        return data.get("response", "")
    except ValueError as exc:
        return f"Error: Unexpected response format from Ollama - {exc}"


def _resolve_model_sequence(model: str) -> list[str]:
    normalized = (model or "").lower().strip()
    if not normalized:
        return []
    if normalized in AUTO_MODEL_ALIASES:
        return MODEL_PRIORITY
    if normalized not in SUPPORTED_MODELS:
        return []
    return [normalized]


def call_model_with_failover(model: str, prompt: str) -> str:
    """
    Attempt to call the requested model, falling back according to configured priority.
    """
    sequence = _resolve_model_sequence(model)
    if not sequence:
        return f"Error: Unsupported model '{model}'"

    last_error = None
    for idx, candidate in enumerate(sequence):
        logger.info("Attempting model '%s' (failover index %d)", candidate, idx)
        if candidate == "mistral":
            result = call_mistral_direct(prompt)
        elif candidate in {"llama", "gpt-oss:20b"}:
            result = call_ollama_direct(prompt)
        else:
            # For models that aren't handled synchronously here (anthropic, openai, gemini),
            # we simply skip because those are only called asynchronously inside the server.
            continue

        if isinstance(result, str) and result.strip().lower().startswith("error:"):
            last_error = result
            if idx + 1 < len(sequence):
                logger.warning(
                    "Model '%s' failed (%s). Falling back to '%s'.",
                    candidate,
                    result,
                    sequence[idx + 1],
                )
            else:
                logger.error("Model '%s' failed with no more fallbacks: %s", candidate, result)
            continue
        if idx > 0:
            logger.info("Failover succeeded using backup model '%s'", candidate)
        return result

    logger.error("All configured models failed. Last error: %s", last_error)
    return last_error or "Error: All configured models failed."


def check_server_health() -> int:
    """
    Check the health of the MCP server.
    This function attempts to connect to the MCP server's health endpoint
    and checks if it returns a status of "ok".
    If the server is reachable and healthy, it returns EC_OK.
    If the server is not running or there is a connection problem,
    it returns EC_SERVERNOTRUNNING or EC_CONNECTIONPROBLEM respectively.

    Returns:
        Server error code (int): Exit code indicating the health status of the MCP server.
    """
    try:
        response = requests.get(f"{MCP_SERVER_URL}/health", timeout=5)
        response.raise_for_status()
        if response.json().get("status") == "ok":
            logger.info("MCP server is running.")
            return EC_OK
        logger.warning("MCP server is not running.")
        return EC_SERVERNOTRUNNING
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout):
        logger.warning(f"Timeout while trying to connect to MCP server at {MCP_SERVER_URL}.")
        return EC_TIMEOUT
    except requests.RequestException as e:
        logger.warning(f"Cannot connect to MCP server at {MCP_SERVER_URL}: {e}")
        return EC_CONNECTIONPROBLEM
    except Exception as e:
        logger.error(f"An unexpected error occurred while checking MCP server health: {e}")
        return EC_UNKNOWN_ERROR
