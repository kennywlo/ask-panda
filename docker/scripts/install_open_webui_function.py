#!/usr/bin/env python3
"""
Ensures the Ask PanDA function is registered inside the Open WebUI SQLite DB.

This script is executed inside the Open WebUI container during startup. It
waits for the database file to appear, loads the Ask PanDA pipe source, and
creates or updates the corresponding function entry so it shows up in the UI
by default.
"""

import os
import sys
import time
from pathlib import Path

# Make sure we can import the Open WebUI backend modules when this script runs
BACKEND_PATH = Path(os.environ.get("OPEN_WEBUI_BACKEND_PATH", "/app/backend"))
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from sqlalchemy.exc import OperationalError  # type: ignore

from open_webui.models.functions import (  # type: ignore
    FunctionForm,
    FunctionMeta,
    Functions,
)
from open_webui.utils.plugin import (  # type: ignore
    load_function_module_by_id,
    replace_imports,
)


FUNCTION_ID = os.environ.get("ASK_PANDA_FUNCTION_ID", "ask_panda_api_pipe")
FUNCTION_NAME = os.environ.get("ASK_PANDA_FUNCTION_NAME", "Ask PanDA")
FUNCTION_DESCRIPTION = os.environ.get(
    "ASK_PANDA_FUNCTION_DESCRIPTION",
    "Routes prompts to the Ask PanDA MCP server via its HTTP API.",
)
FUNCTION_SOURCE_PATH = Path(
    os.environ.get(
        "ASK_PANDA_FUNCTION_PATH",
        "/ask-panda/open-webui/ask_panda_api_pipe.py",
    )
)
FUNCTION_USER_ID = os.environ.get("ASK_PANDA_FUNCTION_USER_ID", "system")
DB_PATH = Path(os.environ.get("OPEN_WEBUI_DB_PATH", "/app/backend/data/webui.db"))
MAX_ATTEMPTS = int(os.environ.get("ASK_PANDA_FUNCTION_MAX_ATTEMPTS", "20"))
SLEEP_SECONDS = float(os.environ.get("ASK_PANDA_FUNCTION_RETRY_SECONDS", "3"))


def wait_for_db():
    """Block until the SQLite database file exists."""
    deadline = time.time() + 180  # 3 minutes should be plenty for migrations
    while not DB_PATH.exists():
        if time.time() > deadline:
            raise RuntimeError(
                f"Timed out waiting for Open WebUI DB at {DB_PATH} to be created."
            )
        time.sleep(1)


def ensure_function_installed():
    """Create or update the Ask PanDA function definition."""
    if not FUNCTION_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Expected function source at {FUNCTION_SOURCE_PATH} is missing."
        )

    content = FUNCTION_SOURCE_PATH.read_text(encoding="utf-8")
    content = replace_imports(content)

    _, function_type, frontmatter = load_function_module_by_id(
        FUNCTION_ID, content=content
    )

    meta = FunctionMeta(description=FUNCTION_DESCRIPTION, manifest=frontmatter or {})
    form_data = FunctionForm(
        id=FUNCTION_ID,
        name=FUNCTION_NAME,
        content=content,
        meta=meta,
    )

    existing = Functions.get_function_by_id(FUNCTION_ID)

    if existing:
        Functions.update_function_by_id(
            FUNCTION_ID,
            {
                "name": FUNCTION_NAME,
                "content": content,
                "meta": meta.model_dump(),
                "type": function_type,
            },
        )
    else:
        Functions.insert_new_function(FUNCTION_USER_ID, function_type, form_data)

    # Ensure it is active and visible to every user.
    Functions.update_function_by_id(
        FUNCTION_ID,
        {
            "is_active": True,
            "is_global": True,
        },
    )


def main():
    wait_for_db()

    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            ensure_function_installed()
            print(
                f"[install_open_webui_function] '{FUNCTION_ID}' registered/updated successfully."
            )
            return
        except OperationalError:
            attempt += 1
            time.sleep(SLEEP_SECONDS)
        except Exception as exc:
            # Fail fast on unrecoverable errors so they show up in docker logs.
            raise exc

    raise RuntimeError(
        f"Failed to install '{FUNCTION_ID}' after {MAX_ATTEMPTS} attempts "
        "due to repeated database lock errors."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[install_open_webui_function] ERROR: {error}", file=sys.stderr)
        sys.exit(1)
