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

"""This script is a simple command-line agent that interacts with a RAG (Retrieval-Augmented Generation) server."""

import argparse
import asyncio
import logging
import os
import requests
import sys
from json import JSONDecodeError
from time import sleep
from typing import Optional

from tools.context_memory import ContextMemory
from tools.errorcodes import EC_TIMEOUT
from tools.server_utils import MCP_SERVER_URL, check_server_health

# mcp = FastMCP("panda") # Removed unused instance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("document_query_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
memory = ContextMemory()


class DocumentQueryAgent:
    """A simple command-line agent that interacts with a RAG server to answer questions."""
    def __init__(self, model: str, session_id: str, mcp_instance: Optional[object] = None) -> None:
        """
        Initialize the DocumentQueryAgent with a model, session ID, and MCP instance.

        Args:
            model (str): The model to use for generating the answer.
            session_id (str): The session ID for tracking the conversation.
            mcp_instance: The PandaMCP instance to use for RAG queries. Optional for HTTP fallback.
        """
        self.model = model  # e.g., OpenAI or Anthropic wrapper
        self.session_id = session_id  # Session ID for tracking conversation
        self.mcp = mcp_instance

    async def ask(self, question: str) -> str:
        """
        Send a question to the RAG server and retrieve the answer.

        Args:
            question (str): The question to ask the RAG server.

        Returns:
            str: The answer from the RAG server. If an error occurs during the
                 request, or if the server responds with an error, a string
                 prefixed with "Error:" is returned detailing the issue.
        """
        # Construct prompt - keep it simple since rag_query() will add context and formatting
        prompt = question

        # If session_id is provided, add conversation history as context
        # (it might not be set e.g. from OpenWebUI since that can handle memory itself)
        if self.session_id != "None":
            history = memory.get_history(self.session_id)
            if history:
                history_text = "\n".join([f"Previous: {user_msg}\nAnswer: {agent_msg}"
                                         for user_msg, agent_msg in history])
                prompt = f"{history_text}\n\nCurrent question: {question}"
        try:
            if self.mcp:
                answer = await self.mcp.rag_query(prompt, self.model)
            else:
                answer = await self._query_via_http(prompt)

            if answer.startswith("Error:"):
                logger.info(answer, file=sys.stderr)
                return ""

            if self.session_id != "None":
                memory.store_turn(self.session_id, question, answer)
                logger.info(
                    f"Answer stored in session ID {self.session_id}:\n\nquestion={question}\n\nanswer={answer}")

            # convert to dictionary before returning if necessary
            if self.session_id == "None":
                return answer

            answer = {
                "session_id": self.session_id,
                "question": question,
                "model": self.model,
                "answer": answer
            }
            return answer
        except Exception as e:
            return f"Error: An unexpected error occurred during RAG query - {e}"

    async def _query_via_http(self, prompt: str) -> str:
        """
        Fallback method that queries the running AskPanDA HTTP server when no MCP instance is provided.
        """
        base_url = os.getenv("MCP_SERVER_URL", MCP_SERVER_URL)
        server_url = f"{base_url.rstrip('/')}/rag_ask"

        def _post():
            return requests.post(server_url, json={"question": prompt, "model": self.model}, timeout=30)

        try:
            response = await asyncio.to_thread(_post)
        except requests.exceptions.RequestException as e:
            return f"Error: Network issue or server unreachable - {e}"

        if response.ok:
            try:
                payload = response.json()
                answer = payload["answer"]
            except JSONDecodeError:
                return "Error: Could not decode JSON response from server."
            except KeyError:
                return "Error: 'answer' key missing in server response."

            return answer

        try:
            error_data = response.json()
            if isinstance(error_data, dict) and "detail" in error_data:
                return f"Error: {error_data['detail']}"
            return f"Error: Server returned status {response.status_code} - {response.text}"
        except JSONDecodeError:
            pass
        return f"Error: Server returned status {response.status_code} - {response.text}"

def main() -> None:
    """
    Parse command-line arguments, call the RAG server, and print the response.

    This function serves as the main entry point for the command-line agent.
    It expects two arguments: the question to ask and the model to use.
    It calls the `ask` function to get a response from the RAG server.
    If the `ask` function returns an error (a string prefixed with "Error:"),
    this error is printed to `sys.stderr` and the script exits with status 1.
    Otherwise, the successful answer is printed to `sys.stdout`.

    Raises:
        SystemExit: If the number of command-line arguments is incorrect, or
                    if an error occurs during the RAG server request.
    """
    # Check server health before proceeding
    # This part of the code is for standalone execution of the agent,
    # so it still needs to check the server health if it's not running within the server.
    ec = check_server_health()
    if ec == EC_TIMEOUT:
        logger.warning(f"Timeout while trying to connect to {MCP_SERVER_URL}.")
        sleep(10)  # Wait for a while before retrying
        ec = check_server_health()
        if ec:
            logger.error("MCP server is not healthy after retry. Exiting.")
            sys.exit(1)
    elif ec:
        logger.error("MCP server is not healthy. Exiting.")
        sys.exit(1)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process some arguments.")

    parser.add_argument('--session-id', type=str, default="None",
                        help='Session ID for the context memory')
    parser.add_argument('--question', type=str, required=True,
                        help='The question to ask the RAG server')
    parser.add_argument('--model', type=str, required=True,
                        help='The model to use for generating the answer')
    args = parser.parse_args()

    # When running as a standalone script, we need to initialize PandaMCP
    # This is a simplified initialization for command-line testing.
    # In the actual server, 'mcp' is a global instance.
    from ask_panda_server import PandaMCP, vectorstore_manager # Import here to avoid circular dependency
    from pathlib import Path
    resources_dir = Path("../resources")
    chroma_dir = Path("../chromadb")
    mcp_instance = PandaMCP("panda", resources_dir, chroma_dir, vectorstore_manager)

    agent = DocumentQueryAgent(args.model, args.session_id, mcp_instance)
    answer = asyncio.run(agent.ask(args.question))
    logger.info(f"Answer:\n{answer}")


if __name__ == "__main__":
    main()
