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

import argparse
import asyncio
import google.generativeai as genai
import json
import logging
import os
import re
import requests
import sys
from json import JSONDecodeError
from time import sleep

from agents.document_query_agent import DocumentQueryAgent
from agents.log_analysis_agent import LogAnalysisAgent
from agents.data_query_agent import TaskStatusAgent
from tools.errorcodes import EC_TIMEOUT
from tools.server_utils import MCP_SERVER_URL, check_server_health, call_mistral_direct

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# mcp = FastMCP("panda") # Removed unused instance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("selection_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SelectionAgent:
    def __init__(self, agents: dict, model: str, session_id: str = None, cache: str = "/app/cache") -> None:
        self.agents = agents  # dict like {"document": ..., "queue": ...}
        self.model = model        # e.g., OpenAI or Anthropic wrapper
        self.session_id = session_id  # Session ID for tracking conversation
        self.cache = cache  # Cache directory for dynamic agent creation

    def classify_question(self, question: str) -> str:
        """
        Enhanced classification with entity extraction and dynamic agent creation.

        Strategy:
        1. Try regex extraction first (fast path for explicit task/job IDs)
        2. If agents already exist from regex, use rule-based routing
        3. Otherwise, use LLM with structured JSON output to extract IDs and classify
        4. Dynamically create agents if LLM extracts IDs that regex missed
        """
        # Fast path: If we have a task or job agent already, use it directly
        # This means regex extraction succeeded in figure_out_agents()
        if self.agents.get("task") is not None:
            return "task"
        if self.agents.get("log_analyzer") is not None:
            return "log_analyzer"

        # Fallback: Use LLM with structured output for entity extraction + classification
        prompt = f"""Analyze this question and return ONLY a valid JSON object with entity extraction and classification.

Question: "{question}"

Return ONLY valid JSON in this exact format:
{{
    "category": "task|log_analyzer|document|queue|pilot_activity",
    "task_id": null or integer,
    "job_id": null or integer,
    "confidence": "high|medium|low",
    "reasoning": "brief explanation"
}}

Category Definitions:
- task: Questions about specific PanDA task status, job counts, or task metadata
- log_analyzer: Questions about why a specific job failed or log analysis
- document: General usage questions, how-to guides, concepts (PanDA, prun, pathena, error codes)
- queue: Questions about site/queue data (corepower, copytool, queue status, rucio usage)
- pilot_activity: Questions about pilot activity, failures, statistics, or Grafana links

Examples:

Q: "What's happening with 47250094?"
{{"category": "task", "task_id": 47250094, "job_id": null, "confidence": "high", "reasoning": "Numeric ID without context likely refers to task"}}

Q: "Why did job 12345 crash?"
{{"category": "log_analyzer", "task_id": null, "job_id": 12345, "confidence": "high", "reasoning": "Explicit job failure question with job ID"}}

Q: "Tell me about 999888777"
{{"category": "task", "task_id": 999888777, "job_id": null, "confidence": "medium", "reasoning": "Large number likely task ID but context unclear"}}

Q: "How do I use pathena?"
{{"category": "document", "task_id": null, "job_id": null, "confidence": "high", "reasoning": "General how-to question about tool usage"}}

Q: "Which queues support multi-core jobs?"
{{"category": "queue", "task_id": null, "job_id": null, "confidence": "high", "reasoning": "Question about queue capabilities"}}

Now analyze: "{question}"

Return ONLY the JSON object, nothing else.
"""

        try:
            response = self.ask(prompt, returnstring=True).strip()

            # Strip markdown code blocks if present
            if response.startswith("```"):
                response = re.sub(r"^```(?:json)?\n?|\n?```$", "", response.strip())

            result = json.loads(response)
            logger.info(f"LLM classification result: {result}")

            # If LLM found a task_id and we don't have a task agent yet, create one dynamically
            if result.get("task_id") and not self.agents.get("task"):
                task_id = str(result["task_id"])
                logger.info(f"Dynamically creating TaskStatusAgent for task {task_id}")
                try:
                    self.agents["task"] = TaskStatusAgent(
                        self.model,
                        task_id,
                        self.cache,
                        self.session_id
                    )
                    return "task"
                except Exception as e:
                    logger.error(f"Failed to create TaskStatusAgent: {e}")

            # If LLM found a job_id and we don't have a log_analyzer agent yet, create one
            if result.get("job_id") and not self.agents.get("log_analyzer"):
                job_id = str(result["job_id"])
                logger.info(f"Dynamically creating LogAnalysisAgent for job {job_id}")
                try:
                    self.agents["log_analyzer"] = LogAnalysisAgent(
                        self.model,
                        job_id,
                        self.cache,
                        self.session_id
                    )
                    return "log_analyzer"
                except Exception as e:
                    logger.error(f"Failed to create LogAnalysisAgent: {e}")

            # Return the classified category
            category = result.get("category", "document")
            return category if category in ["task", "log_analyzer", "document", "queue", "pilot_activity"] else "document"

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"LLM returned invalid JSON or missing keys: {e}. Response: {response[:200]}")
            # Fallback to simple text classification if JSON parsing fails
            response_lower = response.lower()
            if "task" in response_lower:
                return "task" if self.agents.get("task") else "document"
            elif "log" in response_lower or "job" in response_lower:
                return "log_analyzer" if self.agents.get("log_analyzer") else "document"
            elif "queue" in response_lower:
                return "queue" if self.agents.get("queue") else "document"
            elif "pilot" in response_lower:
                return "pilot_activity" if self.agents.get("pilot_activity") else "document"
            return "document"

    def answer(self, question: str) -> str:
        return self.classify_question(question)

    def ask(self, question: str, returnstring=False) -> str or dict:
        """
        Send a question to the LLM and retrieve the answer.
        Calls Gemini directly to avoid HTTP deadlock when invoked during request handling.

        Args:
            question (str): The question to ask the LLM.
            returnstring (bool): If True, return only the answer string. If False, return a dict.

        Returns:
            str or dict: The answer from the LLM, or a dictionary containing the session ID, if
            returnstring is False. If returnstring is True, returns the answer as a string.
        """
        # Call Gemini directly to avoid HTTP deadlock
        try:
            if self.model == "gemini":
                gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
                response = gemini_model.generate_content(question)
                _answer = response.text
            elif self.model == "mistral":
                _answer = call_mistral_direct(question)
            else:
                # For other models, fall back to HTTP (but this creates deadlock risk)
                server_url = os.getenv("MCP_SERVER_URL", f"{MCP_SERVER_URL}/rag_ask")
                response = requests.post(server_url, json={"question": question, "model": self.model}, timeout=30)
                if not response.ok:
                    error_msg = f"Error: Server returned status {response.status_code} - {response.text}"
                    return error_msg if returnstring else {"session_id": self.session_id, "answer": error_msg}

                try:
                    _answer = response.json()["answer"]
                except (JSONDecodeError, KeyError) as e:
                    error_msg = f"Error: Could not parse server response - {e}"
                    return error_msg if returnstring else {"session_id": self.session_id, "answer": error_msg}

            # Return answer in requested format
            if returnstring:
                return _answer

            answer = {
                "session_id": self.session_id,
                "question": question,
                "model": self.model,
                "answer": _answer
            }
            return answer

        except Exception as e:
            error_msg = f"Error: {e}"
            logger.error(f"Error in SelectionAgent.ask(): {e}")
            return error_msg if returnstring else {"session_id": self.session_id, "answer": error_msg}


def get_agents(model: str, session_id: str or None, pandaid: str or None, taskid: str or None, cache: str, mcp_instance=None) -> dict:
    """
    Create and return a dictionary of agents for different categories.

    Args:
        model (str): The model to use for generating answers.
        session_id (str or None): The session ID for the context memory.
        pandaid (str or None): The PanDA ID for the job or task, if applicable.
        taskid (str or None): The task ID for the job or task, if applicable.
        cache (str): The location of the cache directory.
        mcp_instance: The PandaMCP instance.

    Returns:
        dict: A dictionary mapping agent categories to their respective agent classes.
    """
    return {
        "document": DocumentQueryAgent(model, session_id, mcp_instance),
        "queue": None,
        "task": TaskStatusAgent(model, taskid, cache, session_id) if session_id and taskid else None,
        "log_analyzer": LogAnalysisAgent(model, pandaid, cache, session_id) if pandaid else None,
        "pilot_activity": None
    }


def _find_last_match(pattern: str, text: str) -> int or None:
    """
    Return the last integer captured by pattern in text, if any.
    """
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None
    # re.findall with capturing groups returns a list of tuples if multiple groups; guard for that
    last = matches[-1]
    if isinstance(last, tuple):
        last = next((group for group in reversed(last) if group), None)
    try:
        return int(last) if last is not None else None
    except ValueError:
        return None


def extract_job_id(text: str) -> int or None:
    """
    Extract a job ID from the given text using a regular expression.

    Args:
        text: The text from which to extract the job ID.

    Returns:
        int or None: The extracted job ID as an integer, or None if no job ID is found.
    """
    pattern = r'\b(?:job|panda[\s_]?id)\s+(\d+)\b'
    return _find_last_match(pattern, text)


def extract_task_id(text: str) -> int or None:
    """
    Extract a task ID from the given text using a regular expression.

    Args:
        text: The text from which to extract the task ID.

    Returns:
        int or None: The extracted task ID as an integer, or None if no task ID is found.
    """
    pattern = r'\b(?:task[\s_]?id|task)\s+(\d+)\b'
    task_id = _find_last_match(pattern, text)
    if task_id is not None:
        return task_id

    # fallback: bare numbers near end (e.g., "tell me 47250094"), but only when the question
    # does not mention jobs/panda ids to avoid misrouting log queries.
    if re.search(r'\b(job|panda[\s_]?id)\b', text, re.IGNORECASE):
        return None

    bare_pattern = r'(\d{6,})'
    return _find_last_match(bare_pattern, text)


def figure_out_agents(question: str, model: str, session_id: str, cache: str = None, mcp_instance=None):
    """
    Determine the appropriate agent to handle the given question.
    Args:
        question:
        mcp_instance:

    Returns:

    """
    # does the question contain a job or task id?
    # use a regex to extract "job NNNNN" from args.question
    pandaid = extract_job_id(question)
    taskid = extract_task_id(question)
    return get_agents(model, session_id, pandaid, taskid, cache, mcp_instance)


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
    parser.add_argument('--cache', type=str, default="cache",
                        help='Location of cache directory (default: cache)')

    args = parser.parse_args()

    # does the question contain a job or task id?
    # use a regex to extract "job NNNNN" from args.question
    pandaid = extract_job_id(args.question)
    if pandaid is not None:
        logger.info(f"Extracted PanDA ID: {pandaid}")
    else:
        logger.info("No PanDA ID found in the question.")
    taskid = extract_task_id(args.question)
    if taskid is not None:
        logger.info(f"Extracted Task ID: {taskid}")
    else:
        logger.info("No Task ID found in the question.")

    agents = get_agents(args.model, args.session_id, pandaid, taskid, args.cache)
    selection_agent = SelectionAgent(agents, args.model, session_id=args.session_id, cache=args.cache)

    category = selection_agent.answer(args.question)
    agent = agents.get(category)
    logger.info(f"Selected agent category: {category}")
    if category == "document":
        logger.info(f"Selected agent category: {category} (DocumentQueryAgent)")
        answer = asyncio.run(agent.ask(args.question))
        logger.info(f"Answer:\n{answer}")
        return answer
    elif category == "log_analyzer":
        logger.info(f"Selected agent category: {category} (LogAnalysisAgent)")
        if pandaid is None:
            return "Sorry, I need a PanDA ID to answer questions about job logs."
        question = agent.generate_question("pilotlog.txt")
        answer = agent.ask(question)
        logger.info(f"Answer:\n{answer}")
        return answer
    elif category == "task":
        logger.info(f"Selected agent category: {category} (TaskStatusAgent)")
        if taskid is None:
            return "Sorry, I need a Task ID to answer questions about task status."
        question = agent.generate_question()
        answer = agent.ask(question)
        logger.info(f"Answer:\n{answer}")
        return answer
    else:
        logger.warning("Not yet implemented")
    if agent is None:
        return "Sorry, I don’t have enough information to answer that kind of question."

    return "Sorry, I found no agent to answer that kind of question."


if __name__ == "__main__":
    main()
