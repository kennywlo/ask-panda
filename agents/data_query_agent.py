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

"""This agent can download task metadata from PanDA and ask an LLM to analyze the relevant parts."""

import argparse
import ast
# import asyncio
import google.generativeai as genai
import logging
import os
import re
import requests
import sys
from collections import deque
from typing import Optional
# from fastmcp import FastMCP
from time import sleep

from tools.context_memory import ContextMemory
from tools.errorcodes import EC_NOTFOUND, EC_OK, EC_UNKNOWN_ERROR, EC_TIMEOUT
from tools.https import get_base_url
from tools.server_utils import (
    MCP_SERVER_URL,
    check_server_health,
    call_model_with_failover,
)
from tools.tools import fetch_data, read_json_file

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("data_query_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
memory = ContextMemory()
# mcp = FastMCP("panda")


class TaskStatusAgent:
    """
    A simple agent that can give information about task status.
    This agent fetches metadata from PanDA, extracts relevant parts, and asks an LLM for analysis.
    """
    def __init__(
            self,
            model: str,
            taskid: str,
            cache: str,
            session_id: str,
            query_type: str = "task",
            identifier_type: str = "task"
    ) -> None:
        """
        Initialize the TaskStatusAgent with a model.

        Args:
            model (str): The model to use for generating the answer (e.g., 'openai', 'anthropic').
            taskid (str): The PanDA job ID to analyze.
            cache (str): The location of the cache directory for storing downloaded files.
            session_id (str): The session ID for tracking the conversation.
            query_type (str): The type of query, 'job' or 'task' (default: 'task').
        """
        self.model = model  # e.g., OpenAI or Anthropic wrapper
        self.query_type = query_type
        self.identifier_type = identifier_type if identifier_type in {"task", "pandaid"} else "task"
        self.last_error: str | None = None
        self.taskid: Optional[int] = None
        self.pandaid: Optional[int] = None
        self._resolved_job: Optional[dict] = None
        self._storage_id = str(taskid)
        try:
            if self.identifier_type == "pandaid":
                self.pandaid = int(taskid)
            else:
                self.taskid = int(taskid)
        except ValueError:
            logger.error(f"Invalid {self.identifier_type} ID: {taskid}. It should be an integer.")
            sys.exit(1)
        self.session_id = session_id
        self.cache = cache
        if not os.path.exists(self.cache):
            logger.info(f"Cache directory {self.cache} does not exist. Creating it.")
            try:
                os.makedirs(self.cache)
            except OSError as e:
                logger.error(f"Failed to create cache directory {self.cache}: {e}")
                sys.exit(1)

        path = os.path.join(os.path.join(self.cache, "tasks"), self._storage_id)
        if not os.path.exists(path):
            logger.info(f"Creating directory for task {self._storage_id} in cache.")
            try:
                os.makedirs(path)
            except OSError as e:
                logger.error(f"Failed to create directory {path}: {e}")
                sys.exit(1)

    def ask(self, question: str) -> str:
        """
        Send a question to the LLM and retrieve the answer.

        Args:
            question (str): The question to ask the LLM.

        Returns:
            str: The answer returned by the LLM.
        """
        # Call LLM directly to avoid HTTP deadlock
        try:
            if self.model == "gemini":
                gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
                response = gemini_model.generate_content(question)
                answer = response.text
            elif self.model in {"auto", "mistral", "llama", "gpt-oss:20b"}:
                answer = call_model_with_failover(self.model, question)
            else:
                # For other models, fall back to HTTP (but this creates deadlock risk)
                server_url = f"{MCP_SERVER_URL}/llm_ask"
                response = requests.post(server_url, json={"question": question, "model": self.model}, timeout=30)
                if not response.ok:
                    return f"requests.post() error: {response.text}"
                answer = response.json()["answer"]
        except Exception as e:
            err = f"Error calling LLM: {e}"
            logger.error(err)
            return f"error: {err}"

        if not answer:
            err = "No answer returned from the LLM."
            logger.error(f"{err}")
            return 'error: {err}'

        if answer.lower().startswith("error:"):
            logger.error(answer)
            return answer

        # Strip code block formatting
        try:
            clean_code = re.sub(r"^```(?:python)?\n|\n```$", "", answer.strip())
        except re.error as e:
            logger.error(f"Regex error while cleaning code: {e}")
            return 'error: {err}'

        # convert the answer to a Python dictionary
        try:
            answer_dict = ast.literal_eval(clean_code)
        except (SyntaxError, ValueError) as e:
            err = f"Error converting answer to dictionary: {e}"
            logger.error(f"{err}")
            return 'error: {err}'

        if not answer_dict:
            err = "Failed to store the answer as a Python dictionary."
            logger.error(f"{err}")
            return 'error: {err}'

        # format the answer for better readability
        formatted_answer = format_answer(answer_dict)
        if not formatted_answer:
            logger.error(f"Failed to format the answer from the LLM using: {answer_dict}")
            sys.exit(1)

        logger.info(f"Answer from {self.model.capitalize()}:\n{formatted_answer}")

        # store the answer in the session memory
        if self.session_id != "None":
            memory.store_turn(self.session_id, "Investigate the job failure", formatted_answer)
            logger.info(f"Answer stored in session ID {self.session_id}:\n\nquestion={question}\n\nanswer={formatted_answer}")

        return formatted_answer

    # async def fetch_all_data(self) -> tuple[int, dict or None, dict or None]:
    def fetch_all_data(self) -> tuple[int, dict or None, dict or None]:
        """
        Fetch metadata from PanDA for a given task ID.

        Returns:
            Exit code (int): The exit code indicating the status of the operation.
            File dictionary (dict): A dictionary containing the file names and their corresponding paths.
            Metadata dictionary (dict): A dictionary containing the relevant metadata for the task.
        """
        self.last_error = None
        _metadata_dictionary = {}
        _file_dictionary = {}

        # Download metadata and pilot log concurrently
        workdir = os.path.join(self.cache, "tasks")
        base_url = get_base_url()
        fetch_key = self.taskid if self.identifier_type != "pandaid" else self.pandaid
        if self.identifier_type == "pandaid":
            url = f"{base_url}/jobs/?pandaid={self.pandaid}&json&mode=nodrop"
            filename = f"pandaid_{self.pandaid}.json"
        elif self.query_type == "job":
            url = f"{base_url}/jobs/?jeditaskid={self.taskid}&json&mode=nodrop"
            filename = "metadata.json"
        else:
            url = f"{base_url}/task/{self.taskid}/?json"
            filename = "metadata.json"
        metadata_success, metadata_message = fetch_data(
            fetch_key,
            filename=filename,
            jsondata=True,
            workdir=workdir,
            url=url
        )

        if metadata_success != 0:
            self.last_error = f"Task {self.taskid} metadata could not be retrieved from PanDA."
            identifier = self.pandaid if self.identifier_type == "pandaid" else self.taskid
            logger.warning(f"Failed to fetch metadata for identifier {identifier} - will not be able to analyze the status")
            return EC_NOTFOUND, _file_dictionary, _metadata_dictionary

        logger.info(f"Downloaded JSON file: {metadata_message}")
        _file_dictionary["json"] = metadata_message

        task_data = read_json_file(metadata_message)
        if not task_data:
            self.last_error = f"Task {self.taskid} metadata file was empty or unreadable."
            logger.warning(f"Error: Failed to read the JSON data from {metadata_message}.")
            return EC_UNKNOWN_ERROR, None, None

        # For job queries, check if the 'jobs' key exists and is not empty
        if (self.query_type == "job" or self.identifier_type == "pandaid") and (not task_data.get("jobs")):
            target = f"PandaID {self.pandaid}" if self.identifier_type == "pandaid" else f"task {self.taskid}"
            self.last_error = (
                f"{target} contains no jobs. "
                "The PanDA monitor returned an empty job list for this identifier."
            )
            logger.warning(f"No jobs found for {target}")
            return EC_NOTFOUND, _file_dictionary, _metadata_dictionary

        if self.identifier_type == "pandaid":
            jobs = task_data.get("jobs", [])
            if jobs:
                self._resolved_job = jobs[0]
                derived_task = jobs[0].get("jeditaskid")
                if derived_task:
                    self.taskid = derived_task

        # Extract relevant metadata from the JSON data
        try:
            _metadata_dictionary["jobs"] = {}
            for job in task_data['jobs']:
                jobstatus = job.get('jobstatus', 'unknown')
                if jobstatus not in _metadata_dictionary["jobs"]:
                    _metadata_dictionary["jobs"][jobstatus] = 0
                _metadata_dictionary["jobs"][jobstatus] += 1

                for key in job:
                    if "errordiag" in key:
                        if "errordiags" not in _metadata_dictionary:
                            _metadata_dictionary["errordiags"] = deque()
                        if job[key] != "":
                            _metadata_dictionary["errordiags"].append(job[key])
                    if "errorcode" in key:
                        if "errorcodes" not in _metadata_dictionary:
                            _metadata_dictionary["errorcodes"] = deque()
                        value = job[key]
                        if isinstance(value, str):
                            try:
                                value = int(value)
                            except ValueError:
                                value = None
                        if isinstance(value, (int, float)) and value > 0:
                            _metadata_dictionary["errorcodes"].append(value)
            if self.identifier_type == "pandaid" and self._resolved_job:
                _metadata_dictionary["resolved_job"] = self._resolved_job
        except KeyError:
            _metadata_dictionary = task_data.copy()
            # logger.warning(f"Error: Missing key in JSON data: {e}")
            # return EC_UNKNOWN_ERROR, None, None

        return EC_OK, _file_dictionary, _metadata_dictionary

    def formulate_question(self, metadata_dictionary: dict) -> str:
        """
        Construct a question to ask the LLM based on the extracted lines and metadata.

        Args:
            metadata_dictionary:

        Returns:
            str: A formatted question string to be sent to the LLM.
        """
        # jobs = metadata_dictionary.get("jobs", None)
        # if not jobs:
        #    logger.warning("Error: No jobs information found in the metadata dictionary.")
        #    # return ""

        question = "You are an expert on distributed analysis.\n\n"
        question += """
Please provide a summary of the task status based on the metadata provided. The task is identified by its PanDA TaskID.
The dictionary should have the task id as the key (an integer), and its value should include the following fields:

"description": A detailed summary (long and thorough) of the task status in plain English. If there are no jobs, state that the task has not started yet.

"problems": A plain-language explanation of any issues (job failures as listed in the dictionary).

"details": A detailed analysis of the task status, including job counts per status, common error codes, and error diagnostics.

Return only a valid Python dictionary. Here's the metadata dictionary:
        """
        label = f"PandaID ({self.pandaid})" if self.identifier_type == "pandaid" else f"task ID ({self.taskid})"
        question = question.replace("TaskID", label)
        description = str(metadata_dictionary)
        question += f"\n\n{description}\n\n"

        return question

    def generate_question(self) -> str:
        """
        Generate a question to ask the LLM based on the task metadata.

        Returns:
            str: A formatted question string to be sent to the LLM.
        """
        # Fetch the files from PanDA
        # exit_code, file_dictionary, metadata_dictionary = asyncio.run(self.fetch_all_data())
        exit_code, file_dictionary, metadata_dictionary = self.fetch_all_data()
        logger.info(f"metadata_dictionary: {metadata_dictionary}")

        identifier_label = f"PandaID {self.pandaid}" if self.identifier_type == "pandaid" else f"task {self.taskid}"
        if exit_code == EC_NOTFOUND:
            if not self.last_error:
                self.last_error = f"No metadata found for {identifier_label}."
            logger.warning(self.last_error)
            return None  # Return None instead of crashing
        elif not file_dictionary:
            self.last_error = f"Failed to download metadata files for {identifier_label}."
            logger.warning(self.last_error)
            return None  # Return None instead of crashing
        elif not metadata_dictionary:  # Add check for empty metadata dictionary
            self.last_error = f"Metadata dictionary is empty for {identifier_label}."
            logger.warning(self.last_error)
            return None

        # Formulate the question based on the extracted lines and metadata
        question = self.formulate_question(metadata_dictionary)
        if not question:
            logger.warning("No question could be generated.")
            return None  # Return None instead of crashing

        return question


def format_answer(answer: dict) -> str:
    """
    Format the answer dictionary into a human-readable string.

    Args:
        answer (dict): The answer dictionary returned by the LLM.

    Returns:
        str: A formatted string containing the description, non-expert guidance, and expert guidance.
    """
    # the dictionary will only ever contain a single key-value pair
    logger.info(f"answer dictionary to format: {answer} type={type(answer)}")
    task_id, value = next(iter(answer.items()))

    to_store = ""

    # for metadata, the value is a string - otherwise a dictionary
    if isinstance(value, str):
        to_store += value
        return to_store

    description = value.get('description')
    if description:
        to_store += f"**Description:**\n{description}\n\n"

    problems = value.get('problems')
    if problems:
        to_store += f"**Problems:**\n{problems}\n\n"

    details = value.get('details')
    if details:
        to_store += f"**Details:**\n{details}\n\n"

    return to_store


def main():
    """
    Check if the correct number of command-line arguments is provided.

    This ensures that the script is executed with exactly two arguments:
    a question and a model.

    Raises:
        SystemExit: If the number of arguments is not equal to 4.
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

    parser.add_argument('--taskid', type=int, required=True,
                        help='PanDA TaskID (integer)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use (e.g., openai, anthropic, etc.)')
    parser.add_argument('--cache', type=str, default="cache",
                        help='Location of cache directory (default: cache)')
    parser.add_argument('--session-id', type=str, default="None",
                        help='Session ID for the context memory')
    args = parser.parse_args()

    agent = TaskStatusAgent(args.model, args.taskid, args.cache)

    # Generate a proper question to ask the LLM based on the metadata and log files
    question = agent.generate_question()
    logger.info(f"Asking question: \n\n{question}")

    # Ask the question to the LLM
    answer = agent.ask(question)

    logger.info(f"Answer from {args.model.capitalize()}:\n{answer}")

    # store the answer in the session memory
    # ..

    sys.exit(0)


if __name__ == "__main__":
    main()
