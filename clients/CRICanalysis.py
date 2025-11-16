'''
CRIC database analysis client

WorkFlow:

-- Call Gemini
    -- Verify whether the question is related to the CRIC database or not.
    -- Generate the fields (database schema) that could relate to the question.
    -- Generate the SQL query

-- Discuss with SQLcriticClient
    -- The SQLcriticClient will review the SQL query
        If the SQL query is fine, then approve it.
        If the SQL query is problematic, then return with suggestions
            The CRICanalysisClient will regenerate SQL according to the suggestion.
    -- After MAX_DISCUSS
        If no conclusion is reached, just returns with warnings.
        If conclusion is reached, continue to execute.                   

-- Execute without LLM
    -- Execute the SQL query, fetch the data

-- Call Gemini
    -- Take the data as the input context, output the answer based on the question.

Note:
    1. Currently the default LLM model is Gemini for simplicity.
    2. This is just used as a lightweighted demo, in the near future, it should
       cooperate with the "database fetching client", "selection agent client",
       and "final answer client" to give a thorough response on the question.
    3. It takes the CRICdescription.txt and cric_schema.txt as the background knowledge,
       which should be improved in the near future.

The two-agent discussion workflow is like this:

Question ====> CRICanalysisClient -------> SQL query  ======> execute SQL =======> LLM =======> Answer
                      Λ                        |                Λ         context
                      |                        |                |
                      |         false          V         true   |
                 suggestions  <------- SQLcriticClient --------->       

Rui XUE, r.xue@cern.ch
Oct, 2025
'''
import sys
import sqlite3
import os
import argparse
import numpy as np
import google.generativeai as genai
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from keybert import KeyBERT

MAX_DISCUSS = 3
FIELDS_LIMIT = 10
ROWS_LIMIT = 30
TABLE = "queuedata"

current_dir = Path(__file__).parent

sys.path.append(str(current_dir.parent / "tools"))
import txt2vecdb
from tools.server_utils import call_model_with_failover

# if the vector database does not exist,
# call txt2vecdb function to convert schema.txt to schemaDB
if not (current_dir.parent / "resources" / "schemaDB").exists():
    txt2vecdb.ToVecDB()
    print("Schema Vector Database Generated! \n")

schema_path = current_dir.parent / "resources" / "cric_schema.txt"
CRICdb_path = current_dir.parent / "resources" / "queuedata.db"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
keyword_model = KeyBERT(model="all-MiniLM-L6-v2")
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _call_llm(model: str, prompt: str) -> tuple[str, int, int]:
    """
    Call the specified LLM model with the given prompt.

    Args:
        model: Model name (e.g., 'mistral', 'gemini', 'auto')
        prompt: The prompt to send to the model

    Returns:
        tuple: (response_text, prompt_tokens, total_tokens)
    """
    if model == "gemini":
        # Call Gemini directly
        gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip() if hasattr(response, "text") else str(response)
        prompt_tokens = response.usage_metadata.prompt_token_count
        total_tokens = response.usage_metadata.total_token_count
        return response_text, prompt_tokens, total_tokens
    else:
        # Use model with failover (mistral, auto, llama, gpt-oss:20b)
        response_text = call_model_with_failover(model, prompt)
        # For non-Gemini models, we don't have token counts
        return response_text, 0, 0

def ans_yes_or_no(response: str) -> bool:
    emb_resp = embedder.embed_query(response)
    emb_yes = embedder.embed_query("yes")
    emb_no = embedder.embed_query("no")

    sim_yes = cos_sim(emb_resp, emb_yes)
    sim_no = cos_sim(emb_resp, emb_no)

    print(f"sim_yes={sim_yes:.3f}, sim_no={sim_no:.3f}")
    return sim_yes > sim_no

def need_CRIC(question: str, model: str = "auto") -> bool:

    '''
    Verify whether the question needs CRIC or not.

    If there are key words like 'Please use CRIC..', then even though
    it is a general question about CERN, the output of this question
    will still return True. This is to give the user more freedom in
    getting professional and detailed answers.

    The LLM should return only 'yes' or 'no'. However, if the LLM returns
    positive or negative answers, the program would call 'ans_yes_or_no()'
    to give a boolean result.
    '''

    classifier_prompt = f"""
    You are a binary classifier for ATLAS PanDA.

    Decide if answering the question requires data from the CRIC
    (Computing Resource Information Catalogue).

    CRIC covers operational details about:
    - PanDA sites, queues, endpoints, copytools, brokerage, storage
    - job environments, resources, CPU/memory, pilots, harvesters
    - site status, country, region, cloud, RC site, configuration

    Rules:
    • If the question is **conceptual or general** (e.g. what PanDA is,
    how it works, or general physics/software topics) → **no**
    • If it is **operational, monitoring, or configuration-related**
    (e.g. site activity, queue status, job location, pilot/core info,
    copytools, brokerage, etc.) → **yes**
    • If it is somewhat general but **explicitly requests CRIC data**
    or would need CRIC for a detailed explanation → **yes**

    Answer strictly with 'yes' or 'no'.

    Examples:
    Q: What is CRIC? → yes
    Q: How many jobs run on CERN grid? → yes
    Q: Which sites have failed pilots? → yes
    Q: Where are my jobs running? → yes
    Q: Which queues use Harvester? → yes
    Q: What is the current state of US sites? → yes
    Q: How does PanDA choose where to send jobs? → yes
    Q: What is PanDA? Need CRIC database search. → yes

    Q: What is PanDA? → no
    Q: Who developed PanDA? → no
    Q: Explain PanDA architecture. → no
    Q: What is the purpose of CRIC? → no
    Q: What is ATLAS? → no
    Q: What is Tier-1? → no

    Question: "{question}"
    Answer only 'yes' or 'no'.
    """
    resp_text, prompt_tokens, total_tokens = _call_llm(model, classifier_prompt)
    print("\n Classifier Prompt Tokens: ", prompt_tokens, ", Total Tokens: ", total_tokens)
    resp_text = resp_text.strip().lower()
    if "yes" in resp_text: return True
    if "no"  in resp_text: return False
    return ans_yes_or_no(resp_text)

def llm_suggest_fields(question: str, schema_text: str, model: str = "auto") -> list[str]:

    """
    Use LLM to analyze the CRIC schema and suggest relevant fields.
    Returns a list of column names.

    The fields are selected according to the cric_schema.txt, which was
    generated from Paul's json2db repository. This work flow shall be
    automated in the near future.

    The maximum number of fields is set by 'FIELDS_LIMIT'.
    If there are more fields suggested, the fields are embedded and
    calculated similarities with the question keywords by KeyBERT().
    Then the top fields are selected after ranking.
    """

    prompt = f"""
    You are given the CRIC queuedata database schema.

    Schema excerpt:
    {schema_text}

    Question: "{question}"

    Which fields (columns) are most relevant to answer this question?
    Output only a comma-separated list of field names, without explanations.
    """

    # Ask LLM model
    ans, prompt_tokens, total_tokens = _call_llm(model, prompt)
    print("\n Field Suggestor Prompt Tokens: ", prompt_tokens, ", Total Tokens: ", total_tokens)
    ans = ans.strip()
    fields = [f.strip() for f in ans.split(",") if f.strip()]

    # rank and filter out fields
    # using KeyBERT to extract keywords from a question
    keywords_tuple = keyword_model.extract_keywords(question, keyphrase_ngram_range=(1, 1), stop_words=None)
    keywords = [kw for kw, _ in keywords_tuple]

    # print("keywords ", keywords)
    # print("original recommended fields ", fields)

    k_embs = np.array(embedder.embed_documents(keywords))
    k_score = np.mean(k_embs, axis=0)

    fields_similarity = []
    for i in range(len(fields)):
        f_score = np.array(embedder.embed_query(fields[i]))
        fields_similarity.append([fields[i], cos_sim(f_score, k_score)])

    fields_similarity = sorted(fields_similarity, key= lambda x: x[1], reverse=True)

    top_fields = [f for f, _ in fields_similarity[:FIELDS_LIMIT]]

    return top_fields

def llm_generate_SQL(question: str, fields: list[str], model: str = "auto") -> str:

    prompt = f"""
    You are given the question that requires CRIC database data

    Fields:
    {fields}

    Table:
    {TABLE}

    Question: "{question}"

    According to the fields and the question, please generate a SQL query to fetch
    the data from the CRIC database.

    Rules:
    1. Since we are using SQLite, please generate SQL query that could be executed by SQLite.
    2. Avoid guessing unknown column values.
    3. If the question is conceptual or descriptive and does not map to numeric or categorical data,
       generate a simple SELECT statement that lists the most relevant columns without a WHERE clause.
    4. Output only the raw SQL statement, without markdown, explanation, or comments.
    5. Do NOT generate SQL statments that could modify the database.
    6. If the question is conceptual or descriptive and does not map to numeric or categorical data,
       limit the maximum returned rows to be {ROWS_LIMIT}.
    """
    # Ask LLM model
    sql_text, prompt_tokens, total_tokens = _call_llm(model, prompt)

    if "```sql" in sql_text or "```" in sql_text:
        sql_text = sql_text.replace("```sql", "").replace("```", "").strip()
    print("\n SQL generation Prompt Tokens: ", prompt_tokens, ", Total Tokens: ", total_tokens)

    return sql_text

class CRICanalysisClient:
    """
    1. Justify whether answering a specific question needs CRIC database search.
    2. Generate SQL lines to select fields related to the question.
    3. Execute the SQL lines and get results.
    4. Summarize the results and generate the context for answering.
    """
    def __init__(self, schema_path: str, model: str = "auto"):
        self.schema_path = Path(schema_path)
        self.schema_text = self.schema_path.read_text(encoding="utf-8")
        self.SQLquery = None
        self.model = model

    def is_related(self, question: str) -> bool:
        return need_CRIC(question, self.model)

    def suggest_fields(self, question: str):
        fields = llm_suggest_fields(question, self.schema_text, self.model)
        # print(f"[LLM-suggested fields for '{question}'] → {fields}")
        return fields

    def generate_SQL(self, question: str, fields: list[str]):
        self.SQLquery = llm_generate_SQL(question, fields, self.model)
        print(f"\n SQL query -> {self.SQLquery}")
        return self.SQLquery

    def execute_SQL(self, SQLquery: str):
        try:
            conn = sqlite3.connect(CRICdb_path)
            cursor = conn.cursor()
            cursor.execute(SQLquery)

            # only execute select method to prevent modification
            if SQLquery.strip().lower().startswith("select"):
                columns = [desc[0] for desc in cursor.description]  # list of column names
                rows = cursor.fetchall() # fetch all selected rows

                # Convert list[tuple] → list[dict]
                data = {col: [row[i] for row in rows] for i, col in enumerate(columns)}
            else:
                Err = f"This SQL query is trying to modify the database: \n {SQLquery} \n Aborted"
                cursor.close()
                conn.close()
                return {"success": False, "data": [], "error": Err}
            
            cursor.close()
            conn.close()
            return {"success": True, "data": data, "error": None}
        
        except sqlite3.Error as e:
            return {"success": False, "data": [], "error": str(e)}

    def Answer_with_Context(self, question:str, context):
        prompt = f"""
        You are a CRIC database analyze AI-agent, the user asked a
        question, and the workflow has fetched data from the CRIC database
        which could be related to answering the question.

        The following SQL query was executed:
        {self.SQLquery}

        With the given context from the CRIC database: {context},
        Please use this context and try to answer the question: {question}

        Note:
        1. If this is a specific question asking for the status of the PanDA
           system, please use all of the context.
        2. If this is a general question, try to use part of the context to
           help answer the question in a more professional way. But do not
           refuse to answer, even if the context could be unrelated to the
           question. If you do not know the answer, just say "I do not know."
        """

        if len(prompt) > 20000:  # rough guard
            prompt = prompt[:20000] + "\n[Context truncated due to length.]"

        resp_text, prompt_tokens, total_tokens = _call_llm(self.model, prompt)
        print("\nFinal Answer Tokens: ", prompt_tokens, ", Total Tokens: ", total_tokens)
        return resp_text.strip()

    def ask(self, question: str) -> str:
        """
        Unified ask() method to match the interface of other agents.
        Returns an answer string based on CRIC database query.
        """
        # Import here to avoid circular dependency
        from clients.SQLcritic import SQLcriticClient

        # Check if question needs CRIC
        if not self.is_related(question):
            return "This question does not require CRIC database information."

        # Suggest relevant fields
        fields = self.suggest_fields(question)

        # Two-client discussion workflow
        critic_client = SQLcriticClient(question, None)
        ques = str(question)
        _bool = True

        for _ in range(MAX_DISCUSS):
            SQLquery = self.generate_SQL(ques, fields)
            critic_client.query = SQLquery
            _bool, _suggestion = critic_client.criticize()
            if _bool:
                break
            ques = str(question) + " " + _suggestion

        if not _bool:
            return f"Could not generate a valid SQL query after {MAX_DISCUSS} attempts. Please rephrase your question."

        # Execute SQL and get answer
        result = self.execute_SQL(SQLquery)
        if result["success"]:
            return self.Answer_with_Context(question, result["data"])
        else:
            return f"Database error: {result['error']}"


def workflow(args):
    from clients.SQLcritic import SQLcriticClient

    client = CRICanalysisClient(schema_path, model=getattr(args, 'model', 'auto'))
    critic_client = SQLcriticClient(args.question, None)

    if (client.is_related(args.question)):
        fields = client.suggest_fields(args.question)

        ques = str(args.question)
        # two client discussion, no more than MAX_DISCUSS rounds
        _bool = True
        for _ in range(MAX_DISCUSS):
            SQLquery = client.generate_SQL(ques, fields)
            # if _ == 0:
            #     SQLquery = "select distinct name from queuedata where copytools != 'rucio'"
            critic_client.query = SQLquery
            _bool, _suggestion = critic_client.criticize()
            if (_bool):
                break
            print("Suggestion: ", _suggestion)
            ques = str(args.question) + _suggestion
        if not _bool:
            print(f"Cannot generate reasonable SQL query within {MAX_DISCUSS} rounds. \n Please rephrase the question instead. \n")
            return
        
        result = client.execute_SQL(SQLquery)
        print("\n")
        print("$"*20)
        if (result["success"]):
            answer = client.Answer_with_Context(args.question,result["data"])
            print("\nAnswer by LLM: \n")
            print(answer)
        else:
            print(result["error"])
    else:
        print("No CRIC data is required for this question.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Suggest CRIC fields for a question")
    parser.add_argument("--question", "-q", required=True, help="Natural-language question about CRIC")
    parser.add_argument("--model", "-m", default="auto", help="Model to use (auto, mistral, gemini, etc.)")
    args = parser.parse_args()
    workflow(args)