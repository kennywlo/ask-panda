# AskPanDA - SC25 Live Demo Preparation Guide

## Critical Pre-Demo Setup

### 1. System Reliability
- [ ] Test all endpoints with 20+ consecutive queries
- [ ] Verify Mistral API key is valid and has sufficient quota
- [ ] Set up API key rotation/backup (Mistral as fallback)
- [ ] Monitor response times (should be <10s for task queries, <3s for docs)
- [ ] Run `./sc25-demo/scripts/pre_demo_check.sh` to validate vector store load, Ollama shim, and disk space
- [ ] Test with conference WiFi beforehand if possible

### 2. Docker & Infrastructure
- [ ] Use `docker-compose up -d` to run in detached mode
- [ ] Verify health checks: `curl http://localhost:8000/health`
- [ ] Check logs for any warnings: `docker compose logs ask-panda --tail=100`
- [ ] Restart containers before demo to clear any accumulated state
- [ ] Have Docker images pre-built (no building during demo)
- [ ] Confirm `ASK_PANDA_CACHE_DIR` (defaults to `./cache`) points to fast local storage so Selection clients in both CLI and Open WebUI share context

### 3. Demo Environment
- [ ] Run on local laptop (not dependent on external servers)
- [ ] Have backup laptop ready with identical setup
- [ ] Test with projector/external display
- [ ] Disable system updates/notifications during demo
- [ ] Close unnecessary applications (free up RAM)

## Recommended Demo Flow

### Architecture Snapshot (set the stage in <30 s)
- FastAPI MCP server (`ask_panda_server.py`) exposes `/rag_ask`, `/llm_ask`, `/agent_ask`.
- Shared clients (`clients/selection.py`, `document_query.py`, etc.) route every question, whether it originates from CLI scripts or Open WebUI.
- Ollama shim simply re-labels `/agent_ask` as `mistral-proxy` for Open WebUI.

This talking point explains why the demo survives upstream merges and how each surface (CLI, WebUI, APIs) stays consistent.

### Opening (1-2 min)
```bash
# Show system status
curl http://localhost:8000/health

# Show available endpoints
curl http://localhost:8000/docs
```

### Demo Scenario 1: General Documentation Query (30 sec)
**Question**: "What is PanDA?"
**Expected**: Clear description of PanDA WMS
**Shows**: RAG retrieval from documentation

```bash
curl -X POST "http://localhost:8000/agent_ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is PanDA?", "model": "mistral"}' | jq '.'
```

### Demo Scenario 2: Task Status Analysis (1-2 min)
**Question**: "Tell me about task 47250094"
**Expected**: Detailed analysis with Description/Problems/Details
**Shows**: Live API data fetching from BigPanda + LLM analysis

```bash
curl -X POST "http://localhost:8000/agent_ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about task 47250094", "model": "mistral"}' | jq '.'
```

### Demo Scenario 3: Job Failure Analysis (1-2 min)
**Question**: "Why did job 6873281623 fail?"
**Expected**: Detailed analysis with expert/non-expert guidance
**Shows**: Log file download, analysis, and actionable recommendations

```bash
curl -X POST "http://localhost:8000/agent_ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Why did job 6873281623 fail?", "model": "mistral"}' | jq '.'
```

### Demo Scenario 4: Interactive Q&A (2-3 min)
Take live questions from audience, showing:
- Agent routing (document vs task vs log_analyzer queries)
- Real-time classification
- Error handling (if ID doesn't exist)
- Ambiguous phrasing like raw IDs, "Show me 47250094", or "Is task 47250094 finished?"

### Demo Scenario 5: Open WebUI Integration (Optional, 1 min)
Show the Ollama shim integration with Open WebUI
- Navigate to Open WebUI interface
- Ask the same questions
- Show conversational interface

## Pre-Loaded Test Queries

### Document Queries (Fast, Reliable)
```
- What is PanDA?
- Do you know about AskPanDA?
- How do I use pathena?
- What is a PanDA pilot?
- How does the PanDA pilot work?
- How do I get help with using PanDA?
- Explain PanDA job workflow
```

### Task Queries (Impressive, Data-Driven)
```
- Tell me about task 47250094
- What is the status of task 47250094?
- Are there any problems with task 47250094?
- Is task 47250094 finished?
```

### Log Analysis Queries (Advanced, Most Impressive)
```
- Why did job 6873281623 fail?
- What caused the error in job 6873281623?
- Can you analyze the failure of job 6873281623?
```

### Edge Cases (Show Robustness)
```
- Tell me about task 99999999999 (non-existent)
- What is AskPanDA? (self-referential)
- Just a number: 47250094 (ambiguous query)
- Show me 47250094 (implicit task lookup)
- Is task 47250094 finished? (status phrasing without explicit context)
```

## Backup Plans

### If Mistral API Fails
1. Switch to Gemini: set `.env` fallback and restart if quota available
2. Have backup API key ready
3. Restart containers: `docker compose restart`

### If Network Fails
1. Use localhost-only demo (no BigPanda access)
2. Show pre-recorded video of working system
3. Walk through architecture slides instead

### If Docker Fails
1. Have screenshots/video ready
2. Explain architecture with diagrams
3. Offer to do demo after session

## Performance Monitoring Script

```bash
#!/bin/bash
# Run before demo to verify system
echo "=== Pre-Demo System Check ==="
echo "1. Health check:"
curl -s http://localhost:8000/health || echo "FAIL: Server not running"
echo -e "\n2. Document query test:"
time curl -sX POST "http://localhost:8000/agent_ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is PanDA?", "model": "mistral"}' | jq -r '.answer' | head -2
echo -e "\n3. Task query test:"
time curl -sX POST "http://localhost:8000/agent_ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about task 47250094", "model": "mistral"}' | jq -r '.answer' | head -3
echo -e "\n=== All checks complete ==="
```

## CRIC Integration: Operational Query Agent

Ask-PanDA includes intelligent **CRIC (Computing Resource Information Catalogue)** integration for answering operational questions about PanDA sites, queues, and computing resources.

### How CRIC Works

The CRIC agent uses a sophisticated **LLM+SQL** pipeline:

1. **Smart Classification**: Questions are classified to determine if CRIC database is needed
   - Yes: "Which sites use rucio?", "What is the state of Tier-1 resources?"
   - No: "What is PanDA?" (documentation query instead)

2. **Semantic Field Matching**: LLM identifies relevant database columns using context understanding
   - Extracts keywords from the question
   - Matches against 20+ available fields (site, tier, copytools, capacity, etc.)

3. **LLM SQL Generation**: LLM generates SQL SELECT queries based on identified fields

4. **SQL Validation**: SQLcritic agent validates queries for:
   - Safety (only SELECT, no modifications)
   - Correctness (field types match operators)
   - Consistency (schema validation via vector DB)

5. **Database Query Execution**: Executes validated queries against SQLite CRIC database
   - 227 site/queue records in `resources/queuedata.db`
   - Includes complex JSON fields (copytools, params, queue configurations)

6. **Answer Synthesis**: LLM generates human-readable answers from query results

### Example CRIC Queries

```
- "Which sites use the rucio copytool?"
- "List all Tier-1 sites by country"
- "What is the current state of resources in Europe?"
- "Which queues have Harvester enabled?"
- "Show me sites with >100 cores available"
```

### Safety Features

- ✅ **Query Validation**: Only SELECT queries permitted (no INSERT/UPDATE/DELETE)
- ✅ **Schema Safety**: Vector database ensures fields exist before execution
- ✅ **Type Checking**: SQLcritic validates operators match field data types
- ✅ **Iterative Refinement**: LLM discussion loop for complex queries (max 3 rounds)
- ✅ **Context Limits**: Automatic truncation at 20K characters to prevent token overflow

### Technical Components

- `clients/CRICanalysis.py` - Main CRIC agent (439 lines)
- `clients/SQLcritic.py` - SQL validation and safety layer (218 lines)
- `resources/queuedata.db` - SQLite database with CRIC data (1.7 MB)
- `resources/cric_schema.txt` - Database schema documentation (94 lines)
- `tools/txt2vecdb.py` - Vector database generator for schema matching

### Demo Talking Point

"If someone asks about queue configurations or resource availability, AskPanDA automatically routes the question to our CRIC agent, which intelligently identifies relevant database fields, generates and validates SQL queries, and synthesizes human-readable answers—all with built-in safety checks to prevent SQL injection."

## Talking Points

### Technical Highlights
- **Layered Architecture**: FastAPI MCP server + shared client/router layer + Open WebUI shim stay in lockstep with upstream AskPanDA
- **Hybrid Approach**: Rule-based + LLM classification for robustness
- **RAG System**: ChromaDB vector store with 5 indexed documents
- **Live Data Integration**: Real-time task metadata from BigPanda
- **Log Analysis**: Automated download and analysis of job failure logs with expert/non-expert guidance
- **CRIC Database Agent**: LLM+SQL pipeline for intelligent operational queries about sites and resources
- **Model Fallback Chain**: Automatic failover from Mistral to gpt-oss:20b with <20ms switchover latency
- **Direct API Integration**: Mistral SDK calls to avoid HTTP deadlock issues
- **Open WebUI Compatible**: Ollama shim for easy integration

### Use Cases
- **Researchers**: Quick answers about PanDA workflow/tools
- **Operators**: Real-time task status monitoring and automated failure analysis
- **Debugging**: Intelligent log analysis with actionable recommendations for both experts and non-experts
- **New Users**: Interactive onboarding with documentation Q&A

### Future Enhancements (if asked)
- Additional agents (job logs, pilot activity, queue status)
- Multi-modal support (charts, graphs from task data)
- Conversational memory for follow-up questions
- Integration with other ATLAS systems

## Day-Of Checklist

**30 minutes before:**
- [ ] Restart Docker containers
- [ ] Run performance monitoring script
- [ ] Run `./sc25-demo/scripts/test_agent_queries.sh` (documents/tasks/logs) in addition to `./sc25-demo/scripts/final_test.sh`
- [ ] Test all demo queries
- [ ] Check API quotas/rate limits
- [ ] Verify display output on projector

**5 minutes before:**
- [ ] Close all unnecessary applications
- [ ] Disable notifications
- [ ] Open terminal with correct working directory
- [ ] Have backup laptop ready
- [ ] Deep breath!

## Contact & Resources
- GitHub: https://github.com/PanDAWMS/panda-...
- Documentation: [your docs URL]
- Email: [your contact]

---
Good luck at SC25! 🎉
