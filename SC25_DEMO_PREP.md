# AskPanDA - SC25 Live Demo Preparation Guide

## Critical Pre-Demo Setup

### 1. System Reliability
- [ ] Test all endpoints with 20+ consecutive queries
- [ ] Verify Mistral API key is valid and has sufficient quota
- [ ] Set up API key rotation/backup (Mistral as fallback)
- [ ] Monitor response times (should be <10s for task queries, <3s for docs)
- [ ] Run `./pre_demo_check.sh` to validate vector store load, Ollama shim, and disk space
- [ ] Test with conference WiFi beforehand if possible

### 2. Docker & Infrastructure
- [ ] Use `docker-compose up -d` to run in detached mode
- [ ] Verify health checks: `curl http://localhost:8000/health`
- [ ] Check logs for any warnings: `docker compose logs ask-panda --tail=100`
- [ ] Restart containers before demo to clear any accumulated state
- [ ] Have Docker images pre-built (no building during demo)

### 3. Demo Environment
- [ ] Run on local laptop (not dependent on external servers)
- [ ] Have backup laptop ready with identical setup
- [ ] Test with projector/external display
- [ ] Disable system updates/notifications during demo
- [ ] Close unnecessary applications (free up RAM)

## Recommended Demo Flow

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

## Talking Points

### Technical Highlights
- **Multi-Agent Architecture**: Automatic routing between document/task/log_analyzer agents
- **Hybrid Approach**: Rule-based + LLM classification for robustness
- **RAG System**: ChromaDB vector store with 5 indexed documents
- **Live Data Integration**: Real-time task metadata from BigPanda
- **Log Analysis**: Automated download and analysis of job failure logs with expert/non-expert guidance
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
