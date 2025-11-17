# AskPanDA Tests

This directory contains test scripts for the AskPanDA system.

## test_prompts.py

Comprehensive test suite for all AskPanDA query types. Tests 19 different prompts across 4 categories:

### Usage

```bash
cd /mnt/c/Users/Lo/IdeaProjects/ask-panda
python3 tests/test_prompts.py
```

### Test Categories

1. **Document Queries (7 tests)**
   - What is PanDA?
   - Do you know about AskPanDA?
   - How do I use pathena?
   - What is a PanDA pilot?
   - How does the PanDA pilot work?
   - How do I get help with using PanDA?
   - Explain PanDA job workflow

2. **Task Queries (4 tests)**
   - Tell me about task 47250094
   - What is the status of task 47250094?
   - Are there any problems with task 47250094?
   - Is task 47250094 finished?

3. **Log Analysis Queries (3 tests)**
   - Why did job 6873281623 fail?
   - What caused the error in job 6873281623?
   - Can you analyze the failure of job 6873281623?

4. **Edge Cases / Robustness Tests (5 tests)**
   - Tell me about task 99999999999 (non-existent task)
   - What is AskPanDA? (self-referential query)
   - Just a number: 47250094 (ambiguous query)
   - Show me 47250094 (implicit task lookup)
   - Is task 47250094 finished? (status phrasing)

### Output

Results are printed to stdout and also saved to `/tmp/askpanda_test_results.json` with detailed response data.

### Requirements

- The AskPanDA server must be running on `http://localhost:8000`
- Python 3.7+
- `requests` library installed
