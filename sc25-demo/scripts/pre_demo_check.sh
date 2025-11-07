#!/bin/bash
# Pre-Demo System Verification for SC25

echo "╔════════════════════════════════════════════╗"
echo "║  AskPanDA - SC25 Demo System Check        ║"
echo "╚════════════════════════════════════════════╝"
echo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0

test_check() {
    local test_name="$1"
    local result="$2"
    
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        ((PASS_COUNT++))
    else
        echo -e "${RED}✗${NC} $test_name"
        ((FAIL_COUNT++))
    fi
}

# Test 1: Docker containers running
echo "━━━ Infrastructure Tests ━━━"
if docker compose ps ask-panda | grep -q "Up"; then
    test_check "Docker container running" "PASS"
else
    test_check "Docker container running" "FAIL"
fi

# Test 2: Health endpoint
if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    test_check "Health endpoint responding" "PASS"
else
    test_check "Health endpoint responding" "FAIL"
fi

# Test 3: RAG system loaded
if docker compose logs ask-panda | grep -q "Vectorstore built successfully"; then
    test_check "Vector store initialized" "PASS"
else
    test_check "Vector store initialized" "FAIL"
fi

echo
echo "━━━ Performance Tests ━━━"

# Test 4: Document query (fast)
echo -n "Testing document query speed... "
START=$(date +%s.%N)
RESPONSE=$(curl -sX POST "http://localhost:8000/agent_ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is PanDA?", "model": "mistral"}' \
  --max-time 10 2>&1)
END=$(date +%s.%N)
DURATION=$(echo "$END - $START" | bc)

if [ $? -eq 0 ] && echo "$RESPONSE" | grep -q "answer"; then
    if (( $(echo "$DURATION < 6" | bc -l) )); then
        test_check "Document query (<6s): ${DURATION}s" "PASS"
    else
        test_check "Document query (${DURATION}s, >6s threshold)" "FAIL"
    fi
else
    test_check "Document query (timeout or error)" "FAIL"
fi

# Test 5: Task query (slower but impressive)
echo -n "Testing task query... "
START=$(date +%s.%N)
RESPONSE=$(curl -sX POST "http://localhost:8000/agent_ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about task 47250094", "model": "mistral"}' \
  --max-time 15 2>&1)
END=$(date +%s.%N)
DURATION=$(echo "$END - $START" | bc)

if [ $? -eq 0 ] && echo "$RESPONSE" | grep -q "Description"; then
    if (( $(echo "$DURATION < 15" | bc -l) )); then
        test_check "Task query (<15s): ${DURATION}s" "PASS"
    else
        test_check "Task query (${DURATION}s, >15s threshold)" "FAIL"
    fi
else
    test_check "Task query (timeout or error)" "FAIL"
fi

# Test 6: Ollama shim (Open WebUI integration)
if curl -sf http://localhost:11435/api/tags > /dev/null 2>&1; then
    test_check "Ollama shim responding" "PASS"
else
    test_check "Ollama shim responding" "FAIL"
fi

echo
echo "━━━ API Resources ━━━"

# Test 7: Mistral API key
if [ -n "$MISTRAL_API_KEY" ]; then
    test_check "Mistral API key configured" "PASS"
else
    test_check "Mistral API key configured" "FAIL"
fi

# Test 8: Check disk space
DISK_USAGE=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 90 ]; then
    test_check "Disk space available (${DISK_USAGE}% used)" "PASS"
else
    test_check "Disk space (${DISK_USAGE}% used, >90%)" "FAIL"
fi

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results: ${GREEN}${PASS_COUNT} passed${NC}, ${RED}${FAIL_COUNT} failed${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ System is DEMO READY!${NC}"
    echo
    exit 0
else
    echo -e "${RED}✗ System has issues - please investigate${NC}"
    echo
    exit 1
fi
