#!/bin/bash
# Test suite for Ask-PanDA agent routing and query handling
# Based on FAQ and README examples

set -e

BASE_URL="http://localhost:8000"
MODEL="gemini"
PASS=0
FAIL=0

echo "======================================"
echo "Ask-PanDA Integration Test Suite"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper function to test a query
test_query() {
    local test_name="$1"
    local question="$2"
    local expected_category="$3"
    local timeout="${4:-10}"

    echo -n "Test: $test_name ... "

    # Make the request
    response=$(curl -s -X POST "$BASE_URL/agent_ask" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"$question\", \"model\": \"$MODEL\"}" \
        --max-time "$timeout" 2>&1)

    # Check if request succeeded
    if echo "$response" | grep -q "answer"; then
        # Extract the answer
        # Extract using python to avoid jq dependency inside containers
        answer=$(python3 -c 'import json, sys
try:
    data = json.loads(sys.argv[1])
    print(data.get("answer", ""))
except Exception:
    pass' "$response" || echo "")

        # Check for errors in answer
        if echo "$answer" | grep -qiE '^\s*error[:\s]'; then
            echo -e "${RED}FAIL${NC} - Error in response"
            echo "  Response: $(echo "$answer" | head -c 100)..."
            FAIL=$((FAIL + 1))
            return 1
        fi

        # Check if answer is substantial (not empty, not generic)
        if [ ${#answer} -lt 50 ]; then
            echo -e "${YELLOW}WARN${NC} - Response too short"
            echo "  Answer: $answer"
            FAIL=$((FAIL + 1))
            return 1
        fi

        # For task queries, check if it contains task-specific data
        if [ "$expected_category" == "task" ]; then
            if echo "$answer" | grep -q "47250094" && echo "$answer" | grep -qi "description"; then
                echo -e "${GREEN}PASS${NC}"
                PASS=$((PASS + 1))
                return 0
            else
                echo -e "${RED}FAIL${NC} - Missing task-specific data"
                FAIL=$((FAIL + 1))
                return 1
            fi
        fi

        # For document queries, check if answer is informative
        if [ "$expected_category" == "document" ]; then
            if echo "$answer" | grep -qi "panda"; then
                echo -e "${GREEN}PASS${NC}"
                PASS=$((PASS + 1))
                return 0
            else
                echo -e "${YELLOW}WARN${NC} - Generic answer"
                echo "  Answer preview: $(echo "$answer" | head -c 100)..."
                PASS=$((PASS + 1))
                return 0
            fi
        fi

        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))

    else
        echo -e "${RED}FAIL${NC} - Request failed or timed out"
        echo "  Response: $(echo "$response" | head -c 200)"
        FAIL=$((FAIL + 1))
        return 1
    fi
}

echo "Testing Document Query Agent (General Questions)"
echo "------------------------------------------------"
test_query "What is PanDA?" "What is PanDA?" "document" 20
sleep 20
test_query "How does pilot work?" "How does the PanDA pilot work?" "document" 20
sleep 20
test_query "Get help with PanDA" "How do I get help with using PanDA?" "document" 1200

echo ""
echo "Testing Task Status Agent (Task Queries)"
echo "------------------------------------------------"
test_query "Standard task query" "Tell me about task 47250094" "task" 60
sleep 20
test_query "Task status query" "What is the status of task 47250094?" "task" 60
sleep 20
test_query "Natural language task" "What happened with task 47250094?" "task" 120

echo ""
echo "Testing Enhanced Classification (Tricky Cases)"
echo "------------------------------------------------"
test_query "Just number (ambiguous)" "47250094" "task" 30
sleep 20
test_query "Task without keyword" "Show me 47250094" "task" 30
sleep 20
test_query "Status inquiry" "Is task 47250094 finished?" "task" 25

echo ""
echo "======================================"
echo "Test Results"
echo "======================================"
echo -e "${GREEN}Passed: $PASS${NC}"
echo -e "${RED}Failed: $FAIL${NC}"
echo "Total: $((PASS + FAIL))"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
