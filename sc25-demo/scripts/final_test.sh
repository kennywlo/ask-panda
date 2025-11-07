#!/bin/bash
echo "=== Comprehensive Open WebUI Test ==="
echo

echo "Test 1: Document query - What is PanDA?"
curl -sX POST "http://localhost:11435/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral-proxy", "messages": [{"role": "user", "content": "What is PanDA?"}]}' \
  --max-time 30 | jq -r '.message.content' | head -3
echo

echo "Test 2: Task query - Tell me about task 47250094"  
curl -sX POST "http://localhost:11435/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral-proxy", "messages": [{"role": "user", "content": "Tell me about task 47250094"}]}' \
  --max-time 40 | jq -r '.message.content' | head -5
echo

echo "Test 3: Another document query - Do you know about AskPanDA?"
curl -sX POST "http://localhost:11435/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral-proxy", "messages": [{"role": "user", "content": "Do you know about AskPanDA?"}]}' \
  --max-time 30 | jq -r '.message.content' | head -3
echo

echo "=== All tests complete ===" 
