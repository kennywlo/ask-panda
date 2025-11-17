#!/usr/bin/env python3
"""Test script for 19 AskPanDA prompts."""

import requests
import json

BASE_URL = "http://localhost:8000"
MODEL = "mistral"  # Using mistral as the default model

prompts = {
    # DOCUMENT QUERIES (7)
    1: "What is PanDA?",
    2: "Do you know about AskPanDA?",
    3: "How do I use pathena?",
    4: "What is a PanDA pilot?",
    5: "How does the PanDA pilot work?",
    6: "How do I get help with using PanDA?",
    7: "Explain PanDA job workflow",

    # TASK QUERIES (4)
    8: "Tell me about task 47250094",
    9: "What is the status of task 47250094?",
    10: "Are there any problems with task 47250094?",
    11: "Is task 47250094 finished?",

    # LOG ANALYSIS QUERIES (3)
    12: "Why did job 6873281623 fail?",
    13: "What caused the error in job 6873281623?",
    14: "Can you analyze the failure of job 6873281623?",

    # EDGE CASES / ROBUSTNESS TESTS (5)
    15: "Tell me about task 99999999999",
    16: "What is AskPanDA?",
    17: "Just a number: 47250094",
    18: "Show me 47250094",
    19: "Is task 47250094 finished?",
}

results = {}

for num, prompt in prompts.items():
    try:
        response = requests.post(
            f"{BASE_URL}/agent_ask",
            json={"question": prompt, "model": MODEL},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            results[num] = {
                "prompt": prompt,
                "status": "✓ SUCCESS",
                "answer": data.get("answer", "")[:200] + ("..." if len(data.get("answer", "")) > 200 else ""),
                "category": data.get("category", "unknown"),
                "full_response": data
            }
        else:
            results[num] = {
                "prompt": prompt,
                "status": f"✗ HTTP {response.status_code}",
                "error": response.text[:200],
                "full_response": response.text
            }
    except Exception as e:
        results[num] = {
            "prompt": prompt,
            "status": f"✗ ERROR",
            "error": str(e)[:200],
            "full_response": str(e)
        }

# Print results
print("\n" + "="*80)
print("ASKPANDA PROMPT TEST RESULTS")
print("="*80 + "\n")

categories = {
    "DOCUMENT QUERIES (7)": range(1, 8),
    "TASK QUERIES (4)": range(8, 12),
    "LOG ANALYSIS QUERIES (3)": range(12, 15),
    "EDGE CASES / ROBUSTNESS TESTS (5)": range(15, 20),
}

for category_name, prompt_range in categories.items():
    print(f"\n{category_name}")
    print("-" * 80)

    for num in prompt_range:
        if num in results:
            result = results[num]
            print(f"\n{num}. {result['prompt']}")
            print(f"   Status: {result['status']}")
            if "category" in result:
                print(f"   Category: {result['category']}")
            if "answer" in result:
                print(f"   Answer: {result['answer']}")
            if "error" in result:
                print(f"   Error: {result['error']}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
success_count = sum(1 for r in results.values() if r["status"].startswith("✓"))
print(f"Passed: {success_count}/19")
print(f"Failed: {19 - success_count}/19")

# Save detailed results to JSON
with open("/tmp/askpanda_test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDetailed results saved to /tmp/askpanda_test_results.json")
