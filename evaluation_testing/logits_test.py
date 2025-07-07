#!/usr/bin/env python3
"""
Test script to check logits/logprobs with OpenRouter API
"""

import json
import os
import pathlib
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv
import requests

# Load environment variables
DOTENV_PATH = pathlib.Path(__file__).resolve().parents[2] / ".env"
load_dotenv(DOTENV_PATH)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
MODEL = "qwen/qwen-2.5-72b-instruct"

def test_logits_single_query():
    """Test logits extraction on a single query."""
    
    # Sample query data
    query_data = {
        "db_id": "address", 
        "natural_query": "What is the relationship between counties and states in the address system?",
        "sql": "SELECT * FROM counties AS t0 INNER JOIN states AS t1 ON t0.state_id = t1.id",
        "path": ["counties", "states"]
    }
    
    prompt = """Generate SQL following the exact path provided.

Database: address

ONLY USE THESE TABLES:

counties:
  - id: INTEGER (PK)
  - name: TEXT
  - state_id: INTEGER

states:
  - id: INTEGER (PK)
  - name: TEXT
  - abbreviation: TEXT

PATH: counties -> states

REFERENCE SQL:
SELECT * FROM counties AS t0 INNER JOIN states AS t1 ON t0.state_id = t1.id

QUESTION: What is the relationship between counties and states in the address system?

RULES:
1. Follow EXACT path: counties -> states
2. Use aliases t0, t1, t2, etc.
3. Copy join patterns from reference
4. SELECT * 

Generate SQL:"""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Test with different top_logprobs values
    test_cases = [
        {"top_logprobs": 1, "name": "top_1"},
        {"top_logprobs": 5, "name": "top_5"},
        {"top_logprobs": 10, "name": "top_10"},
        {"top_logprobs": 0, "name": "disabled"},  # Should disable logprobs
    ]
    
    print("🧪 Testing Logits/Logprobs with OpenRouter")
    print(f"🤖 Model: {MODEL}")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] Testing {test_case['name']}...")
        
        data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 200,
            "top_logprobs": test_case["top_logprobs"]
        }
        
        try:
            response = requests.post(
                f"{OPENAI_API_BASE}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract response content
                content = result['choices'][0]['message']['content']
                print(f"   ✅ Generated: {content[:100]}...")
                
                # Check for logprobs in response
                choice = result['choices'][0]
                
                if 'logprobs' in choice:
                    logprobs = choice['logprobs']
                    print(f"   📊 Logprobs found: {type(logprobs)}")
                    
                    if logprobs and 'content' in logprobs:
                        content_logprobs = logprobs['content']
                        print(f"   📋 Content logprobs: {len(content_logprobs)} tokens")
                        
                        # Show first few tokens with logprobs
                        for j, token_info in enumerate(content_logprobs[:3]):
                            token = token_info.get('token', 'N/A')
                            logprob = token_info.get('logprob', 'N/A')
                            print(f"      Token {j+1}: '{token}' (logprob: {logprob})")
                            
                            # Show top logprobs for this token
                            if 'top_logprobs' in token_info:
                                top_probs = token_info['top_logprobs']
                                print(f"         Top alternatives: {len(top_probs)}")
                                for k, alt in enumerate(top_probs[:2]):
                                    alt_token = alt.get('token', 'N/A')
                                    alt_logprob = alt.get('logprob', 'N/A')
                                    print(f"           {k+1}. '{alt_token}' (logprob: {alt_logprob})")
                    
                    # Save detailed results
                    output_file = f"logits_test_{test_case['name']}.json"
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"   💾 Saved to: {output_file}")
                    
                else:
                    print(f"   ❌ No logprobs in response")
                    print(f"   📝 Response keys: {list(choice.keys())}")
                    
            else:
                print(f"   ❌ API error: {response.status_code}")
                print(f"   📝 Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {str(e)}")
    
    print(f"\n{'='*60}")
    print("✅ Logits test completed!")
    print("Check the generated JSON files for detailed logprob data")

if __name__ == "__main__":
    test_logits_single_query() 