#!/usr/bin/env python3
"""
Fixed test script for logits/logprobs with proper parameters
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

def test_logits_fixed():
    """Test logits with proper parameters."""
    
    # Test with different models
    test_models = [
        "openai/gpt-4o-mini",  # Cheaper for testing
        "openai/gpt-4o",
        "qwen/qwen-2.5-72b-instruct",
    ]
    
    prompt = "Generate SQL: SELECT * FROM counties AS t0 INNER JOIN states AS t1 ON t0.state_id = t1.id"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print("🧪 Testing Logits/Logprobs with Fixed Parameters")
    print("=" * 60)
    
    for i, model in enumerate(test_models):
        print(f"\n[{i+1}/{len(test_models)}] Testing {model}...")
        
        # For OpenAI models, we need both logprobs and top_logprobs
        if model.startswith("openai/"):
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 100,
                "logprobs": True,  # This is required for OpenAI models
                "top_logprobs": 5  # This specifies how many alternatives to return
            }
        else:
            # For other models, try just top_logprobs
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 100,
                "top_logprobs": 5
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
                print(f"   ✅ Generated: {content[:50]}...")
                
                # Check for logprobs in response
                choice = result['choices'][0]
                
                if 'logprobs' in choice and choice['logprobs'] is not None:
                    logprobs = choice['logprobs']
                    print(f"   📊 Logprobs found: {type(logprobs)}")
                    
                    if logprobs and 'content' in logprobs:
                        content_logprobs = logprobs['content']
                        print(f"   📋 Content logprobs: {len(content_logprobs)} tokens")
                        
                        # Show first few tokens with logprobs
                        for j, token_info in enumerate(content_logprobs[:3]):
                            token = token_info.get('token', 'N/A')
                            logprob = token_info.get('logprob', 'N/A')
                            print(f"      Token {j+1}: '{token}' (logprob: {logprob:.4f})")
                            
                            # Show top logprobs for this token
                            if 'top_logprobs' in token_info:
                                top_probs = token_info['top_logprobs']
                                print(f"         Top {len(top_probs)} alternatives:")
                                for k, alt in enumerate(top_probs[:3]):
                                    alt_token = alt.get('token', 'N/A')
                                    alt_logprob = alt.get('logprob', 'N/A')
                                    print(f"           {k+1}. '{alt_token}' (logprob: {alt_logprob:.4f})")
                        
                        # Save detailed results
                        model_name = model.replace("/", "_")
                        output_file = f"logits_fixed_{model_name}.json"
                        with open(output_file, 'w') as f:
                            json.dump(result, f, indent=2)
                        print(f"   💾 Saved to: {output_file}")
                        
                        # SUCCESS! Store the working config
                        if model.startswith("openai/"):
                            print(f"   ✅ SUCCESS: OpenAI model supports logprobs with logprobs=True + top_logprobs=5")
                        
                else:
                    print(f"   ❌ No logprobs in response")
                    print(f"   📝 Provider: {result.get('provider', 'Unknown')}")
                    
            else:
                print(f"   ❌ API error: {response.status_code}")
                print(f"   📝 Error: {response.text[:200]}...")
                
        except Exception as e:
            print(f"   ❌ Request failed: {str(e)}")
    
    print(f"\n{'='*60}")
    print("✅ Fixed logits test completed!")
    print("\n🔍 FINDINGS:")
    print("- OpenAI models require: logprobs=True + top_logprobs=N")
    print("- Other models may not support logprobs via OpenRouter")
    print("- Working models can be integrated into batch generator")

if __name__ == "__main__":
    test_logits_fixed() 