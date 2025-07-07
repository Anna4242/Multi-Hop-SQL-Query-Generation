#!/usr/bin/env python3
"""
Qwen 3 4B Multi-Hop Test
Test batch_001.json from 5, 10, 15, and 20 hop directories
"""

import json
import os
import sqlite3
import pathlib
import time
import re
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
import requests
from datetime import datetime

# Load environment variables
DOTENV_PATH = pathlib.Path(__file__).resolve().parents[2] / ".env"
load_dotenv(DOTENV_PATH)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
MODEL = "qwen/qwen3-8b"  # Using Qwen 3 8B as requested

class Qwen3MultiHopTest:
    def __init__(self, bird_db_path: str):
        self.api_key = OPENROUTER_API_KEY
        self.api_base = OPENAI_API_BASE
        self.model = MODEL
        self.bird_db_path = Path(bird_db_path)
        self.min_time_between_requests = 1.0
        self.last_request_time = 0
        
        # Results tracking per hop length
        self.hop_results = {}
        
    def get_database_schema(self, db_name: str, target_tables: List[str]) -> str:
        """Get schema for target tables only."""
        db_path = self.bird_db_path / db_name / f"{db_name}.sqlite"
        
        if not db_path.exists():
            return f"Database {db_name} not found"
        
        schema_text = f"Database: {db_name}\n\nONLY USE THESE TABLES:\n"
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            for table_name in target_tables:
                try:
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    
                    schema_text += f"\n{table_name}:\n"
                    for col in columns:
                        pk_marker = " (PK)" if col[5] else ""
                        schema_text += f"  - {col[1]}: {col[2]}{pk_marker}\n"
                        
                except Exception:
                    schema_text += f"\n{table_name}: ERROR\n"
            
            conn.close()
            
        except Exception as e:
            schema_text += f"Database error: {str(e)}\n"
        
        return schema_text

    def create_prompt(self, question: str, schema_info: str, target_path: List[str], true_sql: str) -> str:
        """Create guided prompt for Qwen 3."""
        return f"""Generate SQL following the exact path provided.

{schema_info}

PATH: {' -> '.join(target_path)}

REFERENCE SQL:
{true_sql}

QUESTION: {question}

RULES:
1. Follow EXACT path: {' -> '.join(target_path)}
2. Use aliases t0, t1, t2, etc.
3. Copy join patterns from reference
4. SELECT * from final result

Generate SQL:"""

    def call_qwen3(self, prompt: str):
        """Call Qwen 2.5 72B API (no logprobs)."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_time_between_requests:
            time.sleep(self.min_time_between_requests - time_since_last)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 600
            # No logprobs - ignored
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                print(f"   ❌ API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"   ❌ Request failed: {str(e)}")
            return None

    def extract_tables(self, sql: str) -> List[str]:
        """Extract table sequence from SQL."""
        sql_upper = sql.upper()
        matches = re.findall(r'FROM\s+(\w+)\s+AS|JOIN\s+(\w+)\s+AS', sql_upper)
        return [t[0] if t[0] else t[1] for t in matches]

    def test_single_hop_batch(self, hop_length: int, max_queries: int = 100) -> Dict:
        """Test batch_001.json from a specific hop directory."""
        batch_file = Path(f"final_data/{hop_length}_hop/batch_001.json")
        
        if not batch_file.exists():
            print(f"❌ File not found: {batch_file}")
            return {"error": f"File not found: {batch_file}"}
        
        # Load data
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        print(f"\n🚀 Testing {hop_length}-hop: {len(batch_data)} queries available")
        print(f"   File: {batch_file}")
        print(f"   Testing: {min(max_queries, len(batch_data))} queries")
        
        # Test queries
        test_queries = batch_data[:max_queries]
        
        results = {
            "hop_length": hop_length,
            "total_queries": len(test_queries),
            "successful": 0,
            "failed": 0,
            "exact_matches": 0,
            "path_matches": 0,
            "start_time": time.time(),
            "details": []
        }
        
        for i, query_data in enumerate(test_queries):
            db_name = query_data["db_id"]
            question = query_data["natural_query"]
            true_sql = query_data["sql"]
            true_path = query_data["path"]
            
            # Get schema
            schema_info = self.get_database_schema(db_name, true_path)
            
            # Create prompt
            prompt = self.create_prompt(question, schema_info, true_path, true_sql)
            
            # Call API
            response = self.call_qwen3(prompt)
            
            if response:
                generated_sql = response['choices'][0]['message']['content']
                
                # Clean SQL
                if "```" in generated_sql:
                    generated_sql = generated_sql.split("```")[1]
                    if generated_sql.startswith("sql"):
                        generated_sql = generated_sql[3:]
                    generated_sql = generated_sql.strip()
                
                # Evaluate
                gen_clean = generated_sql.strip().replace('\n', ' ').replace('  ', ' ').upper()
                true_clean = true_sql.strip().replace('\n', ' ').replace('  ', ' ').upper()
                
                exact_match = gen_clean == true_clean
                
                gen_tables = self.extract_tables(generated_sql)
                expected_tables = [t.upper() for t in true_path]
                path_match = gen_tables == expected_tables
                
                results["successful"] += 1
                if exact_match:
                    results["exact_matches"] += 1
                if path_match:
                    results["path_matches"] += 1
                
                # Store details
                results["details"].append({
                    "query_idx": i,
                    "db_id": db_name,
                    "exact_match": exact_match,
                    "path_match": path_match,
                    "provider": response.get('provider', 'Unknown'),
                    "usage": response.get('usage', {})
                })
                
                status = "✅" if exact_match else "❌"
                exact_rate = (results["exact_matches"] / results["successful"]) * 100
                path_rate = (results["path_matches"] / results["successful"]) * 100
                
                print(f"   [{i+1:3d}/{len(test_queries)}] {status} {db_name:15s} | "
                      f"Exact: {exact_rate:5.1f}% | Path: {path_rate:5.1f}%", end='\r')
            else:
                results["failed"] += 1
                results["details"].append({
                    "query_idx": i,
                    "db_id": db_name,
                    "status": "failed"
                })
        
        results["end_time"] = time.time()
        results["duration"] = results["end_time"] - results["start_time"]
        
        print()  # New line after progress
        return results

    def run_all_hops(self, max_queries_per_hop: int = 100):
        """Run tests on all hop lengths."""
        hop_lengths = [5, 10, 15, 20]
        
        print(f"🧪 Qwen 3 8B Multi-Hop Test")
        print(f"🤖 Model: {MODEL}")
        print(f"📊 Testing {max_queries_per_hop} queries per hop length")
        print(f"📋 Hop lengths: {', '.join(map(str, hop_lengths))}")
        print("=" * 80)
        
        overall_start = time.time()
        
        for hop_length in hop_lengths:
            results = self.test_single_hop_batch(hop_length, max_queries_per_hop)
            self.hop_results[hop_length] = results
        
        overall_time = time.time() - overall_start
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"📊 QWEN 3 8B MULTI-HOP RESULTS")
        print(f"{'='*80}")
        print(f"🤖 Model: {MODEL}")
        print(f"⏱️  Total Time: {overall_time/60:.1f} minutes")
        print()
        
        # Results table
        print(f"{'Hop':>3} | {'Total':>5} | {'Success':>7} | {'Failed':>6} | {'Exact%':>6} | {'Path%':>5} | {'Time':>4}")
        print(f"{'-'*3}-+-{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*4}")
        
        total_queries = 0
        total_successful = 0
        total_exact = 0
        total_path = 0
        
        for hop_length in hop_lengths:
            if hop_length in self.hop_results:
                r = self.hop_results[hop_length]
                
                if "error" in r:
                    print(f"{hop_length:>3} | ERROR: {r['error']}")
                    continue
                
                exact_pct = (r["exact_matches"] / max(1, r["successful"])) * 100
                path_pct = (r["path_matches"] / max(1, r["successful"])) * 100
                
                print(f"{hop_length:>3} | {r['total_queries']:>5} | {r['successful']:>7} | {r['failed']:>6} | "
                      f"{exact_pct:>5.1f}% | {path_pct:>4.1f}% | {r['duration']/60:>3.1f}m")
                
                total_queries += r["total_queries"]
                total_successful += r["successful"]
                total_exact += r["exact_matches"]
                total_path += r["path_matches"]
        
        # Overall totals
        print(f"{'-'*3}-+-{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*4}")
        overall_exact_pct = (total_exact / max(1, total_successful)) * 100
        overall_path_pct = (total_path / max(1, total_successful)) * 100
        
        print(f"{'TOT':>3} | {total_queries:>5} | {total_successful:>7} | {total_queries-total_successful:>6} | "
              f"{overall_exact_pct:>5.1f}% | {overall_path_pct:>4.1f}% | {overall_time/60:>3.1f}m")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"   Total Queries:    {total_queries:,}")
        print(f"   Success Rate:     {(total_successful/total_queries)*100:.1f}%")
        print(f"   Overall Exact:    {overall_exact_pct:.1f}%")
        print(f"   Overall Path:     {overall_path_pct:.1f}%")
        print(f"   Avg Time/Query:   {overall_time/total_queries:.1f}s")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"qwen3_multihop_results_{timestamp}.json"
        
        full_results = {
            "metadata": {
                "model": MODEL,
                "timestamp": timestamp,
                "total_time_seconds": overall_time,
                "hop_lengths_tested": hop_lengths,
                "max_queries_per_hop": max_queries_per_hop,
                "total_queries": total_queries,
                "total_successful": total_successful,
                "total_exact_matches": total_exact,
                "total_path_matches": total_path,
                "overall_exact_rate": overall_exact_pct,
                "overall_path_rate": overall_path_pct
            },
            "hop_results": self.hop_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        print(f"{'='*80}")

def main():
    """Main function."""
    bird_db_path = "../bird/train/train_databases/train_databases"
    
    print("🔍 Checking batch files...")
    for hop in [5, 10, 15, 20]:
        batch_file = Path(f"final_data/{hop}_hop/batch_001.json")
        if batch_file.exists():
            print(f"   ✅ {hop}-hop: {batch_file}")
        else:
            print(f"   ❌ {hop}-hop: {batch_file} NOT FOUND")
    
    # Run tests
    tester = Qwen3MultiHopTest(bird_db_path)
    tester.run_all_hops(max_queries_per_hop=100)

if __name__ == "__main__":
    main() 