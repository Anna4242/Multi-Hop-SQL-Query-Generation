#!/usr/bin/env python3
"""
Qwen 2.5 72B Batch Test - 100 queries with path guidance
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
MODEL = "qwen/qwen-2.5-72b-instruct"

class QwenBatchTest:
    def __init__(self, bird_db_path: str):
        self.api_key = OPENROUTER_API_KEY
        self.api_base = OPENAI_API_BASE
        self.model = MODEL
        self.bird_db_path = Path(bird_db_path)
        self.min_time_between_requests = 1.0
        self.last_request_time = 0
        
        # Results tracking
        self.results = []
        self.total_queries = 0
        self.successful_generations = 0
        self.exact_matches = 0
        self.path_matches = 0
        self.generation_failures = 0
        
        # Performance metrics
        self.start_time = None
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
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
        """Create guided prompt for Qwen."""
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
    
    def call_llm(self, prompt: str, query_idx: int) -> Optional[Dict]:
        """Call LLM API and return full response info."""
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
            "max_tokens": 600,
            "top_logprobs": 5  # Won't work but won't break
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
                
                # Track token usage
                if 'usage' in result:
                    usage = result['usage']
                    self.token_usage["prompt_tokens"] += usage.get('prompt_tokens', 0)
                    self.token_usage["completion_tokens"] += usage.get('completion_tokens', 0)
                    self.token_usage["total_tokens"] += usage.get('total_tokens', 0)
                
                return result
            else:
                print(f"   ❌ API error for query {query_idx}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"   ❌ Request failed for query {query_idx}: {str(e)}")
            return None
    
    def extract_tables(self, sql: str) -> List[str]:
        """Extract table sequence from SQL."""
        sql_upper = sql.upper()
        matches = re.findall(r'FROM\s+(\w+)\s+AS|JOIN\s+(\w+)\s+AS', sql_upper)
        return [t[0] if t[0] else t[1] for t in matches]
    
    def evaluate_query(self, query_data: Dict, idx: int) -> Dict:
        """Evaluate single query with Qwen."""
        db_name = query_data["db_id"]
        question = query_data["natural_query"]
        true_sql = query_data["sql"]
        true_path = query_data["path"]
        
        # Get schema
        schema_info = self.get_database_schema(db_name, true_path)
        
        # Create prompt
        prompt = self.create_prompt(question, schema_info, true_path, true_sql)
        
        # Generate
        llm_response = self.call_llm(prompt, idx)
        
        if not llm_response:
            self.generation_failures += 1
            return {"status": "failed", "error": "API call failed"}
        
        # Extract generated SQL
        generated_sql = llm_response['choices'][0]['message']['content']
        
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
        
        # Update counters
        self.successful_generations += 1
        if exact_match:
            self.exact_matches += 1
        if path_match:
            self.path_matches += 1
        
        # Create result
        result = {
            "status": "success",
            "query_idx": idx,
            "db_id": db_name,
            "exact_match": exact_match,
            "path_match": path_match,
            "generated_sql": generated_sql[:100] + "..." if len(generated_sql) > 100 else generated_sql,
            "provider": llm_response.get('provider', 'Unknown'),
            "token_usage": llm_response.get('usage', {})
        }
        
        return result
    
    def print_progress(self, current: int, total: int, current_result: Dict):
        """Print progress update."""
        if self.successful_generations > 0:
            exact_rate = (self.exact_matches / self.successful_generations) * 100
            path_rate = (self.path_matches / self.successful_generations) * 100
        else:
            exact_rate = 0
            path_rate = 0
        
        status = "✅" if current_result.get("exact_match") else "❌" if current_result.get("status") == "success" else "⚠️"
        
        print(f"[{current:3d}/{total}] {status} {current_result.get('db_id', 'unknown'):15s} | "
              f"Exact: {exact_rate:5.1f}% | Path: {path_rate:5.1f}% | "
              f"Tokens: {self.token_usage['total_tokens']:,}", end='\r')
    
    def run_batch(self, batch_data: List[Dict], num_queries: int = 100):
        """Run batch evaluation with Qwen."""
        print(f"🚀 Qwen 2.5 72B Batch Test: {num_queries} queries")
        print(f"🤖 Model: {MODEL}")
        print("=" * 80)
        
        test_queries = batch_data[:num_queries]
        self.total_queries = len(test_queries)
        self.start_time = time.time()
        
        for i, query_data in enumerate(test_queries):
            result = self.evaluate_query(query_data, i)
            self.results.append(result)
            
            # Progress update every 10 queries
            if i % 10 == 0:
                self.print_progress(i + 1, len(test_queries), result)
        
        total_time = time.time() - self.start_time
        
        # Final results
        print(f"\n\n{'='*80}")
        print(f"📊 QWEN 2.5 72B BATCH RESULTS")
        print(f"{'='*80}")
        print(f"🤖 Model: {MODEL}")
        print(f"📋 Total Queries: {self.total_queries}")
        print(f"✅ Successful: {self.successful_generations}")
        print(f"❌ Failed: {self.generation_failures}")
        print(f"")
        print(f"🎯 ACCURACY METRICS:")
        if self.successful_generations > 0:
            print(f"   Exact Matches:     {self.exact_matches}/{self.successful_generations} ({(self.exact_matches/self.successful_generations)*100:.1f}%)")
            print(f"   Path Following:    {self.path_matches}/{self.successful_generations} ({(self.path_matches/self.successful_generations)*100:.1f}%)")
        else:
            print(f"   No successful generations to analyze")
        
        print(f"")
        print(f"📊 TOKEN USAGE:")
        print(f"   Prompt Tokens:     {self.token_usage['prompt_tokens']:,}")
        print(f"   Completion Tokens: {self.token_usage['completion_tokens']:,}")
        print(f"   Total Tokens:      {self.token_usage['total_tokens']:,}")
        
        cost_estimate = (self.token_usage['total_tokens'] / 1000000) * 1.2  # $1.2 per 1M tokens
        print(f"   Estimated Cost:    ${cost_estimate:.4f}")
        
        print(f"")
        print(f"⏱️  PERFORMANCE:")
        print(f"   Total Time:        {total_time/60:.1f} minutes")
        print(f"   Rate:              {self.successful_generations/total_time:.1f} queries/second")
        print(f"   Avg Time/Query:    {total_time/self.total_queries:.1f} seconds")
        
        print(f"{'='*80}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"qwen_batch_results_{timestamp}.json"
        
        full_results = {
            "metadata": {
                "model": MODEL,
                "timestamp": timestamp,
                "total_queries": self.total_queries,
                "successful_generations": self.successful_generations,
                "exact_matches": self.exact_matches,
                "path_matches": self.path_matches,
                "generation_failures": self.generation_failures,
                "total_time_seconds": total_time,
                "token_usage": self.token_usage
            },
            "results": self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Detailed results saved to: {output_file}")
        return full_results

def main():
    """Main function."""
    bird_db_path = "../bird/train/train_databases/train_databases"
    batch_file = Path("final_data/5_hop/batch_001.json")
    
    if not batch_file.exists():
        print(f"❌ File not found: {batch_file}")
        return
    
    # Load data
    with open(batch_file, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
    
    print(f"📋 Loaded {len(batch_data)} queries from {batch_file}")
    
    # Run evaluation
    evaluator = QwenBatchTest(bird_db_path)
    evaluator.run_batch(batch_data, num_queries=100)

if __name__ == "__main__":
    main() 