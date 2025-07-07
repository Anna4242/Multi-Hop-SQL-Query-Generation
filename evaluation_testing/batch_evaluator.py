#!/usr/bin/env python3
"""
Batch evaluator for testing 100 queries with path guidance
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

class BatchEvaluator:
    def __init__(self, bird_db_path: str):
        self.api_key = OPENROUTER_API_KEY
        self.api_base = OPENAI_API_BASE
        self.model = MODEL
        self.bird_db_path = Path(bird_db_path)
        self.min_time_between_requests = 1.0
        self.last_request_time = 0
        
        # Results tracking
        self.total_queries = 0
        self.successful_generations = 0
        self.exact_matches = 0
        self.path_matches = 0
    
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
        """Create guided prompt."""
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
4. SELECT * 

Generate SQL:"""
    
    def call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM API."""
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
                return result['choices'][0]['message']['content'].strip()
            else:
                return None
                
        except Exception:
            return None
    
    def extract_tables(self, sql: str) -> List[str]:
        """Extract table sequence from SQL."""
        sql_upper = sql.upper()
        matches = re.findall(r'FROM\s+(\w+)\s+AS|JOIN\s+(\w+)\s+AS', sql_upper)
        return [t[0] if t[0] else t[1] for t in matches]
    
    def evaluate_query(self, query_data: Dict, idx: int) -> Dict:
        """Evaluate single query."""
        db_name = query_data["db_id"]
        question = query_data["natural_query"]
        true_sql = query_data["sql"]
        true_path = query_data["path"]
        
        # Get schema
        schema_info = self.get_database_schema(db_name, true_path)
        
        # Create prompt
        prompt = self.create_prompt(question, schema_info, true_path, true_sql)
        
        # Generate
        generated_sql = self.call_llm(prompt)
        
        if not generated_sql:
            return {"status": "failed"}
        
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
        
        self.successful_generations += 1
        if exact_match:
            self.exact_matches += 1
        if path_match:
            self.path_matches += 1
        
        return {
            "status": "success",
            "exact_match": exact_match,
            "path_match": path_match,
            "db_id": db_name,
            "generated_sql": generated_sql[:100] + "..." if len(generated_sql) > 100 else generated_sql
        }
    
    def run_batch(self, batch_data: List[Dict], num_queries: int = 100):
        """Run batch evaluation."""
        print(f"🚀 Batch Evaluation: {num_queries} queries")
        print(f"🤖 Model: {MODEL}")
        print("=" * 60)
        
        test_queries = batch_data[:num_queries]
        self.total_queries = len(test_queries)
        
        start_time = time.time()
        
        for i, query_data in enumerate(test_queries):
            result = self.evaluate_query(query_data, i)
            
            # Progress update
            if i % 10 == 0 or result["status"] == "failed":
                exact_rate = (self.exact_matches / max(1, self.successful_generations)) * 100
                path_rate = (self.path_matches / max(1, self.successful_generations)) * 100
                
                status = "✅" if result.get("exact_match") else "❌" if result["status"] == "success" else "⚠️"
                
                print(f"[{i+1:3d}/{len(test_queries)}] {status} | "
                      f"Exact: {exact_rate:5.1f}% | Path: {path_rate:5.1f}%")
        
        total_time = time.time() - start_time
        
        # Final results
        print(f"\n{'='*60}")
        print(f"📊 FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Total Queries: {self.total_queries}")
        print(f"Successful: {self.successful_generations}")
        print(f"Failed: {self.total_queries - self.successful_generations}")
        print(f"")
        print(f"🎯 ACCURACY:")
        print(f"Exact Matches: {self.exact_matches}/{self.successful_generations} ({(self.exact_matches/max(1,self.successful_generations))*100:.1f}%)")
        print(f"Path Matches:  {self.path_matches}/{self.successful_generations} ({(self.path_matches/max(1,self.successful_generations))*100:.1f}%)")
        print(f"")
        print(f"⏱️ Time: {total_time/60:.1f} minutes")
        print(f"Rate: {self.successful_generations/total_time:.1f} queries/second")

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
    evaluator = BatchEvaluator(bird_db_path)
    evaluator.run_batch(batch_data, num_queries=100)

if __name__ == "__main__":
    main() 