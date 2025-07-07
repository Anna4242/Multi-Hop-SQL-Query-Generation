#!/usr/bin/env python3
"""
Batch Path-Guided SQL Generation Evaluator
Tests 100+ queries and calculates comprehensive performance metrics
"""

import json
import os
import sqlite3
import pathlib
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import requests
from datetime import datetime

# Load environment variables
DOTENV_PATH = pathlib.Path(__file__).resolve().parents[2] / ".env"
load_dotenv(DOTENV_PATH)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

MODEL = "qwen/qwen-2.5-72b-instruct"

class BatchPathEvaluator:
    """Evaluate LLM performance on multiple queries with path guidance."""
    
    def __init__(self, bird_db_path: str):
        self.api_key = OPENROUTER_API_KEY
        self.api_base = OPENAI_API_BASE
        self.model = MODEL
        self.bird_db_path = Path(bird_db_path)
        self.min_time_between_requests = 1.1  # Slightly longer for batch processing
        self.last_request_time = 0
        
        # Performance tracking
        self.results = {
            "total_queries": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "exact_matches": 0,
            "path_matches": 0,
            "join_count_matches": 0,
            "evaluations": []
        }
    
    def get_database_schema(self, db_name: str, target_tables: List[str]) -> str:
        """Get schema info for target tables only."""
        db_path = self.bird_db_path / db_name / f"{db_name}.sqlite"
        
        if not db_path.exists():
            return f"Database {db_name} not found"
        
        schema_text = f"Database: {db_name}\n\n"
        schema_text += "ONLY USE THESE TABLES AND COLUMNS:\n"
        schema_text += "=" * 50 + "\n"
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            for table_name in target_tables:
                try:
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    
                    schema_text += f"\n{table_name}:\n"
                    for col in columns:
                        pk_marker = " (PRIMARY KEY)" if col[5] else ""
                        schema_text += f"  - {col[1]}: {col[2]}{pk_marker}\n"
                        
                except Exception as e:
                    schema_text += f"\n{table_name}: ERROR - {str(e)}\n"
            
            conn.close()
            
        except Exception as e:
            schema_text += f"Error accessing database: {str(e)}\n"
        
        return schema_text
    
    def create_enhanced_prompt(self, question: str, schema_info: str, target_path: List[str], true_sql: str) -> str:
        """Create enhanced prompt with all guidance."""
        prompt = f"""You are an expert SQL query generator. Generate SQL following the exact path provided.

{schema_info}

REQUIRED PATH: {' -> '.join(target_path)}

GROUND TRUTH REFERENCE:
{true_sql}

QUESTION: {question}

INSTRUCTIONS:
1. Follow EXACT path: {' -> '.join(target_path)}
2. Use table aliases t0, t1, t2, etc. in path order
3. Study the ground truth SQL for join patterns
4. Use appropriate JOIN types (INNER, LEFT) as shown
5. SELECT * from final result
6. No WHERE clauses unless in ground truth

Generate SQL following the exact path:"""
        
        return prompt
    
    def call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM API with rate limiting."""
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
            "max_tokens": 800
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
                print(f"❌ API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
            return None
    
    def extract_table_sequence(self, sql: str) -> List[str]:
        """Extract table sequence from SQL."""
        sql_upper = sql.upper()
        matches = re.findall(r'FROM\s+(\w+)\s+AS|JOIN\s+(\w+)\s+AS', sql_upper)
        return [t[0] if t[0] else t[1] for t in matches]
    
    def evaluate_single_query(self, query_data: Dict, query_idx: int) -> Dict:
        """Evaluate a single query."""
        db_name = query_data["db_id"]
        question = query_data["natural_query"]
        true_sql = query_data["sql"]
        true_path = query_data["path"]
        
        # Get schema
        schema_info = self.get_database_schema(db_name, true_path)
        
        # Create prompt
        prompt = self.create_enhanced_prompt(question, schema_info, true_path, true_sql)
        
        # Generate SQL
        generated_sql = self.call_llm(prompt)
        
        if not generated_sql:
            self.results["failed_generations"] += 1
            return {
                "query_idx": query_idx,
                "db_id": db_name,
                "status": "failed",
                "error": "Generation failed"
            }
        
        # Clean generated SQL
        if "```" in generated_sql:
            generated_sql = generated_sql.split("```")[1]
            if generated_sql.startswith("sql"):
                generated_sql = generated_sql[3:]
            generated_sql = generated_sql.strip()
        
        # Evaluate
        evaluation = self.evaluate_sql_match(generated_sql, true_sql, true_path)
        
        # Update counters
        self.results["successful_generations"] += 1
        if evaluation["exact_match"]:
            self.results["exact_matches"] += 1
        if evaluation["path_match"]:
            self.results["path_matches"] += 1
        if evaluation["join_count_match"]:
            self.results["join_count_matches"] += 1
        
        return {
            "query_idx": query_idx,
            "db_id": db_name,
            "question": question[:100] + "..." if len(question) > 100 else question,
            "true_path": true_path,
            "generated_sql": generated_sql,
            "true_sql": true_sql,
            "evaluation": evaluation,
            "status": "success"
        }
    
    def evaluate_sql_match(self, generated_sql: str, true_sql: str, true_path: List[str]) -> Dict:
        """Evaluate how well generated SQL matches ground truth."""
        # Normalize for comparison
        gen_clean = generated_sql.strip().replace('\n', ' ').replace('  ', ' ').upper()
        true_clean = true_sql.strip().replace('\n', ' ').replace('  ', ' ').upper()
        
        # Extract table sequences
        gen_tables = self.extract_table_sequence(generated_sql)
        expected_tables = [t.upper() for t in true_path]
        
        # Count joins
        gen_joins = len(re.findall(r'JOIN', generated_sql.upper()))
        true_joins = len(re.findall(r'JOIN', true_sql.upper()))
        
        # Calculate metrics
        exact_match = gen_clean == true_clean
        path_match = gen_tables == expected_tables
        join_count_match = gen_joins == true_joins
        
        return {
            "exact_match": exact_match,
            "path_match": path_match,
            "join_count_match": join_count_match,
            "generated_tables": gen_tables,
            "expected_tables": expected_tables,
            "generated_joins": gen_joins,
            "expected_joins": true_joins
        }
    
    def print_progress(self, current: int, total: int, current_result: Dict):
        """Print progress update."""
        exact_rate = (self.results["exact_matches"] / max(1, self.results["successful_generations"])) * 100
        path_rate = (self.results["path_matches"] / max(1, self.results["successful_generations"])) * 100
        
        status = "✅" if current_result.get("evaluation", {}).get("exact_match") else "❌"
        
        print(f"[{current:3d}/{total}] {status} {current_result.get('db_id', 'unknown'):15s} | "
              f"Exact: {exact_rate:5.1f}% | Path: {path_rate:5.1f}%", end='\r')
    
    def evaluate_batch(self, batch_data: List[Dict], max_queries: int = 100) -> Dict:
        """Evaluate multiple queries."""
        print(f"🚀 Batch Path-Guided Evaluation")
        print(f"📊 Testing {min(max_queries, len(batch_data))} queries...")
        print(f"🤖 Model: {MODEL}")
        print("=" * 80)
        
        test_queries = batch_data[:max_queries]
        self.results["total_queries"] = len(test_queries)
        
        start_time = time.time()
        
        for i, query_data in enumerate(test_queries):
            result = self.evaluate_single_query(query_data, i)
            self.results["evaluations"].append(result)
            
            self.print_progress(i + 1, len(test_queries), result)
            
            # Estimate time remaining
            if i > 0 and i % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(test_queries) - i - 1) / rate
                print(f"\n   ⏱️  ETA: {remaining/60:.1f} minutes")
        
        total_time = time.time() - start_time
        
        # Calculate final metrics
        successful = self.results["successful_generations"]
        self.results["exact_match_rate"] = (self.results["exact_matches"] / max(1, successful)) * 100
        self.results["path_match_rate"] = (self.results["path_matches"] / max(1, successful)) * 100
        self.results["join_match_rate"] = (self.results["join_count_matches"] / max(1, successful)) * 100
        self.results["success_rate"] = (successful / self.results["total_queries"]) * 100
        self.results["total_time"] = total_time
        
        return self.results
    
    def print_summary(self):
        """Print comprehensive performance summary."""
        print(f"\n\n{'='*80}")
        print(f"📊 BATCH EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"🤖 Model: {MODEL}")
        print(f"📋 Total Queries: {self.results['total_queries']}")
        print(f"⏱️  Total Time: {self.results['total_time']/60:.1f} minutes")
        print(f"🚀 Success Rate: {self.results['success_rate']:.1f}%")
        print(f"")
        print(f"🎯 ACCURACY METRICS:")
        print(f"   Exact Matches:     {self.results['exact_matches']:3d}/{self.results['successful_generations']} ({self.results['exact_match_rate']:5.1f}%)")
        print(f"   Path Following:    {self.results['path_matches']:3d}/{self.results['successful_generations']} ({self.results['path_match_rate']:5.1f}%)")
        print(f"   JOIN Count Match:  {self.results['join_count_matches']:3d}/{self.results['successful_generations']} ({self.results['join_match_rate']:5.1f}%)")
        print(f"   Failed Generations: {self.results['failed_generations']}")
        print(f"")
        
        # Database performance breakdown
        db_stats = {}
        for eval_result in self.results["evaluations"]:
            if eval_result["status"] == "success":
                db = eval_result["db_id"]
                if db not in db_stats:
                    db_stats[db] = {"total": 0, "exact": 0, "path": 0}
                
                db_stats[db]["total"] += 1
                if eval_result["evaluation"]["exact_match"]:
                    db_stats[db]["exact"] += 1
                if eval_result["evaluation"]["path_match"]:
                    db_stats[db]["path"] += 1
        
        print(f"📈 TOP PERFORMING DATABASES:")
        sorted_dbs = sorted(db_stats.items(), key=lambda x: x[1]["exact"]/max(1, x[1]["total"]), reverse=True)
        for i, (db, stats) in enumerate(sorted_dbs[:10]):
            exact_rate = (stats["exact"] / stats["total"]) * 100
            path_rate = (stats["path"] / stats["total"]) * 100
            print(f"   {i+1:2d}. {db:20s}: {exact_rate:5.1f}% exact, {path_rate:5.1f}% path ({stats['total']} queries)")
        
        print(f"{'='*80}")

def main():
    """Main evaluation function."""
    print("🎯 Batch Path-Guided SQL Generation Evaluator")
    print("=" * 60)
    
    bird_db_path = "../bird/train/train_databases/train_databases"
    batch_file = Path("final_data/5_hop/batch_001.json")
    
    if not batch_file.exists():
        print(f"❌ Batch file not found: {batch_file}")
        return
    
    evaluator = BatchPathEvaluator(bird_db_path)
    
    # Load batch data
    with open(batch_file, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
    
    print(f"📁 Database Path: {bird_db_path}")
    print(f"📋 Batch File: {batch_file}")
    print(f"📊 Available Queries: {len(batch_data)}")
    
    # Run evaluation on 100 queries
    results = evaluator.evaluate_batch(batch_data, max_queries=100)
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"batch_evaluation_results_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main() 