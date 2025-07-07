#!/usr/bin/env python3
"""
Bulk SQL Query Generator using Qwen 2.5 72B with Verification and Logging
Generates SQL for all batches and hop lengths without ground truth in prompts
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import glob
import re
import sqlite3
from collections import defaultdict
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    OPENROUTER_API_KEY, DEFAULT_MODEL, BIRD_DB_PATH,
    FINAL_DATA_DIR, GENERATED_QUERY_DIR, LOGS_DIR,
    get_database_description_path, get_hop_data_dir,
    validate_config
)

# Validate configuration
validate_config()

# Configuration
MODEL = DEFAULT_MODEL
HOP_LENGTHS = [5, 10, 15, 20]

class LLMAPILogger:
    """Logger class for LLM API calls and responses."""
    
    def __init__(self, log_dir: str = None):
        self.log_dir = Path(log_dir) if log_dir else LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"llm_api_log_{timestamp}.jsonl"
        
        # Initialize counters
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # Log session start
        self.log_event({
            "event": "session_start",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "log_file": str(self.log_file)
        })
    
    def log_api_call(self, prompt: str, response: Optional[Dict], api_time: float, 
                    query_info: Dict = None, error: str = None):
        """Log an API call with all relevant information."""
        self.total_calls += 1
        
        log_entry = {
            "event": "api_call",
            "timestamp": datetime.now().isoformat(),
            "call_id": self.total_calls,
            "model": MODEL,
            "api_time": api_time,
            "prompt_length": len(prompt),
            "query_info": query_info or {},
            "success": response is not None,
            "error": error
        }
        
        if response:
            self.successful_calls += 1
            usage = response.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            
            self.total_tokens += total_tokens
            
            log_entry.update({
                "response_length": len(response.get('choices', [{}])[0].get('message', {}).get('content', '')),
                "usage": usage,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            })
        else:
            self.failed_calls += 1
        
        self.log_event(log_entry)
    
    def log_event(self, event: Dict):
        """Log a general event."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
    
    def get_stats(self) -> Dict:
        """Get current logging statistics."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": (self.successful_calls / max(1, self.total_calls)) * 100,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_call": self.total_tokens / max(1, self.successful_calls)
        }
    
    def log_session_end(self):
        """Log session end with final statistics."""
        self.log_event({
            "event": "session_end",
            "timestamp": datetime.now().isoformat(),
            "final_stats": self.get_stats()
        })

class SQLVerifier:
    """Verifies generated SQL against ground truth."""
    
    def __init__(self):
        self.exact_matches = 0
        self.path_matches = 0
        self.semantic_matches = 0
        self.total_verified = 0
        
        # Store examples for analysis
        self.correct_examples = []
        self.incorrect_examples = []
        self.path_match_examples = []
        
    def extract_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL query."""
        if not sql or not sql.strip():
            return []
        
        # Clean and normalize SQL
        sql_upper = sql.upper()
        
        # Find all table references
        tables = []
        
        # Pattern for FROM clause
        from_pattern = r'FROM\s+(\w+)(?:\s+AS\s+\w+)?'
        from_matches = re.findall(from_pattern, sql_upper)
        tables.extend(from_matches)
        
        # Pattern for JOIN clauses
        join_pattern = r'(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+)?JOIN\s+(\w+)(?:\s+AS\s+\w+)?'
        join_matches = re.findall(join_pattern, sql_upper)
        tables.extend(join_matches)
        
        # Remove duplicates while preserving order
        unique_tables = []
        seen = set()
        for table in tables:
            if table not in seen:
                unique_tables.append(table)
                seen.add(table)
        
        return unique_tables
    
    def normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison."""
        if not sql:
            return ""
        
        # Remove extra whitespace and normalize
        normalized = re.sub(r'\s+', ' ', sql.strip())
        normalized = normalized.replace('\n', ' ').replace('\t', ' ')
        
        return normalized.upper()
    
    def verify_sql(self, generated_sql: str, ground_truth_sql: str, 
                  expected_path: List[str], query_info: Dict) -> Dict[str, Any]:
        """Verify generated SQL against ground truth."""
        self.total_verified += 1
        
        # Normalize SQLs
        gen_normalized = self.normalize_sql(generated_sql)
        truth_normalized = self.normalize_sql(ground_truth_sql)
        
        # Check exact match
        exact_match = gen_normalized == truth_normalized
        if exact_match:
            self.exact_matches += 1
        
        # Check path match
        gen_tables = self.extract_tables(generated_sql)
        expected_tables = [t.upper() for t in expected_path]
        path_match = gen_tables == expected_tables
        if path_match:
            self.path_matches += 1
        
        # Create verification result
        result = {
            "exact_match": exact_match,
            "path_match": path_match,
            "generated_tables": gen_tables,
            "expected_tables": expected_tables,
            "generated_sql_normalized": gen_normalized,
            "ground_truth_normalized": truth_normalized,
            "query_info": query_info
        }
        
        # Store examples
        example = {
            "db_id": query_info.get("db_id"),
            "question": query_info.get("natural_query", "")[:100] + "...",
            "generated_sql": generated_sql,
            "ground_truth_sql": ground_truth_sql,
            "expected_path": expected_path,
            "generated_tables": gen_tables,
            "result": result
        }
        
        if exact_match:
            self.correct_examples.append(example)
        elif path_match:
            self.path_match_examples.append(example)
        else:
            self.incorrect_examples.append(example)
        
        return result
    
    def get_verification_stats(self) -> Dict:
        """Get verification statistics."""
        return {
            "total_verified": self.total_verified,
            "exact_matches": self.exact_matches,
            "path_matches": self.path_matches,
            "exact_match_rate": (self.exact_matches / max(1, self.total_verified)) * 100,
            "path_match_rate": (self.path_matches / max(1, self.total_verified)) * 100,
            "correct_examples_count": len(self.correct_examples),
            "incorrect_examples_count": len(self.incorrect_examples),
            "path_match_examples_count": len(self.path_match_examples)
        }
    
    def get_example_analysis(self) -> Dict:
        """Get detailed analysis of correct/incorrect examples."""
        return {
            "correct_examples": self.correct_examples[:10],  # First 10
            "incorrect_examples": self.incorrect_examples[:10],  # First 10
            "path_match_examples": self.path_match_examples[:10],  # First 10
            "common_error_patterns": self.analyze_error_patterns()
        }
    
    def analyze_error_patterns(self) -> List[Dict]:
        """Analyze common error patterns in incorrect examples."""
        patterns = []
        
        # Group by database
        db_errors = defaultdict(int)
        for example in self.incorrect_examples:
            db_errors[example["db_id"]] += 1
        
        patterns.append({
            "type": "database_errors",
            "description": "Errors by database",
            "data": dict(sorted(db_errors.items(), key=lambda x: x[1], reverse=True)[:5])
        })
        
        # Analyze table path errors
        path_errors = defaultdict(int)
        for example in self.incorrect_examples:
            gen_tables = example["generated_tables"]
            expected_tables = example["expected_path"]
            if gen_tables != expected_tables:
                path_errors[f"Expected: {expected_tables}, Got: {gen_tables}"] += 1
        
        patterns.append({
            "type": "path_errors",
            "description": "Common path deviations",
            "data": dict(sorted(path_errors.items(), key=lambda x: x[1], reverse=True)[:5])
        })
        
        return patterns

class BulkSQLGenerator:
    def __init__(self, bird_db_path: str = None, source_dir: str = None):
        self.bird_db_path = Path(bird_db_path) if bird_db_path else Path(BIRD_DB_PATH)
        self.source_dir = Path(source_dir) if source_dir else FINAL_DATA_DIR
        self.output_dir = GENERATED_QUERY_DIR
        self.logger = LLMAPILogger()
        self.verifier = SQLVerifier()
        
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_time": 0,
            "by_hop": {}
        }
        
    def get_database_schema(self, db_name: str, target_tables: List[str]) -> str:
        """Get simplified schema for target tables only."""
        db_path = get_database_description_path(db_name)
        
        if not db_path.exists():
            return f"Schema for {db_name} not found"
        
        with open(db_path, 'r', encoding='utf-8') as f:
            full_schema = f.read()
        
        # Extract only target tables
        lines = full_schema.split('\n')
        filtered_lines = []
        current_table = None
        include_table = False
        
        for line in lines:
            if line.startswith('Table:'):
                table_name = line.split('Table:')[1].strip()
                current_table = table_name
                include_table = table_name.upper() in [t.upper() for t in target_tables]
                if include_table:
                    filtered_lines.append(line)
            elif include_table and line.strip():
                filtered_lines.append(line)
            elif not line.strip() and include_table:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def create_clean_prompt(self, question: str, schema_info: str, target_path: List[str]) -> str:
        """Create prompt WITHOUT ground truth SQL."""
        path_str = " -> ".join(target_path)
        
        return f"""You are an expert SQL developer. Generate a SQL query that answers the question using EXACTLY the specified table path.

DATABASE SCHEMA:
{schema_info}

QUESTION: {question}

REQUIRED PATH: {path_str}
(You MUST use these tables in this exact order)

Generate ONLY the SQL query without any explanations or markdown formatting."""

    def call_qwen(self, prompt: str, query_info: Dict = None) -> Dict[str, Any]:
        """Call Qwen 2.5 72B via OpenRouter API with logging."""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/yourusername/sqlmultihop",
            "X-Title": "SQL MultiHop Bulk Generation",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        start_time = time.time()
        response = None
        error = None
        
        try:
            api_response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if api_response.status_code == 200:
                response = api_response.json()
            else:
                error = f"API Error {api_response.status_code}: {api_response.text}"
                
        except Exception as e:
            error = f"Request failed: {e}"
        
        api_time = time.time() - start_time
        
        # Log the API call
        self.logger.log_api_call(prompt, response, api_time, query_info, error)
        
        return response
    
    def process_single_query(self, query_data: Dict) -> Dict[str, Any]:
        """Process a single query and generate SQL with verification."""
        db_name = query_data["db_id"]
        question = query_data["natural_query"]
        true_path = query_data["path"]
        ground_truth_sql = query_data.get("sql", "")
        
        # Get schema
        schema_info = self.get_database_schema(db_name, true_path)
        
        # Create clean prompt (without ground truth SQL)
        prompt = self.create_clean_prompt(question, schema_info, true_path)
        
        # Call API
        query_info = {
            "db_id": db_name,
            "natural_query": question,
            "path": true_path
        }
        
        response = self.call_qwen(prompt, query_info)
        
        result = {
            "db_id": db_name,
            "natural_query": question,
            "path": true_path,
            "ground_truth_sql": ground_truth_sql,
            "generated_sql": None,
            "success": False,
            "error": None,
            "usage": None,
            "verification": None
        }
        
        if response and 'choices' in response:
            generated_sql = response['choices'][0]['message']['content']
            
            # Clean SQL
            if "```" in generated_sql:
                generated_sql = generated_sql.split("```")[1]
                if generated_sql.startswith("sql"):
                    generated_sql = generated_sql[3:]
                generated_sql = generated_sql.strip()
            
            result["generated_sql"] = generated_sql
            result["success"] = True
            
            # Verify against ground truth
            if ground_truth_sql:
                verification = self.verifier.verify_sql(
                    generated_sql, ground_truth_sql, true_path, query_info
                )
                result["verification"] = verification
            
            # Store usage info
            if 'usage' in response:
                result["usage"] = response['usage']
        else:
            result["error"] = "API call failed"
        
        return result
    
    def process_batch_file(self, batch_path: Path, hop_length: int) -> Dict[str, Any]:
        """Process a single batch file."""
        print(f"Processing {batch_path.name}...")
        
        # Load batch data
        with open(batch_path, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        # Process each query
        results = []
        batch_stats = {
            "total_queries": len(batch_data),
            "successful": 0,
            "failed": 0,
            "exact_matches": 0,
            "path_matches": 0,
            "start_time": time.time()
        }
        
        for i, query_data in enumerate(batch_data):
            result = self.process_single_query(query_data)
            results.append(result)
            
            if result["success"]:
                batch_stats["successful"] += 1
                
                # Check verification results
                if result["verification"]:
                    if result["verification"]["exact_match"]:
                        batch_stats["exact_matches"] += 1
                    elif result["verification"]["path_match"]:
                        batch_stats["path_matches"] += 1
            else:
                batch_stats["failed"] += 1
            
            # Progress update
            if (i + 1) % 10 == 0:
                success_rate = (batch_stats["successful"] / (i + 1)) * 100
                exact_rate = (batch_stats["exact_matches"] / max(1, batch_stats["successful"])) * 100
                print(f"   [{i+1:3d}/{len(batch_data)}] Success: {success_rate:5.1f}% | Exact: {exact_rate:5.1f}%", end='\r')
        
        batch_stats["end_time"] = time.time()
        batch_stats["duration"] = batch_stats["end_time"] - batch_stats["start_time"]
        
        # Save results
        output_hop_dir = get_hop_data_dir(hop_length, "generated_query")
        output_hop_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_hop_dir / batch_path.name
        
        output_data = {
            "metadata": {
                "source_file": str(batch_path),
                "hop_length": hop_length,
                "model": MODEL,
                "generated_at": datetime.now().isoformat(),
                "stats": batch_stats
            },
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"   COMPLETE {batch_path.name}: {batch_stats['successful']}/{batch_stats['total_queries']} "
              f"({batch_stats['successful']/batch_stats['total_queries']*100:.1f}%) | "
              f"Exact: {batch_stats['exact_matches']} | Path: {batch_stats['path_matches']} | "
              f"Time: {batch_stats['duration']/60:.1f}m")
        
        return batch_stats
    
    def process_hop_length(self, hop_length: int):
        """Process all batches for a specific hop length."""
        hop_dir = get_hop_data_dir(hop_length, "final_data")
        
        if not hop_dir.exists():
            print(f"ERROR: Directory not found: {hop_dir}")
            return
        
        # Find all batch files
        batch_files = list(hop_dir.glob("batch_*.json"))
        batch_files.sort()
        
        print(f"\nProcessing {hop_length}-hop: {len(batch_files)} batches found")
        print(f"Source: {hop_dir}")
        print(f"Output: {get_hop_data_dir(hop_length, 'generated_query')}")
        
        hop_stats = {
            "total_batches": len(batch_files),
            "total_queries": 0,
            "successful": 0,
            "failed": 0,
            "exact_matches": 0,
            "path_matches": 0,
            "start_time": time.time()
        }
        
        for batch_file in batch_files:
            batch_stats = self.process_batch_file(batch_file, hop_length)
            
            hop_stats["total_queries"] += batch_stats["total_queries"]
            hop_stats["successful"] += batch_stats["successful"]
            hop_stats["failed"] += batch_stats["failed"]
            hop_stats["exact_matches"] += batch_stats["exact_matches"]
            hop_stats["path_matches"] += batch_stats["path_matches"]
        
        hop_stats["end_time"] = time.time()
        hop_stats["duration"] = hop_stats["end_time"] - hop_stats["start_time"]
        
        self.stats["by_hop"][hop_length] = hop_stats
        
        print(f"COMPLETE {hop_length}-hop: {hop_stats['successful']}/{hop_stats['total_queries']} "
              f"({hop_stats['successful']/hop_stats['total_queries']*100:.1f}%) | "
              f"Exact: {hop_stats['exact_matches']} | Path: {hop_stats['path_matches']} | "
              f"Time: {hop_stats['duration']/60:.1f}m")
    
    def run_bulk_generation(self):
        """Run bulk generation for all hop lengths."""
        print(f"Bulk SQL Generation with Verification")
        print(f"Model: {MODEL}")
        print(f"Processing hop lengths: {HOP_LENGTHS}")
        print(f"Output directory: {self.output_dir}")
        print(f"API logs: {self.logger.log_file}")
        print("=" * 80)
        
        overall_start = time.time()
        
        for hop_length in HOP_LENGTHS:
            self.process_hop_length(hop_length)
        
        overall_time = time.time() - overall_start
        
        # Calculate overall stats
        total_queries = sum(stats["total_queries"] for stats in self.stats["by_hop"].values())
        total_successful = sum(stats["successful"] for stats in self.stats["by_hop"].values())
        total_failed = sum(stats["failed"] for stats in self.stats["by_hop"].values())
        total_exact = sum(stats["exact_matches"] for stats in self.stats["by_hop"].values())
        total_path = sum(stats["path_matches"] for stats in self.stats["by_hop"].values())
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"BULK GENERATION SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {MODEL}")
        print(f"Total Time: {overall_time/3600:.1f} hours")
        print()
        
        # Results table
        print(f"{'Hop':>3} | {'Batches':>7} | {'Queries':>7} | {'Success':>7} | {'Exact':>5} | {'Path':>4} | {'Rate':>6} | {'Time':>6}")
        print(f"{'-'*3}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}-+-{'-'*4}-+-{'-'*6}-+-{'-'*6}")
        
        for hop_length in HOP_LENGTHS:
            if hop_length in self.stats["by_hop"]:
                s = self.stats["by_hop"][hop_length]
                success_rate = (s["successful"] / s["total_queries"]) * 100
                exact_rate = (s["exact_matches"] / max(1, s["successful"])) * 100
                path_rate = (s["path_matches"] / max(1, s["successful"])) * 100
                print(f"{hop_length:>3} | {s['total_batches']:>7} | {s['total_queries']:>7} | "
                      f"{s['successful']:>7} | {s['exact_matches']:>5} | {s['path_matches']:>4} | "
                      f"{success_rate:>5.1f}% | {s['duration']/60:>5.1f}m")
        
        # Overall totals
        print(f"{'-'*3}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}-+-{'-'*4}-+-{'-'*6}-+-{'-'*6}")
        overall_success_rate = (total_successful / total_queries) * 100
        overall_exact_rate = (total_exact / max(1, total_successful)) * 100
        overall_path_rate = (total_path / max(1, total_successful)) * 100
        
        print(f"{'TOT':>3} | {sum(s['total_batches'] for s in self.stats['by_hop'].values()):>7} | "
              f"{total_queries:>7} | {total_successful:>7} | {total_exact:>5} | {total_path:>4} | "
              f"{overall_success_rate:>5.1f}% | {overall_time/60:>5.1f}m")
        
        print(f"\nKEY METRICS:")
        print(f"   Total Queries Generated: {total_queries:,}")
        print(f"   Overall Success Rate:    {overall_success_rate:.1f}%")
        print(f"   Overall Exact Match:     {overall_exact_rate:.1f}%")
        print(f"   Overall Path Match:      {overall_path_rate:.1f}%")
        print(f"   Average Time per Query:  {overall_time/total_queries:.1f}s")
        print(f"   Total API Calls:         {total_queries:,}")
        
        # Get verification stats
        verification_stats = self.verifier.get_verification_stats()
        print(f"\nVERIFICATION RESULTS:")
        print(f"   Exact Matches:           {verification_stats['exact_matches']}")
        print(f"   Path Matches:            {verification_stats['path_matches']}")
        print(f"   Exact Match Rate:        {verification_stats['exact_match_rate']:.1f}%")
        print(f"   Path Match Rate:         {verification_stats['path_match_rate']:.1f}%")
        
        # Get API logger stats
        api_stats = self.logger.get_stats()
        print(f"\nAPI STATISTICS:")
        print(f"   Total API Calls:         {api_stats['total_calls']}")
        print(f"   Successful Calls:        {api_stats['successful_calls']}")
        print(f"   Failed Calls:            {api_stats['failed_calls']}")
        print(f"   API Success Rate:        {api_stats['success_rate']:.1f}%")
        print(f"   Total Tokens Used:       {api_stats['total_tokens']:,}")
        print(f"   Avg Tokens per Call:     {api_stats['avg_tokens_per_call']:.1f}")
        
        # Save comprehensive summary
        summary_file = self.output_dir / f"generation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        example_analysis = self.verifier.get_example_analysis()
        
        summary_data = {
            "metadata": {
                "model": MODEL,
                "hop_lengths": HOP_LENGTHS,
                "total_time_seconds": overall_time,
                "generated_at": datetime.now().isoformat()
            },
            "stats": {
                "total_queries": total_queries,
                "total_successful": total_successful,
                "total_failed": total_failed,
                "total_exact_matches": total_exact,
                "total_path_matches": total_path,
                "overall_success_rate": overall_success_rate,
                "overall_exact_rate": overall_exact_rate,
                "overall_path_rate": overall_path_rate,
                "by_hop": self.stats["by_hop"]
            },
            "verification": verification_stats,
            "api_stats": api_stats,
            "example_analysis": example_analysis
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Log session end
        self.logger.log_session_end()
        
        print(f"\nSummary saved to: {summary_file}")
        print(f"API logs saved to: {self.logger.log_file}")
        print(f"Generated queries in: {self.output_dir}")
        print(f"{'='*80}")

def main():
    """Main function."""
    # Check if source directories exist
    print("Checking source directories...")
    for hop in HOP_LENGTHS:
        source_dir = get_hop_data_dir(hop, "final_data")
        if source_dir.exists():
            batch_count = len(list(source_dir.glob("batch_*.json")))
            print(f"   OK {hop}-hop: {batch_count} batches in {source_dir}")
        else:
            print(f"   ERROR {hop}-hop: {source_dir} NOT FOUND")
    
    print()
    
    # Run bulk generation
    generator = BulkSQLGenerator()
    generator.run_bulk_generation()

if __name__ == "__main__":
    main() 