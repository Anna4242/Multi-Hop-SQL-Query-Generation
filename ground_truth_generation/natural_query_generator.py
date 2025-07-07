#!/usr/bin/env python3
"""
LLM-Based Natural Query Generator with Token Tracking
Generates natural language queries for SQL batches using OpenRouter API

Quick test: python llm_natural_query_generator.py --test10
"""

import json
import os
import pathlib
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from datetime import datetime

# ----------------------------------------------------------------------------- 
# Load environment variables from .env
# ----------------------------------------------------------------------------- 
DOTENV_PATH = pathlib.Path(__file__).resolve().parents[2] / ".env"
load_dotenv(DOTENV_PATH)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing environment variable: OPENROUTER_API_KEY")
if not OPENAI_API_BASE:
    raise RuntimeError("Missing environment variable: OPENAI_API_BASE")

# Model to use for generation - Using Qwen 72B as requested
MODEL = "qwen/qwen-2.5-72b-instruct"

class TokenTracker:
    """Track token usage and costs across API calls."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        
        # Cost estimates for Qwen 2.5 72B Instruct - $1.2 per million tokens
        self.cost_per_million_tokens = 1.2  # $1.2 per 1M tokens
    
    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        """Add token usage from a successful API call."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)
        self.total_requests += 1
    
    def add_failed_request(self):
        """Record a failed API call."""
        self.failed_requests += 1
    
    @property
    def estimated_cost(self) -> float:
        """Calculate estimated cost based on token usage."""
        return (self.total_tokens / 1_000_000) * self.cost_per_million_tokens
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict:
        """Get a summary of token usage and stats."""
        elapsed = self.elapsed_time
        return {
            'model': self.model_name,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.total_requests / (self.total_requests + self.failed_requests)) * 100 if (self.total_requests + self.failed_requests) > 0 else 0,
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.total_prompt_tokens,
            'completion_tokens': self.total_completion_tokens,
            'estimated_cost_usd': round(self.estimated_cost, 4),
            'elapsed_time_seconds': round(elapsed, 1),
            'tokens_per_second': round(self.total_tokens / elapsed, 2) if elapsed > 0 else 0,
            'requests_per_minute': round((self.total_requests / elapsed) * 60, 2) if elapsed > 0 else 0,
            'avg_tokens_per_request': round(self.total_tokens / self.total_requests, 1) if self.total_requests > 0 else 0
        }
    
    def print_summary(self):
        """Print a formatted summary of token usage."""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print(f"🔢 TOKEN USAGE SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {summary['model']}")
        print(f"Total Requests: {summary['total_requests']:,} (Success: {summary['success_rate']:.1f}%)")
        print(f"Failed Requests: {summary['failed_requests']:,}")
        print(f"")
        print(f"💰 TOKEN BREAKDOWN:")
        print(f"  Prompt Tokens:     {summary['prompt_tokens']:,}")
        print(f"  Completion Tokens: {summary['completion_tokens']:,}")
        print(f"  Total Tokens:      {summary['total_tokens']:,}")
        print(f"  Estimated Cost:    ${summary['estimated_cost_usd']:.4f}")
        print(f"")
        print(f"⏱️  PERFORMANCE:")
        print(f"  Elapsed Time:      {summary['elapsed_time_seconds']}s")
        print(f"  Tokens/Second:     {summary['tokens_per_second']}")
        print(f"  Requests/Minute:   {summary['requests_per_minute']}")
        print(f"  Avg Tokens/Req:    {summary['avg_tokens_per_request']}")
        print(f"{'='*60}")


class LLMNaturalQueryGenerator:
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.api_base = OPENAI_API_BASE
        self.model = MODEL
        self.sql_dir = Path("sql_queries")
        self.output_dir = Path("final_data")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize token tracker
        self.token_tracker = TokenTracker(self.model)
        
        # Rate limiting
        self.requests_per_minute = 60
        self.last_request_time = 0
        self.min_time_between_requests = 60.0 / self.requests_per_minute
    
    def extract_join_columns(self, sql: str) -> List[tuple]:
        """Extract join columns from SQL to provide context."""
        joins = []
        # Pattern to match JOIN clauses
        join_pattern = r'(LEFT|INNER)\s+JOIN\s+(\w+)\s+AS\s+t(\d+)\s+ON\s+t(\d+)\.(\w+)\s+=\s+t(\d+)\.(\w+)'
        
        for match in re.finditer(join_pattern, sql):
            left_col = match.group(5)
            right_col = match.group(7)
            joins.append((left_col, right_col))
        
        return joins
    
    def create_prompt(self, query_data: Dict) -> str:
        """Create a prompt for the LLM to generate natural language query."""
        db_id = query_data.get('db_id', '')
        sql = query_data.get('sql', '')
        path = query_data.get('path', [])
        
        # Extract source and destination
        source = path[0] if path else 'unknown'
        dest = path[-1] if len(path) > 1 else source
        
        # Extract join information
        joins = self.extract_join_columns(sql)
        join_info = ""
        if joins:
            join_info = f"\nJoin columns: {', '.join([f'{j[0]}={j[1]}' for j in joins])}"
        
        prompt = f""" You are an expert at writing concise, natural multi‑step questions for SQL databases.
    I will give you a description of a single multi‑hop chain of subtasks. Each subtask corresponds
    to a SQL query on a certain table, in order from source → destination. Your job is to produce
    exactly one short, clear overarching question that implicitly requires performing each subtask
    in sequence to arrive at the final answer—without explicitly listing each sub‑question or includig it  in the question.
        Here is a concrete example:

    Database ID: language_corpus
    Source table: langs_words
    Destination table: pages
    Exact path (length 4 hops; min 3, max 6):
      langs_words → words → biwords → langs → pages

    Each step corresponds to this SQL query (in order):
      1. Table: langs_words
         SQL: SELECT wid
              FROM langs_words
              WHERE occurrences <= 10

      2. Table: words
         SQL: SELECT occurrences
              FROM words
              WHERE word = 'desena'

      3. Table: biwords
         SQL: SELECT occurrences
              FROM biwords
              WHERE w1st = 1 AND w2nd = 25

      4. Table: langs
         SQL: SELECT pages
              FROM langs
              WHERE lang = 'ca'

      5. Table: pages
         SQL: SELECT title
              FROM pages
              WHERE words < 10

    Example of the desired output:
      "What is the title of the Catalan Wikipedia page (with fewer than 10 distinct words) that
       contains a biword whose first term occurs at most 10 times in Catalan, and whose second
       term is 'desena'?"

    Notice that the overarching question:
      • Mentions "Catalan" (implied by langs.lang = 'ca'),
      • Implies checking langs_words (occurrences ≤ 10) and words ('desena'),
      • Implies joining through biwords, langs, and pages,
      • And does not explicitly enumerate "Step 1, Step 2, …" but flows as a single coherent request.

Database: {db_id}
Source table: {source}
Destination table: {dest}
Full path: {' → '.join(path)}
SQL query: {sql}{join_info}

Requirements:
1. Make it sound like a real question, not technical
2. Focus on what the user wants to achieve, not how
3. Be concise - one clear sentence
4. Don't mention tables, joins, or SQL terms
5. Make it specific to the domain 



Generate only the question, nothing else:"""
        
        return prompt
    
    def call_llm(self, prompt: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Call the LLM API with rate limiting. Returns (response, usage_info)."""
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_time_between_requests:
            time.sleep(self.min_time_between_requests - time_since_last_request)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 100
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
                
                # Extract usage information
                usage = result.get('usage', {})
                if usage:
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    self.token_tracker.add_usage(prompt_tokens, completion_tokens)
                
                return result['choices'][0]['message']['content'].strip(), usage
            else:
                print(f"❌ API error: {response.status_code} - {response.text}")
                self.token_tracker.add_failed_request()
                return None, None
                
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
            self.token_tracker.add_failed_request()
            return None, None
    
    def process_query(self, query_data: Dict) -> Dict:
        """Process a single query and add natural language question."""
        prompt = self.create_prompt(query_data)
        natural_query, usage = self.call_llm(prompt)
        
        if natural_query:
            # Clean up the response
            natural_query = natural_query.strip()
            # Remove quotes if present
            if natural_query.startswith('"') and natural_query.endswith('"'):
                natural_query = natural_query[1:-1]
            # Ensure it ends with ?
            if not natural_query.endswith('?'):
                natural_query = natural_query.rstrip('.') + '?'
            
            query_data['natural_query'] = natural_query
            
            # Add token usage info to query data
            if usage:
                query_data['token_usage'] = usage
        else:
            # Fallback to template if LLM fails
            path = query_data.get('path', [])
            source = path[0] if path else 'data'
            dest = path[-1] if len(path) > 1 else source
            query_data['natural_query'] = f"How does {source} relate to {dest}?"
        
        return query_data
    
    def print_progress_with_tokens(self, current: int, total: int):
        """Print progress including current token usage."""
        summary = self.token_tracker.get_summary()
        print(f"   Query {current}/{total} | "
              f"Tokens: {summary['total_tokens']:,} | "
              f"Cost: ${summary['estimated_cost_usd']:.4f} | "
              f"Rate: {summary['tokens_per_second']:.1f} tok/s", end='\r')
    
    def process_batch_file(self, input_file: Path, output_file: Path):
        """Process a single batch file."""
        print(f"📄 Processing {input_file.name}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        processed_queries = []
        
        for i, query in enumerate(queries):
            self.print_progress_with_tokens(i+1, len(queries))
            processed_query = self.process_query(query)
            
            # Save only essential fields
            essential_query = {
                'db_id': processed_query.get('db_id', ''),
                'sql': processed_query.get('sql', ''),
                'path': processed_query.get('path', []),
                'natural_query': processed_query.get('natural_query', '')
            }
            processed_queries.append(essential_query)
            
            # Show example
            if i == 0:
                print(f"\n   Example: {essential_query.get('natural_query', 'N/A')}")
        
        # Ensure output directory exists (including hop subdirectories)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed batch with only essential fields
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_queries, f, indent=2, ensure_ascii=False)
        
        summary = self.token_tracker.get_summary()
        print(f"\n✅ Saved to {output_file}")
        print(f"   Batch tokens: {summary['total_tokens']:,} | Cost: ${summary['estimated_cost_usd']:.4f}")
        return len(processed_queries)
    
    def test_single_batch(self, num_examples=10):
        """Test with a limited number of examples from a single batch file."""
        # Find first batch file
        test_file = None
        for hop_dir in sorted(self.sql_dir.iterdir()):
            if hop_dir.is_dir() and hop_dir.name.endswith('_hop'):
                batch_files = list(hop_dir.glob('batch_*.json'))
                if batch_files:
                    test_file = batch_files[0]
                    break
        
        if not test_file:
            print("❌ No batch files found!")
            return
        
        print(f"🧪 Testing with: {test_file}")
        print(f"📊 Processing only {num_examples} examples")
        
        # Load queries
        with open(test_file, 'r', encoding='utf-8') as f:
            all_queries = json.load(f)
        
        # Take only the first num_examples queries
        test_queries = all_queries[:num_examples]
        
        print(f"📋 Selected {len(test_queries)} queries from {len(all_queries)} total")
        
        # Process test queries
        processed_queries = []
        start_time = time.time()
        
        for i, query in enumerate(test_queries):
            print(f"\n[{i+1}/{len(test_queries)}] Processing query...")
            print(f"   Path: {' → '.join(query['path'])}")
            
            processed_query = self.process_query(query)
            processed_queries.append(processed_query)
            
            print(f"   ✅ Generated: {processed_query.get('natural_query', 'N/A')}")
            
            # Show token usage for this query
            if 'token_usage' in processed_query:
                usage = processed_query['token_usage']
                total_tokens = usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
                print(f"   🔢 Tokens: {total_tokens} (prompt: {usage.get('prompt_tokens', 0)}, completion: {usage.get('completion_tokens', 0)})")
        
        elapsed = time.time() - start_time
        
        # Save test results
        test_output_file = self.output_dir / "test_results.json"
        test_output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add summary to test results
        test_results = {
            'metadata': {
                'test_file': str(test_file),
                'timestamp': datetime.now().isoformat(),
                'token_summary': self.token_tracker.get_summary()
            },
            'queries': processed_queries
        }
        
        with open(test_output_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Test Results:")
        print(f"   Processed: {len(processed_queries)} queries")
        print(f"   Time: {elapsed:.1f} seconds")
        print(f"   Rate: {elapsed/len(processed_queries):.1f} seconds/query")
        print(f"   Saved to: {test_output_file}")
        
        # Print token summary
        self.token_tracker.print_summary()
        
        # Show all results
        print(f"\n📝 All generated questions:")
        print("=" * 80)
        for i, result in enumerate(processed_queries):
            print(f"\n{i+1}. Database: {result['db_id']}")
            print(f"   Path: {' → '.join(result['path'])}")
            print(f"   SQL: {result['sql'].split('FROM')[1].split('AS')[0].strip()}...")
            print(f"   Question: {result.get('natural_query', 'N/A')}")
            if 'token_usage' in result:
                usage = result['token_usage']
                total = usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
                print(f"   Tokens: {total}")
    
    def process_specific_hop(self):
        """Process all batch files from specific hop directories."""
        print("📂 Available hop directories:")
        print("=" * 60)
        
        # Find all hop directories
        hop_dirs = []
        for hop_dir in sorted(self.sql_dir.iterdir()):
            if hop_dir.is_dir() and hop_dir.name.endswith('_hop'):
                batch_files = list(hop_dir.glob('batch_*.json'))
                if batch_files:
                    # Count total queries in this hop directory
                    total_queries = 0
                    for batch_file in batch_files:
                        try:
                            with open(batch_file, 'r', encoding='utf-8') as f:
                                queries = json.load(f)
                                total_queries += len(queries)
                        except:
                            pass
                    
                    hop_dirs.append((hop_dir, len(batch_files), total_queries))
        
        if not hop_dirs:
            print("❌ No hop directories found!")
            return
        
        # Display available hop directories
        for i, (hop_dir, batch_count, query_count) in enumerate(hop_dirs):
            print(f"{i+1:2d}. {hop_dir.name} - {batch_count} batch files, {query_count:,} queries")
        
        print(f"\n📊 Total: {len(hop_dirs)} hop directories available")
        
        # Get user selection - support multiple selections
        print(f"\nSelect hop directories to process:")
        print("Examples:")
        print("  Single: 3")
        print("  Multiple: 2,5,10,15,20")
        print("  Range: 1-5")
        print("  Mixed: 1,3-6,10,15-20")
        print("  All: a")
        
        selection = input(f"\nEnter your choice: ").strip()
        
        # Parse selection
        selected_indices = []
        
        if selection.lower() == 'a':
            selected_indices = list(range(1, len(hop_dirs) + 1))
        else:
            try:
                # Parse comma-separated values with range support
                parts = selection.split(',')
                for part in parts:
                    part = part.strip()
                    if '-' in part:
                        # Handle range (e.g., 3-6)
                        start, end = map(int, part.split('-'))
                        selected_indices.extend(range(start, end + 1))
                    else:
                        # Handle single number
                        selected_indices.append(int(part))
                
                # Remove duplicates and sort
                selected_indices = sorted(set(selected_indices))
                
                # Validate indices
                invalid_indices = [i for i in selected_indices if not 1 <= i <= len(hop_dirs)]
                if invalid_indices:
                    print(f"❌ Invalid choices: {invalid_indices}")
                    return
                    
            except ValueError:
                print("❌ Invalid selection format")
                return
        
        if not selected_indices:
            print("❌ No directories selected")
            return
        
        # Get selected hop directories
        selected_hops = [hop_dirs[i-1] for i in selected_indices]
        
        # Show selection summary
        print(f"\n✅ Selected {len(selected_hops)} hop directories:")
        total_batches = 0
        total_queries = 0
        
        for i, (hop_dir, batch_count, query_count) in enumerate(selected_hops):
            print(f"   {i+1}. {hop_dir.name} - {batch_count} batches, {query_count:,} queries")
            total_batches += batch_count
            total_queries += query_count
        
        # Estimate cost
        estimated_tokens = total_queries * 850  # Average tokens per query
        estimated_cost = (estimated_tokens / 1_000_000) * self.token_tracker.cost_per_million_tokens
        
        print(f"\n📊 Summary:")
        print(f"   Hop directories: {len(selected_hops)}")
        print(f"   Total batch files: {total_batches}")
        print(f"   Total queries: {total_queries:,}")
        print(f"   Estimated tokens: {estimated_tokens:,}")
        print(f"   Estimated cost: ${estimated_cost:.4f}")
        
        # Confirm processing
        hop_names = [hop[0].name for hop in selected_hops]
        confirm = input(f"\nProcess all batches from {len(selected_hops)} hop directories ({', '.join(hop_names)})? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return
        
        # Collect all files to process
        all_files_to_process = []
        for hop_dir, _, _ in selected_hops:
            batch_files = sorted(hop_dir.glob('batch_*.json'))
            for batch_file in batch_files:
                relative_path = batch_file.relative_to(self.sql_dir)
                output_file = self.output_dir / relative_path
                all_files_to_process.append((batch_file, output_file, hop_dir.name))
        
        print(f"\n🚀 Processing {len(all_files_to_process)} batch files from {len(selected_hops)} hop directories...")
        
        processed_queries = 0
        start_time = time.time()
        current_hop = None
        
        # Process files sequentially (to respect rate limits)
        for i, (input_file, output_file, hop_name) in enumerate(all_files_to_process):
            # Show hop directory change
            if current_hop != hop_name:
                current_hop = hop_name
                print(f"\n📂 Now processing: {hop_name}")
            
            print(f"[{i+1}/{len(all_files_to_process)}] {hop_name}/{input_file.name}")
            count = self.process_batch_file(input_file, output_file)
            processed_queries += count
            
            # Estimate time remaining
            elapsed = time.time() - start_time
            if i > 0:  # Avoid division by zero
                rate = (i + 1) / elapsed
                remaining = (len(all_files_to_process) - i - 1) / rate
                print(f"⏱️  ETA: {remaining/60:.1f} minutes")
            
            # Show running token totals
            summary = self.token_tracker.get_summary()
            print(f"🔢 Running totals: {summary['total_tokens']:,} tokens | ${summary['estimated_cost_usd']:.4f}")
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 COMPLETED {len(selected_hops)} HOP DIRECTORIES!")
        print(f"   Hop directories: {', '.join(hop_names)}")
        print(f"   Total batch files: {len(all_files_to_process)}")
        print(f"   Total queries: {processed_queries:,}")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Average rate: {processed_queries/total_time:.1f} queries/second")
        
        # Final token summary
        self.token_tracker.print_summary()
        
        # Save final summary in the main output directory
        hop_names_str = "_".join(hop_names)
        summary_file = self.output_dir / f"multi_hop_processing_{hop_names_str}_summary.json"
        final_summary = {
            'processing_complete': True,
            'processing_type': 'multiple_hops',
            'hop_directories': hop_names,
            'timestamp': datetime.now().isoformat(),
            'hop_directories_count': len(selected_hops),
            'files_processed': len(all_files_to_process),
            'queries_processed': processed_queries,
            'total_time_seconds': total_time,
            'token_summary': self.token_tracker.get_summary(),
            'selected_indices': selected_indices
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Summary saved to: {summary_file}")
    
    def list_available_batches(self):
        """List all available batch files for user selection."""
        all_files = []
        
        # Collect all batch files
        for hop_dir in sorted(self.sql_dir.iterdir()):
            if hop_dir.is_dir() and hop_dir.name.endswith('_hop'):
                batch_files = sorted(hop_dir.glob('batch_*.json'))
                for batch_file in batch_files:
                    # Get file size and query count
                    try:
                        with open(batch_file, 'r', encoding='utf-8') as f:
                            queries = json.load(f)
                            query_count = len(queries)
                    except:
                        query_count = "unknown"
                    
                    relative_path = batch_file.relative_to(self.sql_dir)
                    all_files.append((batch_file, relative_path, query_count))
        
        return all_files
    
    def process_specific_batches(self):
        """Allow user to select and process specific batch files."""
        print("📋 Available batch files:")
        print("=" * 80)
        
        all_files = self.list_available_batches()
        
        if not all_files:
            print("❌ No batch files found!")
            return
        
        # Display available files
        for i, (batch_file, relative_path, query_count) in enumerate(all_files):
            print(f"{i+1:2d}. {relative_path} ({query_count} queries)")
        
        print(f"\n📊 Total: {len(all_files)} batch files")
        
        # Get user selection
        print("\nOptions:")
        print("a. Process all files")
        print("r. Process files by range (e.g., 1-5, 10-15)")
        print("s. Process specific files (e.g., 1,3,7,12)")
        print("h. Process by hop count (e.g., 3_hop, 4_hop)")
        
        choice = input("\nEnter your choice (a/r/s/h): ").strip().lower()
        
        selected_files = []
        
        if choice == 'a':
            selected_files = all_files
        elif choice == 'r':
            range_input = input("Enter range (e.g., 1-5): ").strip()
            try:
                if '-' in range_input:
                    start, end = map(int, range_input.split('-'))
                    selected_files = all_files[start-1:end]
                else:
                    print("❌ Invalid range format")
                    return
            except ValueError:
                print("❌ Invalid range format")
                return
        elif choice == 's':
            indices_input = input("Enter file numbers separated by commas (e.g., 1,3,7): ").strip()
            try:
                indices = [int(x.strip()) for x in indices_input.split(',')]
                selected_files = [all_files[i-1] for i in indices if 1 <= i <= len(all_files)]
            except (ValueError, IndexError):
                print("❌ Invalid file numbers")
                return
        elif choice == 'h':
            hop_input = input("Enter hop count (e.g., 3, 4, 5): ").strip()
            try:
                hop_count = int(hop_input)
                hop_dir_name = f"{hop_count}_hop"
                selected_files = [f for f in all_files if hop_dir_name in str(f[1])]
            except ValueError:
                print("❌ Invalid hop count")
                return
        else:
            print("❌ Invalid choice")
            return
        
        if not selected_files:
            print("❌ No files selected")
            return
        
        # Confirm selection
        print(f"\n✅ Selected {len(selected_files)} files:")
        total_queries = 0
        for i, (_, relative_path, query_count) in enumerate(selected_files):
            print(f"   {i+1}. {relative_path} ({query_count} queries)")
            if isinstance(query_count, int):
                total_queries += query_count
        
        if isinstance(total_queries, int):
            estimated_cost = (total_queries * 850 / 1_000_000) * self.token_tracker.cost_per_million_tokens
            print(f"\n📊 Estimated: {total_queries:,} queries, ~${estimated_cost:.4f}")
        
        confirm = input(f"\nProcess these {len(selected_files)} files? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return
        
        # Process selected files
        print(f"\n🚀 Processing {len(selected_files)} selected batch files...")
        
        processed_queries = 0
        start_time = time.time()
        
        for i, (input_file, relative_path, _) in enumerate(selected_files):
            output_file = self.output_dir / relative_path
            print(f"\n[{i+1}/{len(selected_files)}] {relative_path}")
            count = self.process_batch_file(input_file, output_file)
            processed_queries += count
            
            # Estimate time remaining
            elapsed = time.time() - start_time
            if i > 0:  # Avoid division by zero
                rate = (i + 1) / elapsed
                remaining = (len(selected_files) - i - 1) / rate
                print(f"⏱️  ETA: {remaining/60:.1f} minutes")
            
            # Show running token totals
            summary = self.token_tracker.get_summary()
            print(f"🔢 Running totals: {summary['total_tokens']:,} tokens | ${summary['estimated_cost_usd']:.4f}")
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 COMPLETED!")
        print(f"   Selected files: {len(selected_files)}")
        print(f"   Total queries: {processed_queries:,}")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Average rate: {processed_queries/total_time:.1f} queries/second")
        
        # Final token summary
        self.token_tracker.print_summary()
        
        # Save final summary
        summary_file = self.output_dir / "specific_processing_summary.json"
        final_summary = {
            'processing_complete': True,
            'processing_type': 'specific_batches',
            'timestamp': datetime.now().isoformat(),
            'files_processed': len(selected_files),
            'queries_processed': processed_queries,
            'total_time_seconds': total_time,
            'selected_files': [str(f[1]) for f in selected_files],
            'token_summary': self.token_tracker.get_summary()
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Summary saved to: {summary_file}")
    
    def process_all_batches(self):
        """Process all batch files in the sql_queries directory."""
        print("🚀 Processing all batch files...")
        
        all_files = []
        
        # Collect all batch files
        for hop_dir in sorted(self.sql_dir.iterdir()):
            if hop_dir.is_dir() and hop_dir.name.endswith('_hop'):
                batch_files = sorted(hop_dir.glob('batch_*.json'))
                for batch_file in batch_files:
                    relative_path = batch_file.relative_to(self.sql_dir)
                    output_file = self.output_dir / relative_path
                    all_files.append((batch_file, output_file))
        
        print(f"📊 Found {len(all_files)} batch files to process")
        
        total_queries = 0
        start_time = time.time()
        
        # Process files sequentially (to respect rate limits)
        for i, (input_file, output_file) in enumerate(all_files):
            print(f"\n[{i+1}/{len(all_files)}] {input_file.parent.name}/{input_file.name}")
            count = self.process_batch_file(input_file, output_file)
            total_queries += count
            
            # Estimate time remaining
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(all_files) - i - 1) / rate
            print(f"⏱️  ETA: {remaining/60:.1f} minutes")
            
            # Show running token totals
            summary = self.token_tracker.get_summary()
            print(f"🔢 Running totals: {summary['total_tokens']:,} tokens | ${summary['estimated_cost_usd']:.4f}")
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 COMPLETED!")
        print(f"   Total files: {len(all_files)}")
        print(f"   Total queries: {total_queries:,}")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Average rate: {total_queries/total_time:.1f} queries/second")
        
        # Final token summary
        self.token_tracker.print_summary()
        
        # Save final summary
        summary_file = self.output_dir / "processing_summary.json"
        final_summary = {
            'processing_complete': True,
            'timestamp': datetime.now().isoformat(),
            'files_processed': len(all_files),
            'queries_processed': total_queries,
            'total_time_seconds': total_time,
            'token_summary': self.token_tracker.get_summary()
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Summary saved to: {summary_file}")

def main():
    """Main function to run the generator."""
    generator = LLMNaturalQueryGenerator()
    
    # Quick test option
    if len(sys.argv) > 1 and sys.argv[1] == "--test10":
        print("🚀 Quick test mode: Processing 10 examples")
        generator.test_single_batch(num_examples=10)
        return
    
    print("🤖 LLM Natural Query Generator with Token Tracking")
    print(f"📍 Model: {MODEL} (${generator.token_tracker.cost_per_million_tokens}/1M tokens)")
    print(f"📁 Input: {generator.sql_dir}")
    print(f"📁 Output: {generator.output_dir}")
    print("=" * 60)
    
    # Ask user what to do
    print("\nOptions:")
    print("1. Test with 10 examples from one batch")
    print("2. Test with full batch")
    print("3. Process all batches")
    print("4. Process specific hop directory (e.g., all 3_hop files)")
    print("5. Process custom selection of batches")
    
    choice = input("\nEnter choice (1, 2, 3, 4, or 5): ").strip()
    
    if choice == "1":
        generator.test_single_batch(num_examples=10)
    elif choice == "2":
        generator.test_single_batch(num_examples=100)  # Process full batch
    elif choice == "3":
        confirm = input("\n⚠️  This will process ALL batches and may take hours. Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            generator.process_all_batches()
        else:
            print("Cancelled.")
    elif choice == "4":
        generator.process_specific_hop()
    elif choice == "5":
        generator.process_specific_batches()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()