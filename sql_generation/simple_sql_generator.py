#!/usr/bin/env python3
"""
Simple SQL Query Generator using Qwen 2.5 72B
Generates SQL and saves in source format with only generated_sql added
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any
import glob
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    OPENROUTER_API_KEY, DEFAULT_MODEL, BIRD_DB_PATH,
    FINAL_DATA_DIR, GENERATED_QUERY_SIMPLE_DIR,
    get_database_description_path, get_hop_data_dir,
    validate_config
)

# Validate configuration
validate_config()

# Configuration
MODEL = DEFAULT_MODEL
HOP_LENGTHS = [20]

class SimpleSQLGenerator:
    def __init__(self, bird_db_path: str = None, source_dir: str = None):
        self.bird_db_path = Path(bird_db_path) if bird_db_path else Path(BIRD_DB_PATH)
        self.source_dir = Path(source_dir) if source_dir else FINAL_DATA_DIR
        self.output_dir = GENERATED_QUERY_SIMPLE_DIR
        
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

    def call_qwen(self, prompt: str) -> Dict[str, Any]:
        """Call Qwen 2.5 72B via OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/yourusername/sqlmultihop",
            "X-Title": "SQL MultiHop Generation",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"ERROR: API Error {response.status_code}")
                return None
                
        except Exception as e:
            print(f"ERROR: Request failed: {e}")
            return None
    
    def process_single_query(self, query_data: Dict) -> Dict[str, Any]:
        """Process a single query and add generated SQL to original format."""
        db_name = query_data["db_id"]
        question = query_data["natural_query"]
        true_path = query_data["path"]
        
        # Get schema
        schema_info = self.get_database_schema(db_name, true_path)
        
        # Create clean prompt (without ground truth SQL)
        prompt = self.create_clean_prompt(question, schema_info, true_path)
        
        # Call API
        response = self.call_qwen(prompt)
        
        # Copy original data
        result = query_data.copy()
        
        if response and 'choices' in response:
            generated_sql = response['choices'][0]['message']['content']
            
            # Clean SQL
            if "```" in generated_sql:
                generated_sql = generated_sql.split("```")[1]
                if generated_sql.startswith("sql"):
                    generated_sql = generated_sql[3:]
                generated_sql = generated_sql.strip()
            
            # Add generated SQL to original format
            result["generated_sql"] = generated_sql
        else:
            result["generated_sql"] = None
        
        return result
    
    def process_batch_file(self, batch_path: Path, hop_length: int) -> None:
        """Process a single batch file."""
        print(f"Processing {batch_path.name}...")
        
        # Load batch data
        with open(batch_path, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        # Process each query
        results = []
        successful = 0
        
        for i, query_data in enumerate(batch_data):
            result = self.process_single_query(query_data)
            results.append(result)
            
            if result.get("generated_sql"):
                successful += 1
            
            # Progress update
            if (i + 1) % 10 == 0:
                success_rate = (successful / (i + 1)) * 100
                print(f"   [{i+1:3d}/{len(batch_data)}] Success: {success_rate:5.1f}%", end='\r')
        
        # Save results in same format as source
        output_hop_dir = get_hop_data_dir(hop_length, "generated_query_simple")
        output_hop_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_hop_dir / batch_path.name
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"   COMPLETE {batch_path.name}: {successful}/{len(batch_data)} "
              f"({successful/len(batch_data)*100:.1f}%) saved to {output_file}")
    
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
        print(f"Output: {get_hop_data_dir(hop_length, 'generated_query_simple')}")
        
        for batch_file in batch_files:
            self.process_batch_file(batch_file, hop_length)
        
        print(f"COMPLETE: {hop_length}-hop finished!")
    
    def run_generation(self):
        """Run generation for all hop lengths."""
        print(f"Simple SQL Generation")
        print(f"Model: {MODEL}")
        print(f"Processing hop lengths: {HOP_LENGTHS}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)
        
        overall_start = time.time()
        
        for hop_length in HOP_LENGTHS:
            self.process_hop_length(hop_length)
        
        overall_time = time.time() - overall_start
        
        print(f"\n{'='*80}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"Model: {MODEL}")
        print(f"Total Time: {overall_time/60:.1f} minutes")
        print(f"Results saved in: {self.output_dir}")
        print(f"Format: Same as source data + generated_sql field")
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
    
    # Run generation
    generator = SimpleSQLGenerator()
    generator.run_generation()

if __name__ == "__main__":
    main() 