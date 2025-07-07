#!/usr/bin/env python3
"""
Qwen 2.5 72B test for 100 queries
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

# Load environment variables
DOTENV_PATH = pathlib.Path(__file__).resolve().parents[2] / ".env"
load_dotenv(DOTENV_PATH)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
MODEL = "qwen/qwen-2.5-72b-instruct"

def get_database_schema(bird_db_path: Path, db_name: str, target_tables: List[str]) -> str:
    """Get schema for target tables only."""
    db_path = bird_db_path / db_name / f"{db_name}.sqlite"
    
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

def call_qwen(prompt: str):
    """Call Qwen API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 600
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
            return result
        else:
            return None
            
    except Exception:
        return None

def extract_tables(sql: str) -> List[str]:
    """Extract table sequence from SQL."""
    sql_upper = sql.upper()
    matches = re.findall(r'FROM\s+(\w+)\s+AS|JOIN\s+(\w+)\s+AS', sql_upper)
    return [t[0] if t[0] else t[1] for t in matches]

def main():
    """Main function."""
    bird_db_path = Path("../bird/train/train_databases/train_databases")
    batch_file = Path("final_data/5_hop/batch_001.json")
    
    if not batch_file.exists():
        print(f"❌ File not found: {batch_file}")
        return
    
    # Load data
    with open(batch_file, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
    
    print(f"📋 Loaded {len(batch_data)} queries")
    print(f"🚀 Testing Qwen 2.5 72B on 100 queries")
    print("=" * 60)
    
    # Test 100 queries
    test_queries = batch_data[:100]
    
    exact_matches = 0
    path_matches = 0
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i, query_data in enumerate(test_queries):
        db_name = query_data["db_id"]
        question = query_data["natural_query"]
        true_sql = query_data["sql"]
        true_path = query_data["path"]
        
        # Get schema
        schema_info = get_database_schema(bird_db_path, db_name, true_path)
        
        # Create prompt
        prompt = f"""Generate SQL following the exact path provided.

{schema_info}

PATH: {' -> '.join(true_path)}

REFERENCE SQL:
{true_sql}

QUESTION: {question}

RULES:
1. Follow EXACT path: {' -> '.join(true_path)}
2. Use aliases t0, t1, t2, etc.
3. Copy join patterns from reference
4. SELECT *

Generate SQL:"""
        
        # Call API
        response = call_qwen(prompt)
        
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
            
            gen_tables = extract_tables(generated_sql)
            expected_tables = [t.upper() for t in true_path]
            path_match = gen_tables == expected_tables
            
            successful += 1
            if exact_match:
                exact_matches += 1
            if path_match:
                path_matches += 1
            
            status = "✅" if exact_match else "❌"
            print(f"[{i+1:3d}/100] {status} {db_name:15s} | "
                  f"Exact: {(exact_matches/successful)*100:5.1f}% | "
                  f"Path: {(path_matches/successful)*100:5.1f}%")
        else:
            failed += 1
            print(f"[{i+1:3d}/100] ⚠️  {db_name:15s} | FAILED")
        
        # Rate limiting
        time.sleep(1.0)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"📊 QWEN 2.5 72B RESULTS")
    print(f"{'='*60}")
    print(f"Total: 100 queries")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"")
    print(f"🎯 ACCURACY:")
    print(f"Exact Matches: {exact_matches}/{successful} ({(exact_matches/max(1,successful))*100:.1f}%)")
    print(f"Path Matches:  {path_matches}/{successful} ({(path_matches/max(1,successful))*100:.1f}%)")
    print(f"")
    print(f"⏱️ Time: {total_time/60:.1f} minutes")
    print(f"Rate: {successful/total_time:.1f} queries/second")

if __name__ == "__main__":
    main() 