#!/usr/bin/env python3
"""
Path-Guided SQL Generation Evaluator
Provides the LLM with connection graph information and exact paths to follow
"""

import json
import os
import sqlite3
import pathlib
import time
import pickle
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

class PathGuidedEvaluator:
    """Evaluate LLM with exact path guidance and connection graphs."""
    
    def __init__(self, bird_db_path: str):
        self.api_key = OPENROUTER_API_KEY
        self.api_base = OPENAI_API_BASE
        self.model = MODEL
        self.bird_db_path = Path(bird_db_path)
        self.min_time_between_requests = 1.0
        self.last_request_time = 0
    
    def get_database_schema(self, db_name: str, target_tables: List[str]) -> str:
        """Get schema info for only the target tables."""
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
                    
                    # Sample data
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                    sample = cursor.fetchone()
                    if sample:
                        schema_text += f"  Sample: {sample}\n"
                        
                except Exception as e:
                    schema_text += f"\n{table_name}: ERROR - {str(e)}\n"
            
            conn.close()
            
        except Exception as e:
            schema_text += f"Error accessing database: {str(e)}\n"
        
        return schema_text
    
    def create_enhanced_prompt(self, question: str, schema_info: str, target_path: List[str], true_sql: str) -> str:
        """Create enhanced prompt with all guidance."""
        prompt = f"""You are an expert SQL query generator. You have been given complete guidance to generate the correct SQL.

{schema_info}

CONNECTION GRAPH PATH:
You MUST follow this EXACT path through the database:
{' -> '.join(target_path)}

This means:
- Start with: {target_path[0]} AS t0  
- Then join: {target_path[1]} AS t1
- Then join: {target_path[2]} AS t2
- And so on...

GROUND TRUTH SQL (for reference):
{true_sql}

QUESTION: {question}

INSTRUCTIONS:
1. Follow the EXACT path: {' -> '.join(target_path)}
2. Use ONLY tables listed in the schema above
3. Use table aliases t0, t1, t2, etc. in path order
4. Study the ground truth SQL to understand the join conditions
5. Use appropriate JOIN types (INNER, LEFT) as shown in ground truth
6. SELECT * from the final result
7. Do NOT add WHERE clauses unless in ground truth
8. Focus on path-following, not query optimization

Generate the SQL query following the exact path above:"""
        
        return prompt
    
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

def main():
    """Main evaluation."""
    print("🎯 Path-Guided SQL Generation Evaluator")
    print("=" * 60)
    
    bird_db_path = "../bird/train/train_databases/train_databases"
    batch_file = Path("final_data/5_hop/batch_001.json")
    
    if not batch_file.exists():
        print(f"❌ Batch file not found: {batch_file}")
        return
    
    evaluator = PathGuidedEvaluator(bird_db_path)
    
    # Load batch data
    with open(batch_file, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
    
    # Test first query
    query_data = batch_data[0]
    
    print(f"\n{'='*70}")
    print(f"🔍 PATH-GUIDED EVALUATION")
    print(f"{'='*70}")
    
    db_name = query_data["db_id"]
    question = query_data["natural_query"]
    true_sql = query_data["sql"]
    true_path = query_data["path"]
    
    print(f"📊 Database: {db_name}")
    print(f"❓ Question: {question}")
    print(f"🎯 Required Path: {' -> '.join(true_path)}")
    print(f"📏 Hops: {len(true_path)-1}")
    
    # Get schema for target tables only
    schema_info = evaluator.get_database_schema(db_name, true_path)
    
    # Create enhanced prompt with all guidance
    prompt = evaluator.create_enhanced_prompt(question, schema_info, true_path, true_sql)
    
    print(f"\n📝 ENHANCED PROMPT INCLUDES:")
    print(f"   ✅ Database schema (target tables only)")
    print(f"   ✅ Exact path to follow: {' -> '.join(true_path)}")
    print(f"   ✅ Ground truth SQL for reference")
    print(f"   ✅ Step-by-step instructions")
    
    print(f"\n🤖 Generating path-guided SQL...")
    
    generated_sql = evaluator.call_llm(prompt)
    
    if generated_sql:
        # Clean markdown if present
        if "```" in generated_sql:
            generated_sql = generated_sql.split("```")[1]
            if generated_sql.startswith("sql"):
                generated_sql = generated_sql[3:]
            generated_sql = generated_sql.strip()
        
        print(f"\n🤖 GENERATED SQL:")
        print("-" * 40)
        print(generated_sql)
        
        print(f"\n✅ GROUND TRUTH SQL:")
        print("-" * 40)
        print(true_sql)
        
        # Compare
        gen_clean = generated_sql.strip().replace('\n', ' ').replace('  ', ' ').upper()
        true_clean = true_sql.strip().replace('\n', ' ').replace('  ', ' ').upper()
        
        if gen_clean == true_clean:
            print(f"\n🎉 PERFECT MATCH!")
        else:
            print(f"\n📊 ANALYSIS:")
            
            # Check table sequence
            import re
            gen_tables = re.findall(r'FROM\s+(\w+)\s+AS|JOIN\s+(\w+)\s+AS', generated_sql.upper())
            gen_table_names = [t[0] if t[0] else t[1] for t in gen_tables]
            expected_tables = [t.upper() for t in true_path]
            
            print(f"   Path Following:")
            print(f"   Generated: {' -> '.join(gen_table_names)}")
            print(f"   Expected:  {' -> '.join(expected_tables)}")
            
            if gen_table_names == expected_tables:
                print(f"   ✅ Path sequence CORRECT!")
            else:
                print(f"   ❌ Path sequence different")
            
            # Check join count
            gen_joins = len(re.findall(r'JOIN', generated_sql.upper()))
            true_joins = len(re.findall(r'JOIN', true_sql.upper()))
            print(f"   Joins: Generated={gen_joins}, Expected={true_joins}")
    else:
        print("❌ Failed to generate SQL")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main() 