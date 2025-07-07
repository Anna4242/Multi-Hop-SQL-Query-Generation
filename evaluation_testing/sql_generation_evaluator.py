#!/usr/bin/env python3
"""
SQL Generation Evaluator for Connect Dots
Evaluates LLM's ability to generate SQL queries from natural language questions
using the generated data as ground truth.
"""

import json
import os
import sqlite3
import pathlib
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dotenv import load_dotenv
import requests
from datetime import datetime

# Load environment variables
DOTENV_PATH = pathlib.Path(__file__).resolve().parents[2] / ".env"
load_dotenv(DOTENV_PATH)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing environment variable: OPENROUTER_API_KEY")
if not OPENAI_API_BASE:
    raise RuntimeError("Missing environment variable: OPENAI_API_BASE")

MODEL = "qwen/qwen-2.5-72b-instruct"

class DatabaseSchemaExtractor:
    """Extract database schema information from BIRD dataset."""
    
    def __init__(self, bird_db_path: str):
        self.bird_db_path = Path(bird_db_path)
    
    def get_database_schema(self, db_name: str) -> Dict:
        """Get complete schema information for a database."""
        db_path = self.bird_db_path / db_name / f"{db_name}.sqlite"
        
        if not db_path.exists():
            return {"error": f"Database {db_name} not found"}
        
        schema_info = {
            "database": db_name,
            "tables": {},
            "foreign_keys": []
        }
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for (table_name,) in tables:
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                table_schema = {
                    "columns": [],
                    "primary_keys": [],
                    "sample_data": []
                }
                
                for col in columns:
                    col_info = {
                        "name": col[1],
                        "type": col[2],
                        "not_null": bool(col[3]),
                        "default": col[4],
                        "primary_key": bool(col[5])
                    }
                    table_schema["columns"].append(col_info)
                    
                    if col[5]:  # Primary key
                        table_schema["primary_keys"].append(col[1])
                
                # Get sample data (first 3 rows)
                try:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                    sample_rows = cursor.fetchall()
                    table_schema["sample_data"] = sample_rows
                except:
                    table_schema["sample_data"] = []
                
                schema_info["tables"][table_name] = table_schema
            
            # Get foreign keys
            for table_name in schema_info["tables"]:
                cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                fks = cursor.fetchall()
                
                for fk in fks:
                    fk_info = {
                        "from_table": table_name,
                        "from_column": fk[3],
                        "to_table": fk[2],
                        "to_column": fk[4]
                    }
                    schema_info["foreign_keys"].append(fk_info)
            
            conn.close()
            
        except Exception as e:
            schema_info["error"] = str(e)
        
        return schema_info

class SQLGenerationEvaluator:
    """Evaluate LLM's SQL generation capabilities."""
    
    def __init__(self, bird_db_path: str):
        self.api_key = OPENROUTER_API_KEY
        self.api_base = OPENAI_API_BASE
        self.model = MODEL
        self.schema_extractor = DatabaseSchemaExtractor(bird_db_path)
        
        # Rate limiting
        self.min_time_between_requests = 1.0
        self.last_request_time = 0
    
    def create_schema_context(self, schema_info: Dict) -> str:
        """Create a formatted schema context for the LLM."""
        if "error" in schema_info:
            return f"Error: {schema_info['error']}"
        
        context = f"Database: {schema_info['database']}\n\n"
        context += "Tables and Columns:\n"
        
        for table_name, table_info in schema_info["tables"].items():
            context += f"\n{table_name}:\n"
            
            for col in table_info["columns"]:
                pk_marker = " (PRIMARY KEY)" if col["primary_key"] else ""
                context += f"  - {col['name']}: {col['type']}{pk_marker}\n"
            
            # Add sample data if available
            if table_info["sample_data"]:
                context += f"  Sample data: {table_info['sample_data'][:2]}\n"
        
        # Add foreign key relationships
        if schema_info["foreign_keys"]:
            context += "\nForeign Key Relationships:\n"
            for fk in schema_info["foreign_keys"]:
                context += f"  {fk['from_table']}.{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}\n"
        
        return context
    
    def call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM API to generate SQL."""
        # Rate limiting
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
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 500
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
    
    def create_sql_generation_prompt(self, question: str, schema_context: str) -> str:
        """Create prompt for SQL generation."""
        prompt = f"""You are an expert SQL query generator. Given a natural language question and database schema, generate a SQL query that answers the question.

Database Schema:
{schema_context}

Question: {question}

Instructions:
1. Generate a SQL query that answers the question accurately
2. Use appropriate JOIN types (INNER JOIN, LEFT JOIN, etc.) to connect tables
3. Use table aliases (t0, t1, t2, etc.) for clarity
4. Follow the exact format: SELECT * FROM table AS t0 JOIN...
5. Make sure the query is syntactically correct
6. Only use tables and columns that exist in the schema
7. Focus on the relationships between tables to create the correct path

Generate only the SQL query, nothing else:"""
        
        return prompt

def main():
    """Main evaluation function."""
    print("🧪 SQL Generation Evaluator")
    print("=" * 50)
    
    # Configuration
    bird_db_path = "../bird/train/train_databases/train_databases"
    batch_file = Path("final_data/5_hop/batch_001.json")
    
    # Check if batch file exists
    if not batch_file.exists():
        print(f"❌ Batch file not found: {batch_file}")
        return
    
    print(f"📁 BIRD Database Path: {bird_db_path}")
    print(f"📋 Batch File: {batch_file}")
    print(f"🤖 Model: {MODEL}")
    
    # Initialize evaluator
    evaluator = SQLGenerationEvaluator(bird_db_path)
    
    # Load batch data
    with open(batch_file, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
    
    # Test with first 3 queries
    test_queries = batch_data[:3]
    
    for i, query_data in enumerate(test_queries):
        print(f"\n{'='*60}")
        print(f"🔍 QUERY {i+1}/3")
        print(f"{'='*60}")
        
        db_name = query_data["db_id"]
        question = query_data["natural_query"]
        true_sql = query_data["sql"]
        true_path = query_data["path"]
        
        print(f"📊 Database: {db_name}")
        print(f"❓ Question: {question}")
        print(f"🎯 True Path: {' -> '.join(true_path)}")
        print(f"✅ True SQL: {true_sql}")
        
        # Get database schema
        schema_info = evaluator.schema_extractor.get_database_schema(db_name)
        
        if "error" in schema_info:
            print(f"❌ Schema error: {schema_info['error']}")
            continue
        
        # Create schema context
        schema_context = evaluator.create_schema_context(schema_info)
        
        # Generate SQL
        prompt = evaluator.create_sql_generation_prompt(question, schema_context)
        print(f"\n📝 Generating SQL...")
        
        generated_sql = evaluator.call_llm(prompt)
        
        if generated_sql:
            print(f"🤖 Generated SQL: {generated_sql}")
            
            # Simple comparison
            if generated_sql.strip() == true_sql.strip():
                print("✅ EXACT MATCH!")
            else:
                print("❌ Different from ground truth")
        else:
            print("❌ Failed to generate SQL")
        
        print(f"\n{'='*60}")

if __name__ == "__main__":
    main() 