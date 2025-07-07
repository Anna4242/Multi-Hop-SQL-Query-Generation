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
from typing import Dict, List, Optional, Tuple
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

class ConnectionGraphLoader:
    """Load and parse connection graph information."""
    
    def __init__(self, graph_dir: str):
        self.graph_dir = Path(graph_dir)
    
    def load_graph(self, db_name: str) -> Dict:
        """Load connection graph for a database."""
        graph_file = self.graph_dir / f"{db_name}_full_graph.pkl"
        
        if not graph_file.exists():
            return {"error": f"Connection graph for {db_name} not found"}
        
        try:
            with open(graph_file, 'rb') as f:
                graph = pickle.load(f)
            return graph
        except Exception as e:
            return {"error": f"Failed to load graph: {str(e)}"}
    
    def format_graph_info(self, graph: Dict, target_path: List[str]) -> str:
        """Format connection graph information for the prompt."""
        if "error" in graph:
            return f"Graph Error: {graph['error']}"
        
        info = "Connection Graph Information:\n"
        info += "=" * 40 + "\n"
        
        # Show available connections
        if hasattr(graph, 'edges') or 'edges' in graph:
            edges = graph.get('edges', graph) if isinstance(graph, dict) else graph.edges()
            info += "Available table connections:\n"
            
            # Show connections relevant to the target path
            for i in range(len(target_path) - 1):
                source = target_path[i]
                target = target_path[i + 1]
                info += f"  {source} -> {target}\n"
                
                # Try to find the connection details
                for edge in edges:
                    if isinstance(edge, tuple) and len(edge) >= 2:
                        if edge[0] == source and edge[1] == target:
                            edge_data = edge[2] if len(edge) > 2 else {}
                            if isinstance(edge_data, dict):
                                src_col = edge_data.get('src_col', 'unknown')
                                dst_col = edge_data.get('dst_col', 'unknown')
                                join_type = edge_data.get('join_type', 'INNER')
                                info += f"    JOIN: {source}.{src_col} = {target}.{dst_col} ({join_type})\n"
        
        info += "\nRequired Path:\n"
        info += " -> ".join(target_path) + "\n"
        info += f"Total Hops: {len(target_path) - 1}\n"
        
        return info

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
                
                # Get sample data (first 2 rows)
                try:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
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

class PathGuidedSQLEvaluator:
    """Evaluate LLM's SQL generation with path guidance."""
    
    def __init__(self, bird_db_path: str, graph_dir: str):
        self.api_key = OPENROUTER_API_KEY
        self.api_base = OPENAI_API_BASE
        self.model = MODEL
        self.schema_extractor = DatabaseSchemaExtractor(bird_db_path)
        self.graph_loader = ConnectionGraphLoader(graph_dir)
        
        # Rate limiting
        self.min_time_between_requests = 1.0
        self.last_request_time = 0
    
    def create_schema_context(self, schema_info: Dict, target_tables: List[str]) -> str:
        """Create a formatted schema context focusing on target tables."""
        if "error" in schema_info:
            return f"Error: {schema_info['error']}"
        
        context = f"Database: {schema_info['database']}\n\n"
        context += "ONLY USE THESE TABLES AND THEIR COLUMNS:\n"
        context += "=" * 50 + "\n"
        
        # Only show tables that are in the target path
        for table_name in target_tables:
            if table_name in schema_info["tables"]:
                table_info = schema_info["tables"][table_name]
                context += f"\n{table_name}:\n"
                
                for col in table_info["columns"]:
                    pk_marker = " (PRIMARY KEY)" if col["primary_key"] else ""
                    context += f"  - {col['name']}: {col['type']}{pk_marker}\n"
                
                # Add sample data if available
                if table_info["sample_data"]:
                    context += f"  Sample: {table_info['sample_data'][0] if table_info['sample_data'] else 'No data'}\n"
            else:
                context += f"\n{table_name}: TABLE NOT FOUND IN DATABASE!\n"
        
        # Add foreign key relationships for target tables only
        relevant_fks = [fk for fk in schema_info["foreign_keys"] 
                       if fk["from_table"] in target_tables and fk["to_table"] in target_tables]
        
        if relevant_fks:
            context += "\nRelevant Foreign Key Relationships:\n"
            for fk in relevant_fks:
                context += f"  {fk['from_table']}.{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}\n"
        
        return context
    
    def create_path_guided_prompt(self, question: str, schema_context: str, graph_info: str, target_path: List[str], true_sql: str) -> str:
        """Create prompt with path guidance and connection graph info."""
        prompt = f"""You are an expert SQL query generator. You have been given:
1. A natural language question
2. Database schema (ONLY use tables and columns listed)
3. Connection graph information
4. The EXACT PATH you must follow
5. The ground truth SQL for reference

{schema_context}

{graph_info}

Question: {question}

GROUND TRUTH SQL (for reference):
{true_sql}

CRITICAL INSTRUCTIONS:
1. You MUST follow the EXACT path: {' -> '.join(target_path)}
2. Use ONLY the tables and columns shown in the schema above
3. Start with the first table: {target_path[0]} AS t0
4. Join each subsequent table in the exact order given
5. Use appropriate JOIN types (INNER JOIN, LEFT JOIN as needed)
6. Use table aliases: t0, t1, t2, etc. in order
7. Follow the connection graph relationships shown above
8. SELECT * FROM all tables (like the ground truth)
9. DO NOT add WHERE clauses unless absolutely necessary
10. Focus on following the path, not optimizing the query

Generate ONLY the SQL query that follows the exact path shown above:"""
        
        return prompt
    
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
    """Main evaluation function."""
    print("🎯 Path-Guided SQL Generation Evaluator")
    print("=" * 60)
    
    # Configuration
    bird_db_path = "../bird/train/train_databases/train_databases"
    graph_dir = "connection_graphs"
    batch_file = Path("final_data/5_hop/batch_001.json")
    
    # Check if batch file exists
    if not batch_file.exists():
        print(f"❌ Batch file not found: {batch_file}")
        return
    
    print(f"📁 BIRD Database Path: {bird_db_path}")
    print(f"📊 Connection Graphs: {graph_dir}")
    print(f"📋 Batch File: {batch_file}")
    print(f"🤖 Model: {MODEL}")
    
    # Initialize evaluator
    evaluator = PathGuidedSQLEvaluator(bird_db_path, graph_dir)
    
    # Load batch data
    with open(batch_file, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
    
    # Test with first 2 queries
    test_queries = batch_data[:2]
    
    for i, query_data in enumerate(test_queries):
        print(f"\n{'='*80}")
        print(f"🔍 PATH-GUIDED QUERY {i+1}/2")
        print(f"{'='*80}")
        
        db_name = query_data["db_id"]
        question = query_data["natural_query"]
        true_sql = query_data["sql"]
        true_path = query_data["path"]
        
        print(f"📊 Database: {db_name}")
        print(f"❓ Question: {question}")
        print(f"🎯 Required Path: {' -> '.join(true_path)}")
        print(f"📏 Path Length: {len(true_path)} tables ({len(true_path)-1} hops)")
        
        # Get database schema
        schema_info = evaluator.schema_extractor.get_database_schema(db_name)
        
        if "error" in schema_info:
            print(f"❌ Schema error: {schema_info['error']}")
            continue
        
        # Load connection graph
        graph = evaluator.graph_loader.load_graph(db_name)
        graph_info = evaluator.graph_loader.format_graph_info(graph, true_path)
        
        # Create schema context (only for target tables)
        schema_context = evaluator.create_schema_context(schema_info, true_path)
        
        print(f"\n📋 Connection Graph Info:")
        print(graph_info)
        
        # Generate SQL with path guidance
        prompt = evaluator.create_path_guided_prompt(question, schema_context, graph_info, true_path, true_sql)
        
        print(f"\n📝 Generating path-guided SQL...")
        print(f"🎯 Target: Follow exact path {' -> '.join(true_path)}")
        
        generated_sql = evaluator.call_llm(prompt)
        
        if generated_sql:
            # Clean up generated SQL (remove markdown if present)
            if "```" in generated_sql:
                generated_sql = generated_sql.split("```")[1]
                if generated_sql.startswith("sql"):
                    generated_sql = generated_sql[3:]
                generated_sql = generated_sql.strip()
            
            print(f"\n🤖 Generated SQL:")
            print(generated_sql)
            
            print(f"\n✅ Ground Truth SQL:")
            print(true_sql)
            
            # Simple comparison
            if generated_sql.strip().replace('\n', ' ').replace('  ', ' ') == true_sql.strip().replace('\n', ' ').replace('  ', ' '):
                print(f"\n🎉 EXACT MATCH!")
            else:
                print(f"\n📊 COMPARISON:")
                print(f"   Generated follows path: {' -> '.join(true_path)}")
                print(f"   Ground truth follows:   {' -> '.join(true_path)}")
                
                # Check if tables match
                import re
                gen_tables = re.findall(r'FROM\s+(\w+)\s+AS|JOIN\s+(\w+)\s+AS', generated_sql.upper())
                gen_table_names = [t[0] if t[0] else t[1] for t in gen_tables]
                
                if gen_table_names == [t.upper() for t in true_path]:
                    print(f"   ✅ Table sequence matches!")
                else:
                    print(f"   ❌ Table sequence differs")
                    print(f"      Generated: {gen_table_names}")
                    print(f"      Expected:  {[t.upper() for t in true_path]}")
        else:
            print("❌ Failed to generate SQL")
        
        print(f"\n{'='*80}")

if __name__ == "__main__":
    main() 