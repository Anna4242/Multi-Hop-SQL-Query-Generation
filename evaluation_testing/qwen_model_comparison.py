#!/usr/bin/env python3
"""
Compare Qwen 2.5 72B vs Qwen 3 8B on the same 5 queries
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

MODELS = {
    "qwen2": "qwen/qwen-2.5-72b-instruct",
    "qwen3": "qwen/qwen3-32b"
}

class QwenModelComparison:
    def __init__(self, bird_db_path: str):
        self.bird_db_path = Path(bird_db_path)
        self.results = {
            "qwen2": [],
            "qwen3": []
        }
        
    def get_database_schema(self, db_name: str, target_tables: List[str]) -> str:
        """Get simplified schema for target tables only."""
        db_path = self.bird_db_path / db_name / "database_description" / "database_description.txt"
        
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
    
    def create_prompt(self, question: str, schema_info: str, target_path: List[str], true_sql: str) -> str:
        """Create prompt for SQL generation."""
        path_str = " -> ".join(target_path)
        
        return f"""You are an expert SQL developer. Generate a SQL query that answers the question using EXACTLY the specified table path.

DATABASE SCHEMA:
{schema_info}

QUESTION: {question}

REQUIRED PATH: {path_str}
(You MUST use these tables in this exact order)

REFERENCE SQL: {true_sql}

Generate ONLY the SQL query without any explanations or markdown formatting."""

    def call_model(self, prompt: str, model_name: str) -> Dict[str, Any]:
        """Call the specified model via OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/yourusername/sqlmultihop",
            "X-Title": "SQL MultiHop Query Generation",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": MODELS[model_name],
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
                print(f"❌ API Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def extract_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL query."""
        import re
        tables = re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', sql.upper())
        return [t[0] if t[0] else t[1] for t in tables]
    
    def test_single_query(self, query_data: Dict, model_name: str) -> Dict[str, Any]:
        """Test a single query with the specified model."""
        db_name = query_data["db_id"]
        question = query_data["natural_query"]
        true_sql = query_data["sql"]
        true_path = query_data["path"]
        
        # Get schema
        schema_info = self.get_database_schema(db_name, true_path)
        
        # Create prompt
        prompt = self.create_prompt(question, schema_info, true_path, true_sql)
        
        # Call API
        start_time = time.time()
        response = self.call_model(prompt, model_name)
        api_time = time.time() - start_time
        
        result = {
            "db_id": db_name,
            "question": question,
            "true_sql": true_sql,
            "true_path": true_path,
            "model": MODELS[model_name],
            "api_time": api_time,
            "success": False,
            "exact_match": False,
            "path_match": False,
            "generated_sql": None,
            "error": None
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
            
            # Evaluate exact match
            gen_clean = generated_sql.strip().replace('\n', ' ').replace('  ', ' ').upper()
            true_clean = true_sql.strip().replace('\n', ' ').replace('  ', ' ').upper()
            result["exact_match"] = gen_clean == true_clean
            
            # Evaluate path match
            gen_tables = self.extract_tables(generated_sql)
            expected_tables = [t.upper() for t in true_path]
            result["path_match"] = gen_tables == expected_tables
            
            # Store usage info
            if 'usage' in response:
                result["usage"] = response['usage']
        else:
            result["error"] = "API call failed"
        
        return result
    
    def run_comparison(self, hop_length: int = 5, num_queries: int = 5):
        """Run comparison between both models."""
        print(f"🔬 Qwen Model Comparison")
        print(f"🤖 Models: {MODELS['qwen2']} vs {MODELS['qwen3']}")
        print(f"📊 Testing {num_queries} queries from {hop_length}-hop")
        print("=" * 80)
        
        # Load test data
        batch_file = Path(f"final_data/{hop_length}_hop/batch_001.json")
        if not batch_file.exists():
            print(f"❌ Batch file not found: {batch_file}")
            return
        
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        # Get first N queries
        test_queries = batch_data[:num_queries]
        
        print(f"📋 Testing {len(test_queries)} queries from {batch_file}")
        print()
        
        # Test each query with both models
        for i, query_data in enumerate(test_queries):
            print(f"🧪 Query {i+1}/{len(test_queries)}: {query_data['db_id']}")
            print(f"❓ Question: {query_data['natural_query'][:100]}...")
            
            # Test with Qwen 2.5
            print(f"   🔄 Testing Qwen 2.5 72B...", end="")
            qwen2_result = self.test_single_query(query_data, "qwen2")
            self.results["qwen2"].append(qwen2_result)
            
            status2 = "✅" if qwen2_result["success"] else "❌"
            exact2 = "✅" if qwen2_result["exact_match"] else "❌"
            path2 = "✅" if qwen2_result["path_match"] else "❌"
            print(f" {status2} (Exact: {exact2}, Path: {path2}, Time: {qwen2_result['api_time']:.1f}s)")
            
            # Test with Qwen 3
            print(f"   🔄 Testing Qwen 3 32B...", end="")
            qwen3_result = self.test_single_query(query_data, "qwen3")
            self.results["qwen3"].append(qwen3_result)
            
            status3 = "✅" if qwen3_result["success"] else "❌"
            exact3 = "✅" if qwen3_result["exact_match"] else "❌"
            path3 = "✅" if qwen3_result["path_match"] else "❌"
            print(f" {status3} (Exact: {exact3}, Path: {path3}, Time: {qwen3_result['api_time']:.1f}s)")
            print()
        
        # Print comparison summary
        self.print_comparison_summary()
        
        # Save results
        self.save_results()
    
    def print_comparison_summary(self):
        """Print comparison summary."""
        print("=" * 80)
        print("📊 COMPARISON SUMMARY")
        print("=" * 80)
        
        # Calculate metrics for each model
        models = ["qwen2", "qwen3"]
        model_names = ["Qwen 2.5 72B", "Qwen 3 32B"]
        
        print(f"{'Model':<15} | {'Success':<7} | {'Exact':<5} | {'Path':<4} | {'Avg Time':<8}")
        print("-" * 15 + "-+-" + "-" * 7 + "-+-" + "-" * 5 + "-+-" + "-" * 4 + "-+-" + "-" * 8)
        
        for model_key, model_name in zip(models, model_names):
            results = self.results[model_key]
            
            success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
            exact_rate = sum(1 for r in results if r["exact_match"]) / len(results) * 100
            path_rate = sum(1 for r in results if r["path_match"]) / len(results) * 100
            avg_time = sum(r["api_time"] for r in results) / len(results)
            
            print(f"{model_name:<15} | {success_rate:>6.1f}% | {exact_rate:>4.1f}% | {path_rate:>3.1f}% | {avg_time:>7.1f}s")
        
        print()
        print("🎯 DETAILED COMPARISON:")
        
        # Query-by-query comparison
        for i in range(len(self.results["qwen2"])):
            r2 = self.results["qwen2"][i]
            r3 = self.results["qwen3"][i]
            
            print(f"   Query {i+1} ({r2['db_id']}):")
            print(f"      Qwen 2.5: Exact={r2['exact_match']}, Path={r2['path_match']}, Time={r2['api_time']:.1f}s")
            print(f"      Qwen 3:   Exact={r3['exact_match']}, Path={r3['path_match']}, Time={r3['api_time']:.1f}s")
            
            # Winner
            if r2['exact_match'] and not r3['exact_match']:
                print(f"      🏆 Winner: Qwen 2.5 (Exact match)")
            elif r3['exact_match'] and not r2['exact_match']:
                print(f"      🏆 Winner: Qwen 3 (Exact match)")
            elif r2['path_match'] and not r3['path_match']:
                print(f"      🏆 Winner: Qwen 2.5 (Path match)")
            elif r3['path_match'] and not r2['path_match']:
                print(f"      🏆 Winner: Qwen 3 (Path match)")
            elif r2['api_time'] < r3['api_time']:
                print(f"      🏆 Winner: Qwen 2.5 (Faster)")
            else:
                print(f"      🏆 Winner: Qwen 3 (Faster)")
            print()
    
    def save_results(self):
        """Save comparison results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"qwen_comparison_results_{timestamp}.json"
        
        comparison_data = {
            "metadata": {
                "timestamp": timestamp,
                "models_tested": MODELS,
                "total_queries": len(self.results["qwen2"]),
                "test_type": "model_comparison"
            },
            "results": self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Detailed results saved to: {output_file}")
        print("=" * 80)

def main():
    """Main function."""
    bird_db_path = "../bird/train/train_databases/train_databases"
    
    # Run comparison
    comparator = QwenModelComparison(bird_db_path)
    comparator.run_comparison(hop_length=5, num_queries=5)

if __name__ == "__main__":
    main() 