#!/usr/bin/env python3
"""
Comprehensive SQL Evaluator
Compares generated SQL queries to ground truth across all generated query batches.
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import sqlite3
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    SQL_QUERIES_DIR, FINAL_DATA_DIR, EVALUATION_RESULTS_DIR,
    BIRD_DB_PATH, get_database_path, get_hop_data_dir,
    get_batch_file_path, validate_config
)

# Validate configuration
validate_config()

class ComprehensiveSQLEvaluator:
    """Comprehensive SQL evaluation comparing generated queries to ground truth."""
    
    def __init__(self):
        self.sql_queries_dir = SQL_QUERIES_DIR
        self.final_data_dir = FINAL_DATA_DIR
        self.results_dir = EVALUATION_RESULTS_DIR
        
        # Database directory for execution testing
        self.db_dir = Path(BIRD_DB_PATH)
        
        # Metrics
        self.total_queries = 0
        self.exact_matches = 0
        self.path_matches = 0
        self.syntax_valid = 0
        self.execution_successful = 0
        
        # Detailed results
        self.detailed_results = []
        self.error_analysis = defaultdict(list)
        
    def normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison."""
        if not sql:
            return ""
        
        # Remove extra whitespace and normalize
        normalized = re.sub(r'\s+', ' ', sql.strip())
        normalized = normalized.replace('\n', ' ').replace('\t', ' ')
        
        # Remove trailing semicolon for comparison
        if normalized.endswith(';'):
            normalized = normalized[:-1]
        
        return normalized.upper()
    
    def extract_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL query in order."""
        if not sql or not sql.strip():
            return []
        
        sql_upper = sql.upper()
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
    
    def check_sql_syntax(self, sql: str) -> Tuple[bool, str]:
        """Check if SQL has valid syntax using sqlite3."""
        try:
            # Create in-memory database for syntax check
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            
            # Try to explain the query (this checks syntax without execution)
            cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
            conn.close()
            return True, "Valid syntax"
        except Exception as e:
            return False, str(e)
    
    def execute_sql_query(self, sql: str, db_path: Path) -> Tuple[bool, str, int]:
        """Execute SQL query against database."""
        try:
            if not db_path.exists():
                return False, "Database file not found", 0
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            
            return True, "Execution successful", len(results)
        except Exception as e:
            return False, str(e), 0
    
    def compare_queries(self, generated_sql: str, ground_truth_sql: str, 
                       expected_path: List[str]) -> Dict[str, Any]:
        """Compare generated SQL to ground truth."""
        # Normalize SQLs
        gen_normalized = self.normalize_sql(generated_sql)
        truth_normalized = self.normalize_sql(ground_truth_sql)
        
        # Check exact match
        exact_match = gen_normalized == truth_normalized
        
        # Check path match
        gen_tables = self.extract_tables(generated_sql)
        expected_tables = [t.upper() for t in expected_path]
        path_match = gen_tables == expected_tables
        
        # Check join count
        gen_joins = len(re.findall(r'JOIN', generated_sql.upper()))
        truth_joins = len(re.findall(r'JOIN', ground_truth_sql.upper()))
        join_count_match = gen_joins == truth_joins
        
        # Check syntax
        syntax_valid, syntax_error = self.check_sql_syntax(generated_sql)
        
        return {
            "exact_match": exact_match,
            "path_match": path_match,
            "join_count_match": join_count_match,
            "syntax_valid": syntax_valid,
            "syntax_error": syntax_error if not syntax_valid else None,
            "generated_tables": gen_tables,
            "expected_tables": expected_tables,
            "generated_joins": gen_joins,
            "expected_joins": truth_joins,
            "generated_sql_normalized": gen_normalized,
            "ground_truth_normalized": truth_normalized
        }
    
    def load_ground_truth_batch(self, hop_count: int, batch_num: int) -> List[Dict]:
        """Load ground truth queries from batch file."""
        batch_file = get_batch_file_path(hop_count, batch_num, "sql_queries")
        
        if not batch_file.exists():
            return []
        
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading {batch_file}: {e}")
            return []
    
    def load_generated_batch(self, hop_count: int, batch_num: int) -> List[Dict]:
        """Load generated queries with natural language from final_data."""
        batch_file = get_batch_file_path(hop_count, batch_num, "final_data")
        
        if not batch_file.exists():
            return []
        
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading {batch_file}: {e}")
            return []
    
    def evaluate_batch(self, hop_count: int, batch_num: int) -> Dict[str, Any]:
        """Evaluate a single batch comparing ground truth to generated queries."""
        print(f"📊 Evaluating {hop_count}-hop batch {batch_num:03d}...")
        
        # Load ground truth and generated queries
        ground_truth_queries = self.load_ground_truth_batch(hop_count, batch_num)
        generated_queries = self.load_generated_batch(hop_count, batch_num)
        
        if not ground_truth_queries:
            print(f"   ❌ No ground truth queries found")
            return {"error": "No ground truth queries"}
        
        if not generated_queries:
            print(f"   ❌ No generated queries found")
            return {"error": "No generated queries"}
        
        # Match queries by index (assuming same order)
        batch_results = []
        batch_metrics = {
            "total": 0,
            "exact_matches": 0,
            "path_matches": 0,
            "syntax_valid": 0,
            "execution_successful": 0
        }
        
        min_length = min(len(ground_truth_queries), len(generated_queries))
        
        for i in range(min_length):
            ground_truth = ground_truth_queries[i]
            generated = generated_queries[i]
            
            # Compare the queries
            comparison = self.compare_queries(
                generated.get("sql", ""),
                ground_truth.get("sql", ""),
                ground_truth.get("path", [])
            )
            
            # Test execution if database exists
            execution_result = {"success": False, "error": "Not tested", "result_count": 0}
            db_name = ground_truth.get("db_id", "")
            if db_name:
                db_path = self.db_dir / db_name / f"{db_name}.sqlite"
                if db_path.exists():
                    exec_success, exec_error, result_count = self.execute_sql_query(
                        generated.get("sql", ""), db_path
                    )
                    execution_result = {
                        "success": exec_success,
                        "error": exec_error if not exec_success else None,
                        "result_count": result_count
                    }
            
            # Create detailed result
            query_result = {
                "query_index": i,
                "db_id": ground_truth.get("db_id", ""),
                "path": ground_truth.get("path", []),
                "hop_count": hop_count,
                "batch_num": batch_num,
                "natural_query": generated.get("natural_query", ""),
                "ground_truth_sql": ground_truth.get("sql", ""),
                "generated_sql": generated.get("sql", ""),
                "comparison": comparison,
                "execution": execution_result
            }
            
            batch_results.append(query_result)
            
            # Update metrics
            batch_metrics["total"] += 1
            if comparison["exact_match"]:
                batch_metrics["exact_matches"] += 1
            if comparison["path_match"]:
                batch_metrics["path_matches"] += 1
            if comparison["syntax_valid"]:
                batch_metrics["syntax_valid"] += 1
            if execution_result["success"]:
                batch_metrics["execution_successful"] += 1
        
        # Calculate percentages
        total = batch_metrics["total"]
        if total > 0:
            batch_metrics["exact_match_rate"] = (batch_metrics["exact_matches"] / total) * 100
            batch_metrics["path_match_rate"] = (batch_metrics["path_matches"] / total) * 100
            batch_metrics["syntax_valid_rate"] = (batch_metrics["syntax_valid"] / total) * 100
            batch_metrics["execution_success_rate"] = (batch_metrics["execution_successful"] / total) * 100
        
        print(f"   ✅ Evaluated {total} queries")
        print(f"      Exact matches: {batch_metrics['exact_matches']}/{total} ({batch_metrics.get('exact_match_rate', 0):.1f}%)")
        print(f"      Path matches: {batch_metrics['path_matches']}/{total} ({batch_metrics.get('path_match_rate', 0):.1f}%)")
        print(f"      Syntax valid: {batch_metrics['syntax_valid']}/{total} ({batch_metrics.get('syntax_valid_rate', 0):.1f}%)")
        
        return {
            "hop_count": hop_count,
            "batch_num": batch_num,
            "metrics": batch_metrics,
            "results": batch_results
        }
    
    def evaluate_hop_directory(self, hop_count: int, max_batches: int = None) -> Dict[str, Any]:
        """Evaluate all batches in a hop directory."""
        print(f"\n🔍 Evaluating {hop_count}-hop queries...")
        
        hop_dir = get_hop_data_dir(hop_count, "sql_queries")
        if not hop_dir.exists():
            print(f"❌ Directory not found: {hop_dir}")
            return {"error": f"Directory not found: {hop_dir}"}
        
        # Find all batch files
        batch_files = sorted(hop_dir.glob("batch_*.json"))
        if max_batches:
            batch_files = batch_files[:max_batches]
        
        print(f"📁 Found {len(batch_files)} batch files")
        
        hop_results = []
        hop_metrics = {
            "total": 0,
            "exact_matches": 0,
            "path_matches": 0,
            "syntax_valid": 0,
            "execution_successful": 0
        }
        
        for batch_file in batch_files:
            batch_num = int(batch_file.stem.split('_')[1])
            batch_result = self.evaluate_batch(hop_count, batch_num)
            
            if "error" not in batch_result:
                hop_results.append(batch_result)
                
                # Aggregate metrics
                batch_metrics = batch_result["metrics"]
                hop_metrics["total"] += batch_metrics["total"]
                hop_metrics["exact_matches"] += batch_metrics["exact_matches"]
                hop_metrics["path_matches"] += batch_metrics["path_matches"]
                hop_metrics["syntax_valid"] += batch_metrics["syntax_valid"]
                hop_metrics["execution_successful"] += batch_metrics["execution_successful"]
        
        # Calculate overall percentages
        total = hop_metrics["total"]
        if total > 0:
            hop_metrics["exact_match_rate"] = (hop_metrics["exact_matches"] / total) * 100
            hop_metrics["path_match_rate"] = (hop_metrics["path_matches"] / total) * 100
            hop_metrics["syntax_valid_rate"] = (hop_metrics["syntax_valid"] / total) * 100
            hop_metrics["execution_success_rate"] = (hop_metrics["execution_successful"] / total) * 100
        
        print(f"\n📊 {hop_count}-hop Summary:")
        print(f"   Total queries: {total}")
        print(f"   Exact matches: {hop_metrics['exact_matches']} ({hop_metrics.get('exact_match_rate', 0):.1f}%)")
        print(f"   Path matches: {hop_metrics['path_matches']} ({hop_metrics.get('path_match_rate', 0):.1f}%)")
        print(f"   Syntax valid: {hop_metrics['syntax_valid']} ({hop_metrics.get('syntax_valid_rate', 0):.1f}%)")
        print(f"   Execution successful: {hop_metrics['execution_successful']} ({hop_metrics.get('execution_success_rate', 0):.1f}%)")
        
        return {
            "hop_count": hop_count,
            "total_batches": len(hop_results),
            "metrics": hop_metrics,
            "batch_results": hop_results
        }
    
    def evaluate_all_hops(self, hop_counts: List[int] = None, max_batches_per_hop: int = None):
        """Evaluate all specified hop counts."""
        if hop_counts is None:
            # Find all available hop directories
            hop_counts = []
            for hop_dir in sorted(SQL_QUERIES_DIR.iterdir()):
                if hop_dir.is_dir() and hop_dir.name.endswith('_hop'):
                    hop_count = int(hop_dir.name.split('_')[0])
                    hop_counts.append(hop_count)
        
        print(f"🚀 Comprehensive SQL Evaluation")
        print(f"📁 Evaluating hop counts: {hop_counts}")
        print(f"📊 Max batches per hop: {max_batches_per_hop or 'All'}")
        print("=" * 80)
        
        all_results = []
        overall_metrics = {
            "total": 0,
            "exact_matches": 0,
            "path_matches": 0,
            "syntax_valid": 0,
            "execution_successful": 0
        }
        
        for hop_count in hop_counts:
            hop_result = self.evaluate_hop_directory(hop_count, max_batches_per_hop)
            
            if "error" not in hop_result:
                all_results.append(hop_result)
                
                # Aggregate overall metrics
                hop_metrics = hop_result["metrics"]
                overall_metrics["total"] += hop_metrics["total"]
                overall_metrics["exact_matches"] += hop_metrics["exact_matches"]
                overall_metrics["path_matches"] += hop_metrics["path_matches"]
                overall_metrics["syntax_valid"] += hop_metrics["syntax_valid"]
                overall_metrics["execution_successful"] += hop_metrics["execution_successful"]
        
        # Calculate overall percentages
        total = overall_metrics["total"]
        if total > 0:
            overall_metrics["exact_match_rate"] = (overall_metrics["exact_matches"] / total) * 100
            overall_metrics["path_match_rate"] = (overall_metrics["path_matches"] / total) * 100
            overall_metrics["syntax_valid_rate"] = (overall_metrics["syntax_valid"] / total) * 100
            overall_metrics["execution_success_rate"] = (overall_metrics["execution_successful"] / total) * 100
        
        # Print overall summary
        print(f"\n" + "=" * 80)
        print(f"🎯 OVERALL EVALUATION SUMMARY")
        print(f"=" * 80)
        print(f"Total queries evaluated: {total:,}")
        print(f"Exact matches: {overall_metrics['exact_matches']:,} ({overall_metrics.get('exact_match_rate', 0):.1f}%)")
        print(f"Path matches: {overall_metrics['path_matches']:,} ({overall_metrics.get('path_match_rate', 0):.1f}%)")
        print(f"Syntax valid: {overall_metrics['syntax_valid']:,} ({overall_metrics.get('syntax_valid_rate', 0):.1f}%)")
        print(f"Execution successful: {overall_metrics['execution_successful']:,} ({overall_metrics.get('execution_success_rate', 0):.1f}%)")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"comprehensive_evaluation_{timestamp}.json"
        
        final_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "hop_counts_evaluated": hop_counts,
            "max_batches_per_hop": max_batches_per_hop,
            "overall_metrics": overall_metrics,
            "hop_results": all_results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Comprehensive results saved to: {results_file}")
        
        return final_results
    
    def generate_error_analysis(self, results: Dict) -> Dict:
        """Generate detailed error analysis from evaluation results."""
        error_analysis = {
            "syntax_errors": defaultdict(int),
            "path_mismatches": [],
            "execution_errors": defaultdict(int),
            "common_issues": []
        }
        
        for hop_result in results.get("hop_results", []):
            for batch_result in hop_result.get("batch_results", []):
                for query_result in batch_result.get("results", []):
                    comparison = query_result.get("comparison", {})
                    execution = query_result.get("execution", {})
                    
                    # Syntax errors
                    if not comparison.get("syntax_valid"):
                        error_type = comparison.get("syntax_error", "Unknown syntax error")
                        error_analysis["syntax_errors"][error_type] += 1
                    
                    # Path mismatches
                    if not comparison.get("path_match"):
                        error_analysis["path_mismatches"].append({
                            "db_id": query_result.get("db_id"),
                            "expected_path": comparison.get("expected_tables"),
                            "generated_path": comparison.get("generated_tables"),
                            "hop_count": query_result.get("hop_count")
                        })
                    
                    # Execution errors
                    if not execution.get("success"):
                        error_type = execution.get("error", "Unknown execution error")
                        error_analysis["execution_errors"][error_type] += 1
        
        return error_analysis

def main():
    """Main function to run comprehensive evaluation."""
    evaluator = ComprehensiveSQLEvaluator()
    
    print("🔍 Comprehensive SQL Query Evaluator")
    print("=" * 60)
    
    # Show available options
    print("\nOptions:")
    print("1. Evaluate specific hop count (e.g., 2-hop, 3-hop)")
    print("2. Evaluate multiple hop counts")
    print("3. Evaluate all available hop counts")
    print("4. Quick test (first 5 batches of 2-hop and 3-hop)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        hop_count = int(input("Enter hop count: "))
        max_batches = input("Max batches to evaluate (press Enter for all): ").strip()
        max_batches = int(max_batches) if max_batches else None
        
        evaluator.evaluate_hop_directory(hop_count, max_batches)
    
    elif choice == "2":
        hop_counts_input = input("Enter hop counts (comma-separated, e.g., 2,3,5): ").strip()
        hop_counts = [int(x.strip()) for x in hop_counts_input.split(',')]
        max_batches = input("Max batches per hop (press Enter for all): ").strip()
        max_batches = int(max_batches) if max_batches else None
        
        evaluator.evaluate_all_hops(hop_counts, max_batches)
    
    elif choice == "3":
        max_batches = input("Max batches per hop (press Enter for all): ").strip()
        max_batches = int(max_batches) if max_batches else None
        
        evaluator.evaluate_all_hops(max_batches_per_hop=max_batches)
    
    elif choice == "4":
        print("🧪 Quick test mode: Evaluating first 5 batches of 2-hop and 3-hop")
        evaluator.evaluate_all_hops([2, 3], 5)
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 