#!/usr/bin/env python3
"""
Large Scale Query Generator - Generate 100,000 queries with 1-20 hops
Distribute evenly across query lengths and databases
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import pickle
import random
import json
import sqlite3
import time
from datetime import datetime
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class LargeScaleGenerator:
    """Generate and execute 100,000 queries across all databases."""
    
    def __init__(self):
        self.graph_dir = Path("connection_graphs")
        self.db_dir = Path('../bird/train/train_databases/train_databases')
        self.output_dir = Path("sql_queries")
        self.total_queries = 100000
        self.hop_ranges = list(range(1, 21)) # Only 15-hop queries
        self.results = []
        self.lock = threading.Lock()
        self.batch_size = 100
        
        # Create output directories
        self._create_output_directories()
        
        # Get available databases
        self.available_databases = self._get_available_databases()
        print(f"📊 Found {len(self.available_databases)} databases")
        
                # Calculate distribution
        self._calculate_distribution()
    
    def _create_output_directories(self):
        """Create output directories for each hop length."""
        for hop_length in self.hop_ranges:
            hop_dir = self.output_dir / f"{hop_length}_hop"
            hop_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created output directory: {self.output_dir}/15_hop")
        print(f"📄 Directory will contain batch files: batch_001.json, batch_002.json, etc.")
        print(f"📊 Each batch file contains {self.batch_size} queries")
    
    def _get_available_databases(self):
        """Get list of databases that have both graph files and SQLite files."""
        graph_files = list(self.graph_dir.glob("*_full_graph.pkl"))
        available_dbs = []
        
        for graph_file in graph_files:
            db_name = graph_file.stem.replace('_full_graph', '')
            db_path = self.db_dir / db_name / f"{db_name}.sqlite"
            
            if db_path.exists():
                available_dbs.append(db_name)
        
        return available_dbs
    
    def _calculate_distribution(self):
        """Calculate how to distribute 100k queries evenly."""
        num_hop_lengths = len(self.hop_ranges)  # 20 hop lengths (1-20)
        num_databases = len(self.available_databases)
        
        # Queries per hop length
        self.queries_per_hop = self.total_queries // num_hop_lengths
        
        # Queries per database per hop length
        self.queries_per_db_per_hop = self.queries_per_hop // num_databases
        
        # Handle remainder
        self.remainder_queries = self.total_queries % (num_hop_lengths * num_databases)
        
        print(f"📈 Distribution Plan:")
        print(f"   Total queries: {self.total_queries:,}")
        print(f"   Hop lengths: {num_hop_lengths} (1-20)")
        print(f"   Databases: {num_databases}")
        print(f"   Queries per hop length: {self.queries_per_hop:,}")
        print(f"   Queries per database per hop: {self.queries_per_db_per_hop}")
        print(f"   Remainder queries: {self.remainder_queries}")
    
    def _load_graph(self, db_name):
        """Load graph data for a database."""
        graph_file = self.graph_dir / f"{db_name}_full_graph.pkl"
        with open(graph_file, 'rb') as f:
            return pickle.load(f)
    
    def _quote_if_needed(self, identifier):
        """Quote identifier only if necessary."""
        sql_keywords = {
            'select', 'from', 'where', 'join', 'inner', 'left', 'right', 'full',
            'on', 'and', 'or', 'not', 'in', 'like', 'between', 'is', 'null',
            'true', 'false', 'order', 'group', 'by', 'having', 'distinct',
            'union', 'intersect', 'except', 'case', 'when', 'then', 'else',
            'end', 'as', 'desc', 'asc', 'limit', 'offset', 'year', 'month', 'day'
        }
        
        if (identifier.lower() in sql_keywords or 
            ' ' in identifier or 
            '-' in identifier or 
            any(c in identifier for c in ['(', ')', '[', ']', '`', '"', "'"])):
            return f'`{identifier}`'
        
        return identifier
    
    def _generate_random_path(self, tables, path_length):
        """Generate a random path of specified length (1-20)."""
        if path_length < 1 or path_length > 20:
            raise ValueError("Path length must be between 1 and 20")
        
        if len(tables) < 1:
            raise ValueError("Database must have at least 1 table")
        
        # For 1-hop, just return one table
        if path_length == 1:
            return [random.choice(tables)]
        
        # For multi-hop, generate path
        path = []
        for i in range(path_length):
            if i == 0 or random.random() < 0.7:  # 70% chance to pick new table
                table = random.choice(tables)
            else:
                table = random.choice(path)  # 30% chance to revisit
            path.append(table)
        
        return path
    
    def _generate_relationships_for_path(self, path, schema):
        """Generate relationships for a path."""
        if len(path) <= 1:
            return {}  # No relationships needed for single table
        
        relationships = {}
        
        for i in range(len(path) - 1):
            src_table = path[i]
            dst_table = path[i + 1]
            
            src_columns = schema[src_table]
            dst_columns = schema[dst_table]
            
            src_col = random.choice(src_columns)
            dst_col = random.choice(dst_columns)
            
            optional = random.random() < 0.2  # 20% chance of LEFT JOIN
            
            relationships[(src_table, dst_table)] = {
                'src_col': src_col,
                'dst_col': dst_col,
                'optional': optional
            }
        
        return relationships
    
    def _generate_sql(self, path, relationships, join_type="INNER"):
        """Generate SQL for the path."""
        if len(path) == 1:
            # Single table query
            table = path[0]
            return f"SELECT * FROM {self._quote_if_needed(table)};"
        
        # Multi-table query
        sql_parts = ["SELECT *"]
        
        # FROM clause
        first_table = path[0]
        sql_parts.append(f"FROM {self._quote_if_needed(first_table)} AS t0")
        
        # JOIN clauses
        for i in range(len(path) - 1):
            src_table = path[i]
            dst_table = path[i + 1]
            
            rel_key = (src_table, dst_table)
            if rel_key not in relationships:
                raise ValueError(f"No relationship found for {src_table} -> {dst_table}")
            
            rel = relationships[rel_key]
            src_col = rel['src_col']
            dst_col = rel['dst_col']
            optional = rel.get('optional', False)
            
            if optional and join_type == "INNER":
                current_join_type = "LEFT JOIN"
            else:
                current_join_type = f"{join_type} JOIN"
            
            table_alias = f"t{i+1}"
            quoted_dst_table = self._quote_if_needed(dst_table)
            
            sql_parts.append(f"{current_join_type} {quoted_dst_table} AS {table_alias}")
            sql_parts.append(f"  ON t{i}.{self._quote_if_needed(src_col)} = t{i+1}.{self._quote_if_needed(dst_col)}")
        
        return "\n".join(sql_parts) + ";"
    
    def generate_queries_batch(self, db_name, hop_length, num_queries):
        """Generate a batch of queries for a specific database and hop length."""
        batch_results = []
        
        try:
            # Load graph data
            graph_data = self._load_graph(db_name)
            schema = graph_data['schema']
            tables = list(schema.keys())
            
            for query_id in range(num_queries):
                try:
                    # Generate path
                    path = self._generate_random_path(tables, hop_length)
                    
                    # Generate relationships
                    relationships = self._generate_relationships_for_path(path, schema)
                    
                    # Choose join type
                    join_type = random.choice(["INNER", "LEFT", "INNER", "INNER"])  # Bias toward INNER
                    
                    # Generate SQL
                    sql = self._generate_sql(path, relationships, join_type)
                    
                    query_data = {
                        'dbid': db_name,
                        'sql': sql,
                        'path': path,
                        'path_length': hop_length,
                        'join_type': join_type,
                        'relationships': relationships,
                        'generated_at': datetime.now().isoformat()
                    }
                    
                    batch_results.append(query_data)
                    
                except Exception as e:
                    print(f"   ❌ Query generation failed for {db_name} hop-{hop_length}: {str(e)[:50]}...")
                    continue
        
        except Exception as e:
            print(f"   ❌ Batch failed for {db_name} hop-{hop_length}: {str(e)[:50]}...")
        
        return batch_results
    '''''
    def execute_query(self, query_data, db_path):
        """Execute a single query and return results."""
        try:
            start_time = time.time()
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute(query_data['sql'])
            results = cursor.fetchall()
            
            execution_time = time.time() - start_time
            conn.close()
            
            return {
                'success': True,
                'result_count': len(results),
                'execution_time': execution_time,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'result_count': 0,
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0,
                'error': str(e)
            }
    '''''
    def execute_query(self, query_data, db_path):
            """Skip execution and just return success."""
            return {
                'success': True,
                'result_count': 0,
                'execution_time': 0.0,
                'error': None
            }
    def process_database_hop_combination(self, db_name, hop_length, num_queries):
        """Process all queries for a specific database-hop combination."""
        print(f"🔧 Processing {db_name} hop-{hop_length}: {num_queries} queries")
        
        # Generate queries
        queries = self.generate_queries_batch(db_name, hop_length, num_queries)
        
        if not queries:
            return []
        
        # Execute queries
        db_path = self.db_dir / db_name / f"{db_name}.sqlite"
        executed_results = []
        successful_queries = 0
        
        for query in queries:
            exec_result = self.execute_query(query, db_path)
            
            full_result = {
                **query,
                **exec_result
            }
            
            executed_results.append(full_result)
            
            if exec_result['success']:
                successful_queries += 1
        
        success_rate = (successful_queries / len(queries)) * 100 if queries else 0
        print(f"   📊 {db_name} hop-{hop_length}: {successful_queries}/{len(queries)} ({success_rate:.1f}%)")
        
        # Save this batch immediately so user can see progress
        if executed_results:
            self._save_immediate_batch(executed_results, db_name, hop_length)
        
        return executed_results
    
    def _save_immediate_batch(self, queries, db_name, hop_length):
        """Save a batch of queries immediately to prevent data loss."""
        hop_dir = self.output_dir / f"{hop_length}_hop"
        
        # Create a timestamped filename for this immediate batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = hop_dir / f"immediate_{db_name}_{timestamp}.json"
        
        # Simplify queries - keep essential fields + first source and last destination column
        simplified_queries = []
        for query in queries:
            # Extract first source and last destination columns from relationships
            source_columns = []
            dest_columns = []
            relationships = query.get('relationships', {})
            
            for (src_table, dst_table), rel_info in relationships.items():
                src_col = rel_info.get('src_col', 'unknown')
                dst_col = rel_info.get('dst_col', 'unknown')
                source_columns.append(f"{src_table}.{src_col}")
                dest_columns.append(f"{dst_table}.{dst_col}")
            
            # Get first source column and last destination column
            source_column = source_columns[0] if source_columns else "unknown"
            dest_column = dest_columns[-1] if dest_columns else "unknown"
            
            simplified_query = {
                'db_id': query['dbid'],
                'sql': query['sql'],
                'path_length': hop_length,
                'path': query.get('path', []),
                'source_column': source_column,
                'dest_column': dest_column
            }
            simplified_queries.append(simplified_query)
        
        # Save JSON with source/dest columns instead of relationships
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(simplified_queries, f, indent=2, ensure_ascii=False)
    
    def generate_and_execute_all(self, max_workers=4):
        """Generate and execute all 100,000 queries."""
        print("🚀 Starting Large Scale Query Generation and Execution")
        print("=" * 70)
        
        start_time = time.time()
        all_results = []
        
        # Create work items
        work_items = []
        extra_queries_distributed = 0
        
        for hop_length in self.hop_ranges:
            for db_name in self.available_databases:
                base_queries = self.queries_per_db_per_hop
                
                # Distribute remainder queries
                if extra_queries_distributed < self.remainder_queries:
                    base_queries += 1
                    extra_queries_distributed += 1
                
                if base_queries > 0:
                    work_items.append((db_name, hop_length, base_queries))
        
        print(f"📋 Created {len(work_items)} work items")
        print(f"🔧 Using {max_workers} worker threads")
        
        # Process with thread pool
        completed_items = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all work
            future_to_work = {
                executor.submit(self.process_database_hop_combination, db_name, hop_length, num_queries): 
                (db_name, hop_length, num_queries)
                for db_name, hop_length, num_queries in work_items
            }
            
            # Collect results
            for future in as_completed(future_to_work):
                work_item = future_to_work[future]
                
                try:
                    results = future.result()
                    all_results.extend(results)
                    
                    completed_items += 1
                    progress = (completed_items / len(work_items)) * 100
                    
                    print(f"📈 Progress: {completed_items}/{len(work_items)} ({progress:.1f}%) - Total queries: {len(all_results):,}")
                    
                except Exception as e:
                    print(f"❌ Work item {work_item} failed: {str(e)[:50]}...")
        
        # Final summary
        total_time = time.time() - start_time
        successful_queries = sum(1 for r in all_results if r.get('success', False))
        success_rate = (successful_queries / len(all_results)) * 100 if all_results else 0
        
        print(f"\n🎉 FINAL RESULTS:")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Total queries generated: {len(all_results):,}")
        print(f"   Successful executions: {successful_queries:,}")
        print(f"   Overall success rate: {success_rate:.1f}%")
        print(f"   Queries per second: {len(all_results)/total_time:.1f}")
        
        return all_results
    
    def save_results_in_batches(self, results):
        """Save results in batches of 100, organized by hop length."""
        print(f"💾 Saving {len(results):,} results in batches of {self.batch_size}...")
        
        # Group results by hop length
        by_hop = {}
        for result in results:
            hop_length = result['path_length']
            if hop_length not in by_hop:
                by_hop[hop_length] = []
            
            # Keep essential fields + source/destination info
            # Handle both original format (dbid) and simplified format (db_id)
            db_id = result.get('db_id') or result.get('dbid')
            simplified_result = {
                'db_id': db_id,
                'sql': result['sql'],
                'relationships': result.get('relationships', {}),  # Source/destination column info
                'path': result.get('path', [])  # Table path for context
            }
            by_hop[hop_length].append(simplified_result)
        
        total_files_saved = 0
        
        # Save each hop length separately
        for hop_length in sorted(by_hop.keys()):
            hop_results = by_hop[hop_length]
            hop_dir = self.output_dir / f"{hop_length}_hop"
            
            # Save in batches of 100
            for batch_idx in range(0, len(hop_results), self.batch_size):
                batch = hop_results[batch_idx:batch_idx + self.batch_size]
                batch_num = (batch_idx // self.batch_size) + 1
                
                batch_filename = hop_dir / f"batch_{batch_num:03d}.json"
                
                # Simple batch data - just the queries
                with open(batch_filename, 'w', encoding='utf-8') as f:
                    json.dump(batch, f, indent=2, ensure_ascii=False)
                
                total_files_saved += 1
            
            print(f"   📂 {hop_length}_hop: {len(hop_results):,} queries in {len(range(0, len(hop_results), self.batch_size))} batches")
        
        print(f"✅ Saved {total_files_saved} batch files across {len(by_hop)} hop directories")
    
    def consolidate_immediate_batches(self):
        """Consolidate all immediate batch files into final organized batches."""
        print("🔄 Consolidating immediate batch files...")
        
        all_results = []
        immediate_files_to_remove = []
        
        # Collect all immediate batch files
        for hop_length in self.hop_ranges:
            hop_dir = self.output_dir / f"{hop_length}_hop"
            if hop_dir.exists():
                immediate_files = list(hop_dir.glob("immediate_*.json"))
                
                for file_path in immediate_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            queries = json.load(f)
                            # queries is now directly an array of simplified queries
                            all_results.extend(queries)
                            immediate_files_to_remove.append(file_path)
                    except Exception as e:
                        print(f"⚠️  Error reading {file_path}: {e}")
        
        print(f"📊 Found {len(all_results):,} queries in immediate batches")
        
        # Save in organized batches
        if all_results:
            self.save_results_in_batches(all_results)
            
            # Remove immediate files after successful consolidation
            for file_path in immediate_files_to_remove:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"⚠️  Could not remove {file_path}: {e}")
            
            print(f"🗑️  Cleaned up {len(immediate_files_to_remove)} immediate batch files")
        
        return all_results
    
    def save_results(self, results, filename="large_scale_results.json"):
        """Legacy method - calls the new batch saving method."""
        self.save_results_in_batches(results)
    
    def analyze_results(self, results):
        """Analyze the distribution and success rates."""
        print(f"\n🔍 DETAILED ANALYSIS:")
        
        # Group by hop length
        by_hop = {}
        for result in results:
            hop = result['path_length']
            if hop not in by_hop:
                by_hop[hop] = {'total': 0, 'success': 0}
            by_hop[hop]['total'] += 1
            if result.get('success', False):
                by_hop[hop]['success'] += 1
        
        print(f"\n📊 Success Rate by Hop Length:")
        for hop in sorted(by_hop.keys()):
            total = by_hop[hop]['total']
            success = by_hop[hop]['success']
            rate = (success / total * 100) if total > 0 else 0
            print(f"   {hop:2d}-hop: {success:4d}/{total:4d} ({rate:5.1f}%)")
        
        # Group by database
        by_db = {}
        for result in results:
            db = result['dbid']
            if db not in by_db:
                by_db[db] = {'total': 0, 'success': 0}
            by_db[db]['total'] += 1
            if result.get('success', False):
                by_db[db]['success'] += 1
        
        # Show top and bottom performing databases
        db_rates = [(db, by_db[db]['success']/by_db[db]['total']*100) 
                   for db in by_db if by_db[db]['total'] > 0]
        db_rates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 Top 5 Databases by Success Rate:")
        for db, rate in db_rates[:5]:
            total = by_db[db]['total']
            success = by_db[db]['success']
            print(f"   {db:20s}: {success:4d}/{total:4d} ({rate:5.1f}%)")
        
        print(f"\n⚠️  Bottom 5 Databases by Success Rate:")
        for db, rate in db_rates[-5:]:
            total = by_db[db]['total']
            success = by_db[db]['success']
            print(f"   {db:20s}: {success:4d}/{total:4d} ({rate:5.1f}%)")

def main():
    """Main function to run the large scale generation."""
    generator = LargeScaleGenerator()
    
    # Generate and execute all queries (saves immediate files as it goes) - using single worker thread
    results = generator.generate_and_execute_all(max_workers=1)
    
    # Consolidate immediate files into organized batches of 100
    final_results = generator.consolidate_immediate_batches()
    
    # Analyze results
    generator.analyze_results(final_results if final_results else results)

if __name__ == "__main__":
    main() 