#!/usr/bin/env python3
"""
SQL Generator using fully connected graphs - generates path lengths 2-15
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import pickle
import random
import json

class SQLFromGraphGenerator:
    """Generate SQL queries using fully connected graphs."""
    
    def __init__(self, graph_file_path):
        """Initialize with a graph file."""
        self.graph_file_path = Path(graph_file_path)
        self.graph_data = self._load_graph()
        self.db_name = self.graph_data['db_name']
        
    def _load_graph(self):
        """Load the graph data from pickle file."""
        with open(self.graph_file_path, 'rb') as f:
            return pickle.load(f)
    
    def generate_random_path(self, path_length):
        """Generate a random path of specified length (2-15)."""
        if path_length < 2 or path_length > 15:
            raise ValueError("Path length must be between 2 and 15")
        
        tables = list(self.graph_data['schema'].keys())
        if len(tables) < 2:
            raise ValueError("Database must have at least 2 tables")
        
        # Generate random path
        path = []
        for i in range(path_length):
            # For first table or when we want variety, choose randomly
            if i == 0 or random.random() < 0.7:  # 70% chance to pick new table
                table = random.choice(tables)
            else:
                # 30% chance to revisit a previous table for interesting patterns
                table = random.choice(path)
            path.append(table)
        
        return path
    
    def generate_relationships_for_path(self, path):
        """Generate relationships dictionary for a given path."""
        relationships = {}
        
        for i in range(len(path) - 1):
            src_table = path[i]
            dst_table = path[i + 1]
            
            # Get available columns for each table
            src_columns = self.graph_data['schema'][src_table]
            dst_columns = self.graph_data['schema'][dst_table]
            
            # Randomly choose columns to connect
            src_col = random.choice(src_columns)
            dst_col = random.choice(dst_columns)
            
            # Determine if join should be optional (LEFT JOIN)
            optional = random.random() < 0.2  # 20% chance of optional join
            
            relationships[(src_table, dst_table)] = {
                'src_col': src_col,
                'dst_col': dst_col,
                'optional': optional
            }
        
        return relationships
    
    def generate_sql(self, path, relationships, join_type="INNER"):
        """Generate SQL according to the specification."""
        if len(path) < 2 or len(path) > 15:
            raise ValueError("Path length must be between 2 and 15")
        
        # Start building SQL
        sql_parts = ["SELECT *"]
        
        # FROM clause with first table
        first_table = path[0]
        sql_parts.append(f"FROM {self._quote_if_needed(first_table)} AS t0")
        
        # Generate JOIN clauses
        for i in range(len(path) - 1):
            src_table = path[i]
            dst_table = path[i + 1]
            
            # Get relationship info
            rel_key = (src_table, dst_table)
            if rel_key not in relationships:
                raise ValueError(f"No relationship found for {src_table} -> {dst_table}")
            
            rel = relationships[rel_key]
            src_col = rel['src_col']
            dst_col = rel['dst_col']
            optional = rel.get('optional', False)
            
            # Determine join type for this specific join
            if optional and join_type == "INNER":
                current_join_type = "LEFT JOIN"
            else:
                current_join_type = f"{join_type} JOIN"
            
            # Build join clause
            table_alias = f"t{i+1}"
            quoted_dst_table = self._quote_if_needed(dst_table)
            
            sql_parts.append(f"{current_join_type} {quoted_dst_table} AS {table_alias}")
            sql_parts.append(f"  ON t{i}.{self._quote_if_needed(src_col)} = t{i+1}.{self._quote_if_needed(dst_col)}")
        
        # Join all parts and add semicolon
        sql = "\n".join(sql_parts) + ";"
        return sql
    
    def _quote_if_needed(self, identifier):
        """Quote identifier only if necessary."""
        # Check if identifier needs quoting (has spaces, special chars, or is SQL keyword)
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
    
    def generate_random_query(self, path_length=None, join_type="INNER"):
        """Generate a complete random query."""
        if path_length is None:
            path_length = random.randint(2, 15)
        
        # Generate random path
        path = self.generate_random_path(path_length)
        
        # Generate relationships for the path
        relationships = self.generate_relationships_for_path(path)
        
        # Generate SQL
        sql = self.generate_sql(path, relationships, join_type)
        
        return {
            'dbid': self.db_name,
            'sql': sql,
            'path': path,
            'path_length': len(path),
            'join_type': join_type,
            'relationships': relationships
        }

def generate_queries_for_database(db_name, num_queries=10, path_lengths=None):
    """Generate multiple queries for a specific database."""
    print(f"🔧 Generating queries for database: {db_name}")
    
    # Load the graph file
    graph_file = Path("connection_graphs") / f"{db_name}_full_graph.pkl"
    if not graph_file.exists():
        print(f"❌ Graph file not found: {graph_file}")
        return []
    
    generator = SQLFromGraphGenerator(graph_file)
    queries = []
    
    # Default path lengths if not specified
    if path_lengths is None:
        path_lengths = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for i in range(num_queries):
        try:
            # Choose random path length and join type
            path_length = random.choice(path_lengths)
            join_type = random.choice(["INNER", "LEFT", "INNER", "INNER"])  # Bias toward INNER
            
            query = generator.generate_random_query(path_length, join_type)
            queries.append(query)
            
            print(f"   ✅ Query {i+1}: {path_length}-hop {join_type} join")
            
        except Exception as e:
            print(f"   ❌ Query {i+1} failed: {str(e)[:50]}...")
            continue
    
    return queries

def test_random_database_queries():
    """Test generating queries for random databases."""
    print("🎯 Testing SQL Generation from Graphs")
    print("=" * 50)
    
    # Find available graph files
    graph_dir = Path("connection_graphs")
    graph_files = list(graph_dir.glob("*_full_graph.pkl"))
    
    if not graph_files:
        print("❌ No graph files found!")
        return
    
    print(f"📁 Found {len(graph_files)} graph files")
    
    # Test with a few random databases
    test_databases = random.sample([f.stem.replace('_full_graph', '') for f in graph_files], 
                                  min(3, len(graph_files)))
    
    all_results = []
    
    for db_name in test_databases:
        print(f"\n🗄️ Testing database: {db_name}")
        queries = generate_queries_for_database(db_name, num_queries=5)
        
        # Show sample queries
        for i, query in enumerate(queries[:2], 1):  # Show first 2
            print(f"\n📝 Sample Query {i}:")
            print(f"   Path: {' -> '.join(query['path'])}")
            print(f"   SQL Preview: {query['sql'][:100]}...")
        
        all_results.extend(queries)
    
    # Output summary in JSON format
    print(f"\n📊 SUMMARY:")
    print(f"   Databases tested: {len(test_databases)}")
    print(f"   Total queries generated: {len(all_results)}")
    
    # Sample JSON output
    print(f"\n📋 Sample JSON Output:")
    if all_results:
        sample_output = {
            'database_count': len(test_databases),
            'total_queries': len(all_results),
            'sample_queries': [
                {
                    'dbid': q['dbid'],
                    'sql': q['sql'],
                    'path_length': q['path_length'],
                    'join_type': q['join_type']
                } for q in all_results[:3]  # First 3 queries
            ]
        }
        print(json.dumps(sample_output, indent=2))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Generate queries for specific database
        db_name = sys.argv[1]
        num_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        queries = generate_queries_for_database(db_name, num_queries)
        
        # Output as JSON
        print(json.dumps(queries, indent=2))
    else:
        # Test mode
        test_random_database_queries() 