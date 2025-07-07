#!/usr/bin/env python3
"""
Inspect Database Connection Graph Structure
"""

import pickle
from pathlib import Path
import json

def inspect_graph(db_name):
    """Inspect the structure of a connection graph."""
    graph_file = Path(f"connection_graphs/{db_name}_full_graph.pkl")
    
    if not graph_file.exists():
        print(f"Graph file not found: {graph_file}")
        return None
    
    try:
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        print(f"\n=== {db_name.upper()} GRAPH STRUCTURE ===")
        print(f"Type: {type(graph_data)}")
        
        if isinstance(graph_data, dict):
            print(f"Keys: {list(graph_data.keys())}")
            
            # Show database info
            if 'db_name' in graph_data:
                print(f"Database: {graph_data['db_name']}")
            if 'tables' in graph_data:
                print(f"Tables: {graph_data['tables']}")
            if 'columns' in graph_data:
                print(f"Columns: {graph_data['columns']}")
            
            # Show the actual connections
            if 'connections' in graph_data:
                connections = graph_data['connections']
                print(f"Connections type: {type(connections)}")
                
                if isinstance(connections, dict):
                    print(f"Number of connection nodes: {len(connections)}")
                    print("Connection structure:")
                    for i, (source, targets) in enumerate(connections.items()):
                        if i >= 5:  # Show first 5
                            print("  ...")
                            break
                        print(f"  {source} -> {targets}")
                        
            # Show schema if available
            if 'schema' in graph_data:
                schema = graph_data['schema']
                print(f"Schema type: {type(schema)}")
                if isinstance(schema, dict):
                    print(f"Schema tables: {list(schema.keys())[:5]}")  # First 5 tables
                
        return graph_data
        
    except Exception as e:
        print(f"Error loading graph: {e}")
        return None

def main():
    """Inspect a few different graph structures."""
    databases = ["restaurant", "university", "movie", "airline"]
    
    for db_name in databases:
        inspect_graph(db_name)

if __name__ == "__main__":
    main() 