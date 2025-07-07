#!/usr/bin/env python3
"""
Generate fully connected graphs - every column to every other column
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import sqlite3
import pickle

def get_database_schema(db_path):
    """Get all tables and columns from database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema = {}
    for (table_name,) in tables:
        # Get columns for each table
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns = cursor.fetchall()
        schema[table_name] = [col[1] for col in columns]  # col[1] is column name
    
    conn.close()
    return schema

def create_full_connection_graph(schema):
    """Create fully connected graph - every column connects to every other column."""
    
    # Get all columns across all tables
    all_columns = []
    for table_name, columns in schema.items():
        for column in columns:
            all_columns.append((table_name, column))
    
    # Create connections - every column to every other column
    connections = {}
    
    for i, (from_table, from_col) in enumerate(all_columns):
        from_key = f"{from_table}.{from_col}"
        connections[from_key] = []
        
        for j, (to_table, to_col) in enumerate(all_columns):
            if i != j:  # Don't connect to self
                to_key = f"{to_table}.{to_col}"
                connections[from_key].append(to_key)
    
    return connections, len(all_columns)

def generate_all_full_graphs():
    """Generate full connection graphs for all databases."""
    print("🔗 Generating FULL Connection Graphs (Every Column to Every Column)")
    print("=" * 70)
    
    # Create connection_graphs directory
    connection_graphs_dir = Path("connection_graphs")
    connection_graphs_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {connection_graphs_dir}")
    
    # Find all databases
    db_dir = Path('../bird/train/train_databases/train_databases')
    available_dbs = []
    
    if db_dir.exists():
        for db_folder in db_dir.iterdir():
            if db_folder.is_dir():
                sqlite_file = db_folder / f"{db_folder.name}.sqlite"
                if sqlite_file.exists():
                    available_dbs.append((db_folder.name, sqlite_file))
    
    if not available_dbs:
        print("❌ No databases found!")
        return
    
    print(f"🗄️ Found {len(available_dbs)} databases")
    
    # Process each database
    successful_graphs = 0
    total_connections = 0
    
    for i, (db_name, db_path) in enumerate(available_dbs, 1):
        print(f"\n🔄 Processing {i}/{len(available_dbs)}: {db_name}")
        
        try:
            # Get database schema
            schema = get_database_schema(db_path)
            table_count = len(schema)
            column_count = sum(len(cols) for cols in schema.values())
            
            print(f"   📊 Tables: {table_count}, Columns: {column_count}")
            
            # Create full connection graph
            connections, total_cols = create_full_connection_graph(schema)
            connection_count = sum(len(targets) for targets in connections.values())
            
            print(f"   🔗 Created {connection_count:,} connections")
            
            # Prepare minimal data to save
            graph_data = {
                'db_name': db_name,
                'tables': table_count,
                'columns': column_count,
                'connections': connections,
                'schema': schema
            }
            
            # Save as pickle file (more efficient than JSON)
            output_file = connection_graphs_dir / f"{db_name}_full_graph.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(graph_data, f)
            
            file_size = output_file.stat().st_size / 1024  # KB
            print(f"   💾 Saved: {output_file.name} ({file_size:.1f} KB)")
            
            successful_graphs += 1
            total_connections += connection_count
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}...")
            continue
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Databases processed: {successful_graphs}/{len(available_dbs)}")
    print(f"   Total connections created: {total_connections:,}")
    print(f"   Average connections per DB: {total_connections//successful_graphs:,}" if successful_graphs > 0 else "   Average: 0")
    print(f"   Success rate: {(successful_graphs/len(available_dbs)*100):.1f}%")
    
    # List generated files
    print(f"\n📋 Generated graph files:")
    for file in sorted(connection_graphs_dir.glob("*_full_graph.pkl")):
        file_size = file.stat().st_size / 1024  # KB
        print(f"   - {file.name} ({file_size:.1f} KB)")

def test_load_graph(db_name):
    """Test loading a graph file."""
    graph_file = Path("connection_graphs") / f"{db_name}_full_graph.pkl"
    
    if not graph_file.exists():
        print(f"❌ Graph file not found: {graph_file}")
        return
    
    with open(graph_file, 'rb') as f:
        graph_data = pickle.load(f)
    
    print(f"✅ Loaded {db_name} graph:")
    print(f"   Tables: {graph_data['tables']}")
    print(f"   Columns: {graph_data['columns']}")
    print(f"   Connections: {sum(len(targets) for targets in graph_data['connections'].values()):,}")
    
    # Show sample connections
    print(f"   Sample connections:")
    for i, (from_col, to_cols) in enumerate(graph_data['connections'].items()):
        if i >= 3:  # Show first 3
            break
        print(f"     {from_col} -> {len(to_cols)} targets")
        print(f"       First 3: {to_cols[:3]}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test loading a specific graph
        if len(sys.argv) > 2:
            test_load_graph(sys.argv[2])
        else:
            print("Usage: python generate_full_connection_graphs.py test <db_name>")
    else:
        generate_all_full_graphs() 