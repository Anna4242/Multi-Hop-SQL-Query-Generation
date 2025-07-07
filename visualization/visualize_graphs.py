#!/usr/bin/env python3
"""
Visualize Database Connection Graphs
Loads pickle files and creates network visualizations
"""

import pickle
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import sys

def load_graph(db_name):
    """Load connection graph from pickle file."""
    graph_file = Path(f"connection_graphs/{db_name}_full_graph.pkl")
    
    if not graph_file.exists():
        print(f"Graph file not found: {graph_file}")
        return None
    
    try:
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        return graph_data
    except Exception as e:
        print(f"Error loading graph: {e}")
        return None

def create_mermaid_diagram(graph_data, db_name):
    """Create a Mermaid diagram representation."""
    if not graph_data or 'connections' not in graph_data:
        return None
    
    connections = graph_data['connections']
    schema = graph_data.get('schema', {})
    
    # Create simplified connections (table to table)
    table_connections = set()
    
    for source_col, target_cols in connections.items():
        source_table = source_col.split('.')[0]
        for target_col in target_cols:
            target_table = target_col.split('.')[0]
            if source_table != target_table:
                table_connections.add((source_table, target_table))
    
    # Create Mermaid syntax
    mermaid_lines = ["graph TD"]
    
    # Add table nodes with their columns
    for table, columns in schema.items():
        col_list = "<br/>".join(columns[:3])  # Show first 3 columns
        if len(columns) > 3:
            col_list += f"<br/>... +{len(columns)-3} more"
        mermaid_lines.append(f'    {table}["{table}<br/>{col_list}"]')
    
    # Add connections
    for source, target in table_connections:
        mermaid_lines.append(f'    {source} --> {target}')
    
    return "\n".join(mermaid_lines)

def visualize_graph(graph_data, db_name, max_nodes=15):
    """Create a network visualization of the database connections."""
    if not graph_data or 'connections' not in graph_data:
        print(f"No connections data found for {db_name}")
        return
    
    connections = graph_data['connections']
    schema = graph_data.get('schema', {})
    
    # Create NetworkX graph at table level
    G = nx.Graph()
    
    # Add table nodes
    for table in schema.keys():
        G.add_node(table)
    
    # Add edges between tables based on column connections
    table_connections = set()
    
    for source_col, target_cols in connections.items():
        source_table = source_col.split('.')[0]
        for target_col in target_cols:
            target_table = target_col.split('.')[0]
            if source_table != target_table:
                table_connections.add((source_table, target_table))
    
    # Add edges to graph
    for source, target in table_connections:
        G.add_edge(source, target)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue', 
                          node_size=2000, 
                          alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray', 
                          alpha=0.6,
                          width=2)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, 
                           font_size=10, 
                           font_weight='bold')
    
    # Add table info as text
    info_text = f"Database: {db_name}\n"
    info_text += f"Tables: {len(schema)}\n"
    info_text += f"Total Columns: {graph_data.get('columns', 'N/A')}\n"
    info_text += f"Connections: {len(table_connections)}"
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.title(f"Database Schema: {db_name.upper()}", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Save the plot
    output_file = f"graph_visualization_{db_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph visualization saved as: {output_file}")
    
    plt.show()

def main():
    """Main function to visualize selected graphs."""
    # List of interesting databases to visualize
    databases = [
        "restaurant",     # Simple 3-table database
        "university",     # Academic database
        "movie",          # Entertainment database
        "airline",        # Transportation database
        "beer_factory",   # Manufacturing database
    ]
    
    print("Database Connection Graph Visualizer")
    print("=" * 50)
    
    for db_name in databases:
        print(f"\nProcessing: {db_name}")
        graph_data = load_graph(db_name)
        
        if graph_data:
            print(f"  Database: {graph_data.get('db_name', 'Unknown')}")
            print(f"  Tables: {graph_data.get('tables', 'Unknown')}")
            print(f"  Columns: {graph_data.get('columns', 'Unknown')}")
            
            # Create Mermaid diagram
            mermaid_code = create_mermaid_diagram(graph_data, db_name)
            if mermaid_code:
                print(f"  Mermaid diagram created")
                
                # Save Mermaid code to file
                with open(f"mermaid_{db_name}.md", "w") as f:
                    f.write(f"# {db_name.upper()} Database Schema\n\n")
                    f.write("```mermaid\n")
                    f.write(mermaid_code)
                    f.write("\n```\n")
                
            # Create matplotlib visualization
            visualize_graph(graph_data, db_name)
        else:
            print(f"  Failed to load graph for {db_name}")
    
    print("\nVisualization complete!")
    print("Files created:")
    print("  - PNG files: graph_visualization_*.png")
    print("  - Mermaid files: mermaid_*.md")

if __name__ == "__main__":
    main() 