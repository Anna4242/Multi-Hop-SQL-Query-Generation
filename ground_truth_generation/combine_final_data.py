#!/usr/bin/env python3
"""
Combine all JSON files from final_data directory into a single CSV file
CSV columns: Index, Db, path_length, question, sql, path
"""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any

def collect_all_data(final_data_dir: Path) -> List[Dict[str, Any]]:
    """Collect all data from JSON files in hop directories."""
    all_data = []
    
    print("🔍 Scanning final_data directory...")
    
    # Find all hop directories
    hop_dirs = [d for d in final_data_dir.iterdir() 
                if d.is_dir() and d.name.endswith('_hop')]
    
    print(f"📂 Found {len(hop_dirs)} hop directories")
    
    for hop_dir in sorted(hop_dirs):
        hop_name = hop_dir.name
        print(f"\n📁 Processing {hop_name}...")
        
        # Find all batch JSON files in this hop directory
        batch_files = list(hop_dir.glob('batch_*.json'))
        print(f"   Found {len(batch_files)} batch files")
        
        batch_count = 0
        query_count = 0
        
        for batch_file in sorted(batch_files):
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    queries = json.load(f)
                
                for query in queries:
                    # Extract required fields
                    db_id = query.get('db_id', '')
                    sql = query.get('sql', '')
                    path = query.get('path', [])
                    natural_query = query.get('natural_query', '')
                    
                    # Calculate path length
                    path_length = len(path)
                    
                    # Convert path array to string representation
                    path_str = ' -> '.join(path) if path else ''
                    
                    # Add to collection
                    all_data.append({
                        'db_id': db_id,
                        'path_length': path_length,
                        'natural_query': natural_query,
                        'sql': sql,
                        'path': path_str,
                        'hop_dir': hop_name  # For reference
                    })
                    
                    query_count += 1
                
                batch_count += 1
                
            except Exception as e:
                print(f"   ❌ Error reading {batch_file.name}: {e}")
        
        print(f"   ✅ Processed {batch_count} batch files, {query_count} queries")
    
    return all_data

def save_to_csv(data: List[Dict[str, Any]], output_file: Path):
    """Save collected data to CSV file."""
    print(f"\n💾 Saving {len(data):,} records to {output_file}...")
    
    # Define CSV columns as requested
    fieldnames = ['Index', 'Db', 'path_length', 'question', 'sql', 'path']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data with sequential index
        for i, record in enumerate(data, 1):
            csv_row = {
                'Index': i,
                'Db': record['db_id'],
                'path_length': record['path_length'],
                'question': record['natural_query'],
                'sql': record['sql'].replace('\n', ' ').strip(),  # Clean up SQL formatting
                'path': record['path']
            }
            writer.writerow(csv_row)
    
    print(f"✅ CSV file saved successfully!")

def main():
    """Main function to combine all final_data into CSV."""
    print("🚀 Final Data Combiner")
    print("=" * 50)
    
    # Set up paths
    script_dir = Path(__file__).parent
    final_data_dir = script_dir / "final_data"
    output_file = script_dir / "combined_final_data.csv"
    
    # Check if final_data directory exists
    if not final_data_dir.exists():
        print(f"❌ final_data directory not found: {final_data_dir}")
        return
    
    print(f"📁 Input directory: {final_data_dir}")
    print(f"📄 Output file: {output_file}")
    
    # Collect all data from JSON files
    all_data = collect_all_data(final_data_dir)
    
    if not all_data:
        print("❌ No data found in final_data directory!")
        return
    
    # Group by database and hop length for summary
    summary = {}
    for record in all_data:
        db = record['db_id']
        hop_len = record['path_length']
        hop_dir = record['hop_dir']
        
        if db not in summary:
            summary[db] = {}
        if hop_len not in summary[db]:
            summary[db][hop_len] = {'count': 0, 'hop_dirs': set()}
        
        summary[db][hop_len]['count'] += 1
        summary[db][hop_len]['hop_dirs'].add(hop_dir)
    
    # Print summary
    print(f"\n📊 DATA SUMMARY:")
    print(f"   Total records: {len(all_data):,}")
    print(f"   Unique databases: {len(summary)}")
    
    # Show distribution by hop length
    hop_distribution = {}
    for record in all_data:
        hop_len = record['path_length']
        if hop_len not in hop_distribution:
            hop_distribution[hop_len] = 0
        hop_distribution[hop_len] += 1
    
    print(f"\n📈 Distribution by hop length:")
    for hop_len in sorted(hop_distribution.keys()):
        count = hop_distribution[hop_len]
        percentage = (count / len(all_data)) * 100
        print(f"   {hop_len:2d}-hop: {count:,} queries ({percentage:.1f}%)")
    
    # Show top databases
    db_counts = {}
    for record in all_data:
        db = record['db_id']
        if db not in db_counts:
            db_counts[db] = 0
        db_counts[db] += 1
    
    print(f"\n🏆 Top 10 databases by query count:")
    sorted_dbs = sorted(db_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (db, count) in enumerate(sorted_dbs[:10], 1):
        percentage = (count / len(all_data)) * 100
        print(f"   {i:2d}. {db:20s}: {count:,} queries ({percentage:.1f}%)")
    
    # Save to CSV
    save_to_csv(all_data, output_file)
    
    # Show file size
    file_size = output_file.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"\n📋 OUTPUT FILE INFO:")
    print(f"   File: {output_file}")
    print(f"   Size: {file_size_mb:.1f} MB ({file_size:,} bytes)")
    print(f"   Records: {len(all_data):,}")
    print(f"   Columns: {', '.join(['Index', 'Db', 'path_length', 'question', 'sql', 'path'])}")
    
    print(f"\n🎉 All done! Combined data saved to: {output_file}")

if __name__ == "__main__":
    main() 