#!/usr/bin/env python3
"""
Quick validation of the large scale system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from large_scale_generator import LargeScaleGenerator

def quick_validation():
    """Quick validation with minimal queries."""
    print("⚡ Quick Validation Test")
    print("=" * 30)
    
    generator = LargeScaleGenerator()
    
    # Test just one database with a few queries
    db_name = generator.available_databases[0]  # First available database
    print(f"🧪 Testing database: {db_name}")
    
    # Test different hop lengths
    test_results = []
    for hop_length in [1, 2, 5, 10]:
        print(f"\n🔧 Testing {hop_length}-hop queries...")
        
        results = generator.process_database_hop_combination(db_name, hop_length, 2)
        test_results.extend(results)
        
        if results:
            success_count = sum(1 for r in results if r.get('success', False))
            print(f"   ✅ {success_count}/{len(results)} successful")
            
            # Show sample SQL
            if results:
                sample_sql = results[0]['sql'][:100] + "..." if len(results[0]['sql']) > 100 else results[0]['sql']
                print(f"   📝 Sample SQL: {sample_sql}")
    
    # Summary
    total_success = sum(1 for r in test_results if r.get('success', False))
    print(f"\n📊 Validation Summary:")
    print(f"   Total test queries: {len(test_results)}")
    print(f"   Successful queries: {total_success}")
    print(f"   Success rate: {total_success/len(test_results)*100:.1f}%")
    
    if total_success > 0:
        print("✅ Validation passed! System is working correctly.")
        return True
    else:
        print("❌ Validation failed! Check the system.")
        return False

if __name__ == "__main__":
    quick_validation() 