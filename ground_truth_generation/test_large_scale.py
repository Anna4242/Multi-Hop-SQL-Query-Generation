#!/usr/bin/env python3
"""
Test Large Scale Generator - Test with smaller numbers first
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from large_scale_generator import LargeScaleGenerator

class TestLargeScaleGenerator(LargeScaleGenerator):
    """Test version with smaller numbers."""
    
    def __init__(self):
        super().__init__()
        # Override with smaller numbers for testing
        self.total_queries = 200  # Test with 200 queries instead of 100k
        self.hop_ranges = list(range(1, 11))  # Test with 1-10 hops instead of 1-20
        
        # Recalculate distribution with new numbers
        self._calculate_distribution()

def test_small_scale():
    """Test the system with a small number of queries."""
    print("🧪 Testing Large Scale System with Small Numbers")
    print("=" * 60)
    
    generator = TestLargeScaleGenerator()
    
    # Generate and execute queries
    results = generator.generate_and_execute_all(max_workers=2)
    
    # Save results
    generator.save_results(results, "test_large_scale_results.json")
    
    # Analyze results
    generator.analyze_results(results)
    
    return results

if __name__ == "__main__":
    test_small_scale() 