#!/usr/bin/env python3
"""
Simple setup script for Connect Dots Multi-Hop SQL Query Generation System
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print("✅ Python version:", sys.version.split()[0])
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['requests', 'python-dotenv']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        print("Or run: pip install -r requirements.txt")
        return False
    
    return True

def check_environment():
    """Check if environment variables are set."""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found")
        print("📝 Create .env file with:")
        print("OPENROUTER_API_KEY=your_api_key_here")
        print("OPENAI_API_BASE=https://openrouter.ai/api/v1")
        return False
    
    print("✅ .env file found")
    return True

def check_bird_dataset():
    """Check if BIRD dataset is available."""
    bird_path = Path('../bird/train/train_databases/train_databases')
    if not bird_path.exists():
        print("❌ BIRD dataset not found at:", bird_path)
        print("📥 Download BIRD dataset and extract to ../bird/")
        return False
    
    print("✅ BIRD dataset found")
    return True

def main():
    """Main setup check."""
    print("🔧 Connect Dots Setup Check")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment", check_environment),
        ("BIRD Dataset", check_bird_dataset),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All checks passed! Ready to use Connect Dots.")
        print("\n🚀 Quick Start:")
        print("1. python generate_full_connection_graphs.py")
        print("2. python large_scale_generator.py")
        print("3. python natural_query_generator.py")
        print("4. python combine_final_data.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 