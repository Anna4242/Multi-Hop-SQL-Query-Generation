#!/usr/bin/env python3
"""
Initialize Git Repository for Connect Dots Project
Helps set up the project for pushing to GitHub
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and print the result."""
    print(f"\n🔧 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
        else:
            print(f"❌ Error: {description}")
            print(f"Error: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def check_git_installed():
    """Check if Git is installed."""
    return run_command("git --version", "Checking Git installation")

def initialize_git_repo():
    """Initialize Git repository."""
    commands = [
        ("git init", "Initializing Git repository"),
        ("git add .", "Adding all files to Git"),
        ("git commit -m 'Initial commit: Organized Connect Dots project structure'", "Creating initial commit"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True

def show_next_steps():
    """Show next steps for GitHub setup."""
    print("\n" + "="*80)
    print("🚀 GIT REPOSITORY INITIALIZED SUCCESSFULLY!")
    print("="*80)
    
    print("\n📋 NEXT STEPS TO PUSH TO GITHUB:")
    print("\n1. Create a new repository on GitHub:")
    print("   - Go to https://github.com/new")
    print("   - Repository name: connect-dots-sql-multihop")
    print("   - Description: Multi-Hop SQL Query Generation & Evaluation System")
    print("   - Make it Public or Private (your choice)")
    print("   - DON'T initialize with README (we already have one)")
    
    print("\n2. Connect your local repository to GitHub:")
    print("   git remote add origin https://github.com/YOUR_USERNAME/connect-dots-sql-multihop.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    
    print("\n3. Alternative: Use GitHub CLI (if installed):")
    print("   gh repo create connect-dots-sql-multihop --public --source=. --remote=origin --push")
    
    print("\n📁 WHAT WILL BE PUSHED:")
    print("   ✅ All Python scripts (organized in folders)")
    print("   ✅ README.md (comprehensive documentation)")
    print("   ✅ requirements.txt (dependencies)")
    print("   ✅ setup.py (package setup)")
    print("   ✅ .gitignore (excludes data directories)")
    print("   ❌ data/ (excluded - contains large files)")
    print("   ❌ logs/ (excluded - contains log files)")
    print("   ❌ results/ (excluded - contains result files)")
    
    print("\n🔧 FOLDER STRUCTURE:")
    print("   📁 ground_truth_generation/  - Core ground truth generation")
    print("   📁 sql_generation/          - AI-powered SQL generation")
    print("   📁 evaluation_testing/      - Model evaluation and testing")
    print("   📁 visualization/           - Schema visualization")
    print("   📁 data/                    - Data files (excluded)")
    print("   📁 logs/                    - Log files (excluded)")
    print("   📁 results/                 - Result files (excluded)")
    
    print("\n💡 TIPS:")
    print("   - The .gitignore file excludes large data files")
    print("   - Users will need to generate their own data using the scripts")
    print("   - Make sure to add your OpenRouter API key to .env file")
    print("   - The repository is ready for collaborative development")
    
    print("\n" + "="*80)

def main():
    """Main function."""
    print("🎯 Connect Dots - Git Repository Initialization")
    print("="*50)
    
    # Check if Git is installed
    if not check_git_installed():
        print("\n❌ Git is not installed or not in PATH")
        print("Please install Git first: https://git-scm.com/downloads")
        return
    
    # Check if already a Git repository
    if Path(".git").exists():
        print("\n⚠️  This directory is already a Git repository")
        print("If you want to reinitialize, delete the .git folder first")
        return
    
    # Initialize Git repository
    if initialize_git_repo():
        show_next_steps()
    else:
        print("\n❌ Failed to initialize Git repository")
        print("Please check the error messages above")

if __name__ == "__main__":
    main() 