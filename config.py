#!/usr/bin/env python3
"""
Configuration file for Connect Dots Multi-Hop SQL Query Generation
All paths and settings are defined here to avoid hardcoding
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data directories
SQL_QUERIES_DIR = DATA_DIR / "sql_queries"
FINAL_DATA_DIR = DATA_DIR / "final_data"
CONNECTION_GRAPHS_DIR = DATA_DIR / "connection_graphs"
GENERATED_QUERY_DIR = DATA_DIR / "generated_query"
GENERATED_QUERY_SIMPLE_DIR = DATA_DIR / "generated_query_simple"

# BIRD database path (configurable via environment variable)
BIRD_DB_PATH = os.getenv("BIRD_DB_PATH", "../bird/train/train_databases/train_databases")

# API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

# Model Configuration
DEFAULT_MODEL = "qwen/qwen-2.5-72b-instruct"
BACKUP_MODEL = "qwen/qwen-2.5-32b-instruct"

# Generation Configuration
DEFAULT_HOP_LENGTHS = [2, 3, 5]
MAX_HOP_LENGTH = 20
BATCH_SIZE = 100

# Rate limiting
REQUESTS_PER_MINUTE = 60
MIN_TIME_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE

# Output Configuration
OUTPUT_FORMATS = ["json", "csv"]
DEFAULT_OUTPUT_FORMAT = "json"

# Visualization Configuration
VISUALIZATION_DIR = PROJECT_ROOT / "visualization"
SCHEMA_IMAGES_DIR = VISUALIZATION_DIR / "schema_images"

# Evaluation Configuration
EVALUATION_RESULTS_DIR = RESULTS_DIR / "evaluation"
BATCH_EVALUATION_SIZE = 10

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        LOGS_DIR,
        RESULTS_DIR,
        SQL_QUERIES_DIR,
        FINAL_DATA_DIR,
        CONNECTION_GRAPHS_DIR,
        GENERATED_QUERY_DIR,
        GENERATED_QUERY_SIMPLE_DIR,
        VISUALIZATION_DIR,
        SCHEMA_IMAGES_DIR,
        EVALUATION_RESULTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def validate_config():
    """Validate configuration and environment variables."""
    errors = []
    
    # Check API key
    if not OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY environment variable is required")
    
    # Check BIRD database path
    bird_path = Path(BIRD_DB_PATH)
    if not bird_path.exists():
        errors.append(f"BIRD database path does not exist: {BIRD_DB_PATH}")
    
    if errors:
        raise RuntimeError("Configuration errors:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True

def get_database_path(db_name: str) -> Path:
    """Get the full path to a specific database."""
    return Path(BIRD_DB_PATH) / db_name / f"{db_name}.sqlite"

def get_database_description_path(db_name: str) -> Path:
    """Get the path to database description file."""
    return Path(BIRD_DB_PATH) / db_name / "database_description" / "database_description.txt"

def get_hop_data_dir(hop_count: int, data_type: str = "sql_queries") -> Path:
    """Get the directory for a specific hop count and data type."""
    if data_type == "sql_queries":
        return SQL_QUERIES_DIR / f"{hop_count}_hop"
    elif data_type == "final_data":
        return FINAL_DATA_DIR / f"{hop_count}_hop"
    elif data_type == "generated_query":
        return GENERATED_QUERY_DIR / f"{hop_count}_hop"
    elif data_type == "generated_query_simple":
        return GENERATED_QUERY_SIMPLE_DIR / f"{hop_count}_hop"
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def get_batch_file_path(hop_count: int, batch_num: int, data_type: str = "sql_queries") -> Path:
    """Get the path to a specific batch file."""
    hop_dir = get_hop_data_dir(hop_count, data_type)
    return hop_dir / f"batch_{batch_num:03d}.json"

def get_connection_graph_path(db_name: str) -> Path:
    """Get the path to a connection graph file."""
    return CONNECTION_GRAPHS_DIR / f"{db_name}_connections.pkl"

# Initialize directories when module is imported
ensure_directories() 