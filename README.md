# Connect Dots: Multi-Hop SQL Query Generation & Evaluation System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/Anna4242/Multi-Hop-SQL-Query-Generation.svg)](https://github.com/Anna4242/Multi-Hop-SQL-Query-Generation/stargazers)

A comprehensive toolkit for generating complex multi-hop SQL queries, creating natural language descriptions, and evaluating SQL generation models across database schemas.

## Overview

This system provides end-to-end functionality for:
- **Ground Truth Generation**: Creating multi-hop SQL queries from database schemas
- **SQL Generation**: Using open source LLMs to generate SQL from natural language
- **Evaluation & Testing**: Comprehensive evaluation of SQL generation models
- **Visualization**: Database schema and connection graph visualization

## Quick Start

### Installation

#### Prerequisites
- Python 3.8 or higher
- Git
- OpenRouter API key (for open source LLM access)

#### Clone the Repository
```bash
git clone https://github.com/Anna4242/Multi-Hop-SQL-Query-Generation.git
cd Multi-Hop-SQL-Query-Generation
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Alternative: Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file in the project root:
```env
OPENROUTER_API_KEY=your_api_key_here
OPENAI_API_BASE=https://openrouter.ai/api/v1
```

**Get your OpenRouter API key:**
1. Visit [OpenRouter.ai](https://openrouter.ai/)
2. Sign up for an account
3. Navigate to API Keys section
4. Generate a new API key
5. Add it to your `.env` file

### Generate Ground Truth Data
```bash
cd ground_truth_generation
python generate_full_connection_graphs.py  # Generate connection graphs
python large_scale_generator.py            # Generate multi-hop queries
python natural_query_generator.py          # Generate natural language
python combine_final_data.py               # Combine results
```

### Generate SQL with Open Source LLMs
```bash
cd sql_generation
python simple_sql_generator.py             # Generate SQL using Qwen 2.5 72B
```

### Evaluate Results
```bash
cd evaluation_testing
python batch_evaluator.py                  # Evaluate generated SQL
python sql_generation_evaluator.py         # Comprehensive evaluation
```

### Visualize Database Schemas
```bash
cd visualization
python visualize_graphs.py                 # Generate visualizations
```

## Key Features

### Ground Truth Generation
- **Multi-Hop Traversal**: Generates queries spanning 2-20 table hops
- **Even Distribution**: Queries distributed across all available databases
- **Natural Language**: Generated human-readable questions using open source LLMs
- **Batch Processing**: Handles large-scale generation with immediate saves

### SQL Generation
- **Clean Generation**: No ground truth in prompts for fair evaluation
- **Multiple Models**: Support for various open source LLMs via OpenRouter
- **Batch Processing**: Efficient processing of large datasets
- **Format Preservation**: Maintains original data format with added fields

### Evaluation & Testing
- **Comprehensive Metrics**: Multiple evaluation approaches
- **Path Validation**: Ensures generated SQL follows required paths
- **Model Comparison**: Compare different open source LLMs
- **Syntax Validation**: Checks SQL syntax correctness

### Visualization
- **Network Graphs**: Visual representation of table relationships
- **Mermaid Diagrams**: Interactive schema diagrams
- **Multiple Formats**: PNG and Markdown outputs
- **Customizable**: Configurable node limits and styling

## Data Structure

### Input Data
- **BIRD Dataset**: Database schemas and descriptions
- **Connection Graphs**: Pre-computed table relationships

### Output Data
- **Ground Truth**: Multi-hop SQL queries with natural language
- **Generated SQL**: Generated SQL queries using open source LLMs
- **Evaluation Results**: Comprehensive evaluation metrics
- **Visualizations**: Schema diagrams and network graphs

## Configuration

### Model Configuration
- Default model: `qwen/qwen-2.5-72b-instruct`
- Configurable via model variables in scripts
- Support for multiple open source LLMs via OpenRouter API

### Generation Parameters
- Hop lengths: 2-20 (configurable)
- Batch sizes: 100 queries per batch
- Temperature: 0.1 for consistent results

## Contributing

1. Follow the existing code structure
2. Add comprehensive error handling
3. Include progress reporting for long operations
4. Update documentation for new features
5. Test with multiple databases before committing 