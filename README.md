# Connect Dots: Multi-Hop SQL Query Generation & Evaluation System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/Anna4242/Multi-Hop-SQL-Query-Generation.svg)](https://github.com/Anna4242/Multi-Hop-SQL-Query-Generation/stargazers)

A comprehensive toolkit for generating complex multi-hop SQL queries, creating natural language descriptions, and evaluating SQL generation models across database schemas.

## 🎯 Overview

This system provides end-to-end functionality for:
- **Ground Truth Generation**: Creating multi-hop SQL queries from database schemas
- **SQL Generation**: Using AI models to generate SQL from natural language
- **Evaluation & Testing**: Comprehensive evaluation of SQL generation models
- **Visualization**: Database schema and connection graph visualization

## 📁 Project Structure

```
connect_dots/
├── ground_truth_generation/    # Core ground truth generation
├── sql_generation/            # AI-powered SQL generation
├── evaluation_testing/        # Model evaluation and testing
├── visualization/            # Schema and graph visualization
├── data/                     # Data files (excluded from git)
├── logs/                     # Log files (excluded from git)
├── results/                  # Result files (excluded from git)
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── .gitignore               # Git ignore rules
```

## 🚀 Quick Start

### 1. Installation

#### Prerequisites
- Python 3.8 or higher
- Git
- OpenRouter API key (for AI model access)

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

### 2. Environment Setup
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

### 3. Generate Ground Truth Data
```bash
cd ground_truth_generation
python generate_full_connection_graphs.py  # Generate connection graphs
python large_scale_generator.py            # Generate multi-hop queries
python natural_query_generator.py          # Generate natural language
python combine_final_data.py               # Combine results
```

### 4. Generate SQL with AI Models
```bash
cd sql_generation
python simple_sql_generator.py             # Generate SQL using Qwen 2.5 72B
```

### 5. Evaluate Results
```bash
cd evaluation_testing
python batch_evaluator.py                  # Evaluate generated SQL
python sql_generation_evaluator.py         # Comprehensive evaluation
```

### 6. Visualize Database Schemas
```bash
cd visualization
python visualize_graphs.py                 # Generate visualizations
```

## 🔧 Core Components

### 1. Ground Truth Generation (`ground_truth_generation/`)
- **Purpose**: Generate high-quality multi-hop SQL queries from database schemas
- **Key Files**:
  - `generate_full_connection_graphs.py` - Creates connection graphs from database schemas
  - `large_scale_generator.py` - Main bulk generator for multi-hop SQL queries
  - `sql_generator_from_graphs.py` - Core SQL generation logic
  - `natural_query_generator.py` - Converts SQL to natural language questions
  - `combine_final_data.py` - Combines JSON results into CSV format
  - `query_executor.py` - Executes and validates SQL queries
  - `quick_validation.py` - Validation utilities
  - `test_large_scale.py` - Testing utilities

### 2. SQL Generation (`sql_generation/`)
- **Purpose**: Generate SQL queries using AI models (without ground truth in prompts)
- **Key Files**:
  - `simple_sql_generator.py` - Clean SQL generation using Qwen 2.5 72B
  - `bulk_sql_generator.py` - Bulk SQL generation with verification

### 3. Evaluation & Testing (`evaluation_testing/`)
- **Purpose**: Comprehensive evaluation and testing of SQL generation models
- **Key Files**:
  - `batch_evaluator.py` - General batch evaluation
  - `batch_evaluator_qwen.py` - Qwen-specific batch evaluation
  - `batch_path_evaluator.py` - Path-guided evaluation
  - `path_guided_evaluator.py` - Path-guided SQL evaluation
  - `path_guided_sql_evaluator.py` - Advanced path-guided evaluation
  - `sql_generation_evaluator.py` - SQL generation evaluation
  - `qwen_model_comparison.py` - Model comparison utilities
  - `qwen*_test.py` - Various Qwen model tests
  - `logits_test*.py` - Logits analysis and testing
  - `quick_batch_test.py` - Quick batch testing

### 4. Visualization (`visualization/`)
- **Purpose**: Visualize database schemas and connection graphs
- **Key Files**:
  - `visualize_graphs.py` - Main visualization script
  - `inspect_graph.py` - Graph structure inspection
  - `simple_inspect.py` - Simple graph inspection utility
  - Output files: `graph_visualization_*.png`, `mermaid_*.md`

## 📊 Key Features

### Ground Truth Generation
- **Multi-Hop Traversal**: Generates queries spanning 2-20 table hops
- **Even Distribution**: Queries distributed across all available databases
- **Natural Language**: AI-generated human-readable questions
- **Batch Processing**: Handles large-scale generation with immediate saves

### SQL Generation
- **Clean Generation**: No ground truth in prompts for fair evaluation
- **Multiple Models**: Support for various AI models via OpenRouter
- **Batch Processing**: Efficient processing of large datasets
- **Format Preservation**: Maintains original data format with added fields

### Evaluation & Testing
- **Comprehensive Metrics**: Multiple evaluation approaches
- **Path Validation**: Ensures generated SQL follows required paths
- **Model Comparison**: Compare different AI models
- **Syntax Validation**: Checks SQL syntax correctness

### Visualization
- **Network Graphs**: Visual representation of table relationships
- **Mermaid Diagrams**: Interactive schema diagrams
- **Multiple Formats**: PNG and Markdown outputs
- **Customizable**: Configurable node limits and styling

## 🗂️ Data Structure

### Input Data
- **BIRD Dataset**: Database schemas and descriptions
- **Connection Graphs**: Pre-computed table relationships

### Output Data
- **Ground Truth**: Multi-hop SQL queries with natural language
- **Generated SQL**: AI-generated SQL queries
- **Evaluation Results**: Comprehensive evaluation metrics
- **Visualizations**: Schema diagrams and network graphs

## 📈 Performance

- **Generation Rate**: ~50-100 queries/second
- **API Rate**: ~1 request/second (rate limited)
- **Memory Usage**: <2GB for full generation
- **Storage**: ~20MB per 20,000 queries (CSV)

## 🔧 Configuration

### Model Configuration
- Default model: `qwen/qwen-2.5-72b-instruct`
- Configurable via model variables in scripts
- Support for multiple models via OpenRouter API

### Generation Parameters
- Hop lengths: 2-20 (configurable)
- Batch sizes: 100 queries per batch
- Temperature: 0.1 for consistent results

## 🛠️ Development

### Adding New Models
1. Add model configuration to relevant scripts
2. Update API call functions if needed
3. Add model-specific evaluation if required

### Adding New Evaluation Metrics
1. Create new evaluation script in `evaluation_testing/`
2. Follow existing patterns for batch processing
3. Add to main evaluation pipeline

### Adding New Visualizations
1. Add visualization functions to `visualization/`
2. Support both PNG and Mermaid outputs
3. Follow existing styling patterns

## 🤝 Contributing

1. Follow the existing code structure
2. Add comprehensive error handling
3. Include progress reporting for long operations
4. Update documentation for new features
5. Test with multiple databases before committing

## 📄 License

This project is part of the ARCEE AI research initiative.

## 🔗 Related Projects

- **BIRD Dataset**: Text-to-SQL benchmark
- **OpenRouter**: API access to multiple AI models
- **NetworkX**: Graph analysis and visualization
- **Matplotlib**: Plotting and visualization

## 📞 Support

For issues and questions:
1. Check existing documentation
2. Review error logs in `logs/` directory
3. Test with smaller datasets first
4. Verify API credentials and rate limits

## 🏆 Citation

If you use this work in your research, please cite:

```bibtex
@misc{connectdots2024,
  title={Connect Dots: Multi-Hop SQL Query Generation & Evaluation System},
  author={Anna4242},
  year={2024},
  url={https://github.com/Anna4242/Multi-Hop-SQL-Query-Generation}
}
``` 