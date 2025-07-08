#Multi-Hop SQL Query Generation 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/Anna4242/Multi-Hop-SQL-Query-Generation.svg)](https://github.com/Anna4242/Multi-Hop-SQL-Query-Generation/stargazers)

A comprehensive toolkit for generating complex multi-hop SQL queries, creating natural language descriptions, and evaluating SQL generation models across database schemas.




#### Clone the Repository
```bash
git clone https://github.com/Anna4242/Multi-Hop-SQL-Query-Generation.git
cd Multi-Hop-SQL-Query-Generation
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```


### Environment Setup
1. **Copy the example environment file:**
   ```bash
   cp env.example .env
   ```

2. **Edit `.env` and configure your settings:**
   ```env
   # Required: OpenRouter API key for LLM access
   OPENROUTER_API_KEY=your_api_key_here
   
   # Optional: Path to BIRD database directory
   BIRD_DB_PATH=../bird/train/train_databases/train_databases
   
   # Optional: OpenAI API base URL (defaults to OpenRouter)
   OPENAI_API_BASE=https://openrouter.ai/api/v1
   
   # Optional: Default model to use
   DEFAULT_MODEL=qwen/qwen-2.5-72b-instruct
   ```


   - Visit [OpenRouter.ai](https://openrouter.ai/)
   - Sign up for an account
   - Navigate to API Keys section
   - Generate a new API key
   - Add it to your `.env` file

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
##TODO
```

### Visualize Database Schemas
```bash
cd visualization
python visualize_graphs.py                 # Generate visualizations
```
