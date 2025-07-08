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

## Ground Truth Pipeline Details

The **ground truth generator** performs two key tasks:

1. **Connection Graph Generation** – `generate_full_connection_graphs.py` crawls every SQLite database inside the BIRD corpus and builds a *fully-connected graph* of all tables.  Each edge represents a foreign-key relationship  so that every table is reachable from every other table.
2. **Natural Query Generation** – `natural_query_generator.py` reads the table-to-table *path* (produced during query generation) and produces an *over-arching* natural-language question whose answer requires traversing that exact path in the graph.

All of the intermediate artefacts are merged by `combine_final_data.py` into a single  file that serves as **ground-truth** for evaluation.

### Prompt Templates
All LLM prompts live in `connect_dots/prompts/`.  For example, `simple_sql_prompt.json` holds the template used by `simple_sql_generator.py`.  Editing this JSON file lets you tweak the system prompt without changing code.

### Quick Test Cycle
Once you have produced ground-truth data (run the scripts in *Generate Ground Truth Data* above) you can immediately assess an LLM’s SQL-generation ability:

```bash
# From project root
cd sql_generation
python simple_sql_generator.py              # Uses ground-truth NL questions & schema
```

`simple_sql_generator.py` reads each natural-language question, injects the **complete schema** for its database, and asks the model to infer the join-path and write the SQL.  You can then compare the generated SQL with the ground-truth answer to gauge model accuracy.

---
