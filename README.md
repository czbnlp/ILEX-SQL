# ILEX-SQL: Intelligent SQL Generation and Evaluation Framework

## Overview
ILEX-SQL is an advanced framework designed for generating, validating, and evaluating SQL queries using a combination of experience-based and exploration-based approaches. It integrates large language models (LLMs) to decompose complex natural language questions into manageable subproblems, generate SQL queries, and validate their correctness. The framework supports hybrid execution modes, dynamic few-shot learning, and robust debugging capabilities.

## Features
- **Hybrid SQL Generation**: Combines experience-based and exploration-based modes for robust query generation.
- **Dynamic Few-shot Learning**: Dynamically selects relevant examples to improve SQL generation accuracy.
- **LLM Integration**: Utilizes LLMs for problem decomposition, SQL generation, and self-correction.
- **SQL Validation and Execution**: Validates and executes generated SQL queries against real databases.
- **Concurrent Evaluation**: Supports multi-threaded evaluation for large datasets.
- **Customizable Configuration**: Easily configurable through YAML files.

## Project Structure

### Root Directory
- `.env`: Environment configuration file.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `requirements.txt`: Lists Python dependencies required for the project.
- `database.db`: Example SQLite database for testing.
- `deploy_vllm.sh`: Deployment script for vLLM models.
- `test_api.py`: Script for testing API-based LLM connectors.

### Core Modules
#### `bird_evaluator_unified.py`
- **Purpose**: Main evaluation script for the BIRD dataset.
- **Key Features**:
  - Supports hybrid SQL generation.
  - Handles concurrent evaluation.
  - Provides detailed logging and debugging information.

#### `enhanced_sql_generator.py`
- **Purpose**: Implements the experience-based SQL generation logic.
- **Key Features**:
  - Dynamic few-shot example selection.
  - Robust SQL extraction and validation.
  - Integration with LLM connectors.

#### `llm_connector_*.py`
- **Purpose**: Connectors for interacting with LLMs.
- **Variants**:
  - `llm_connector_local.py`: For local LLM deployments.
  - `llm_connector_api.py`: For API-based LLMs.
  - `llm_connector_unified.py`: Unified interface for both local and API-based LLMs.

#### `master_sql_postprocessor.py`
- **Purpose**: Post-processes generated SQL queries to ensure syntactic and semantic correctness.

#### `sql_executor.py`
- **Purpose**: Executes SQL queries against SQLite databases and returns results.

### Configuration
#### `config/ilex_config.yaml`
- **Purpose**: Central configuration file for the framework.
- **Key Parameters**:
  - `experience_mode`: Settings for experience-based SQL generation.
  - `exploration_mode`: Settings for exploration-based SQL generation.
  - `llm_connector`: Configuration for LLM connectors.

### Data
#### `data/`
- **Purpose**: Contains datasets and database files for evaluation.
- **Structure**:
  - `dev.json`: Development dataset.
  - `dev.sql`: Ground truth SQL queries for the development dataset.
  - `dev_databases/`: Directory containing SQLite database files.

### Documentation
#### `docs/subproblem_fields_explanation.md`
- **Purpose**: Detailed explanation of subproblem decomposition fields and their usage.

### Source Code
#### `src/ilex_core/`
- **Modules**:
  - `execution_memory.py`: Manages execution history for exploration mode.
  - `experience_retriever.py`: Retrieves relevant examples for few-shot learning.
  - `experience_upgrader.py`: Analyzes and upgrades the experience knowledge base.
  - `exploration_engine_llm.py`: Implements the exploration-based SQL generation engine.
  - `problem_decomposer.py`: Decomposes complex questions into subproblems.
  - `sql_generator_hybrid.py`: Combines experience and exploration modes for SQL generation.

### Knowledge Base
#### `knowledge_base/`
- **Files**:
  - `correct_set.json`: Stores correct examples for experience-based learning.
  - `init_correct_set.json`: Initial correct examples.
  - `init_mistake_set.json`: Initial mistake examples.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/czbnlp/ILEX-SQL.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ILEX-SQL
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running the Evaluator
To evaluate the BIRD dataset:
```bash
python bird_evaluator_unified.py --concurrency 5 --max-questions 10 --output results.json --api-model
```
