# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ILEX-SQL is a dual-mode intelligent SQL generation system that integrates LPE-SQL's experience reuse capabilities with Amber's iterative exploration abilities. The system processes natural language questions and generates SQL queries with high accuracy through a hybrid approach combining experience-based and exploration-based modes.

## Key Commands

### Running the System
```bash
# Run main example with test questions
python run_example.py

# Test individual components
python run_example.py --test

# Interactive mode for user queries
python run_example.py --interactive

# Evaluate BIRD dataset (requires vLLM service)
python run_example.py --bird

# Mock evaluation (no vLLM required)
python run_example.py --bird-mock
```

### vLLM Deployment
```bash
# Deploy local vLLM service
bash deploy_vllm.sh
```

### Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

## Architecture Overview

### Core Components

1. **Hybrid SQL Generator** (`src/ilex_core/sql_generator_hybrid.py`)
   - Central orchestrator that combines experience and exploration modes
   - Automatically switches modes based on complexity analysis
   - Manages fallback strategies when primary approaches fail

2. **LLM Connectors**
   - `llm_connector_local.py`: Local vLLM integration for open-source models
   - `llm_connector.py`: External API integration (OpenAI, Anthropic)
   - Supports multiple concurrent connections and failover

3. **SQL Execution Layer** (`sql_executor.py`)
   - Database abstraction using SQLAlchemy
   - Supports SQLite, PostgreSQL, MySQL
   - Automatic sample database creation for testing
   - Connection pooling for concurrent operations

4. **BIRD Dataset Evaluator** (`bird_evaluator_unified.py`)
   - Comprehensive evaluation framework
   - Supports concurrent processing with configurable limits
   - Mock mode for testing without LLM dependencies
   - Detailed statistics and result reporting

### Dual-Mode Architecture

**Experience Mode**: 
- Uses `EnhancedSQLGenerator` for direct SQL generation
- Leverages historical success patterns
- Fast processing for simple to moderate complexity questions

**Exploration Mode**:
- Uses `ExplorationEngine` for complex multi-step problems
- Employs problem decomposition and iterative refinement
- Includes intermediate validation and error recovery

### Configuration System

Main configuration in `config/ilex_config.yaml`:
- Complexity thresholds for mode selection
- Exploration parameters (max steps, timeouts)
- Memory and caching settings
- Debug and logging options

### Key Design Patterns

1. **Error Feedback Loop**: SQL execution errors are captured and fed back to LLM for iterative improvement
2. **Multi-stage Post-processing**: Comprehensive SQL syntax fixing including special character handling
3. **Concurrent Processing**: Thread-safe evaluation with progress tracking
4. **Modular LLM Integration**: Pluggable connectors for different model providers

## Development Guidelines

### When Adding New Features
- Follow the existing dual-mode architecture pattern
- Ensure thread-safety for concurrent operations
- Add appropriate logging using the existing logger setup
- Test with both local vLLM and mock modes

### Database Schema Handling
- Schema information is dynamically extracted from databases
- Special attention to column names with special characters (%, (), -, etc.)
- Automatic table relationship detection

### Error Handling
- Implement comprehensive error capture and retry mechanisms
- Use the established error feedback system for SQL generation improvements
- Provide detailed error context for debugging

### Performance Considerations
- Configure appropriate concurrency limits based on available resources
- Use connection pooling for database operations
- Implement caching where appropriate (configured in ilex_config.yaml)