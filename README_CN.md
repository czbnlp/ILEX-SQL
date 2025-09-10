# ILEX-SQL: 智能SQL生成与评估框架

## 概述
ILEX-SQL 是一个高级框架，旨在通过结合基于经验和基于探索的方法来生成、验证和评估 SQL 查询。它集成了大型语言模型（LLMs），将复杂的自然语言问题分解为可管理的子问题，生成 SQL 查询并验证其正确性。该框架支持混合执行模式、动态少样本学习以及强大的调试功能。

## 功能特点
- **混合SQL生成**：结合经验模式和探索模式，生成稳健的查询。
- **动态少样本学习**：动态选择相关示例以提高SQL生成的准确性。
- **LLM集成**：利用LLM进行问题分解、SQL生成和自我纠错。
- **SQL验证与执行**：验证生成的SQL查询并在真实数据库上执行。
- **并发评估**：支持多线程评估大规模数据集。
- **可定制配置**：通过YAML文件轻松配置。

## 项目结构

### 根目录
- `.env`：环境配置文件。
- `.gitignore`：指定Git忽略的文件和目录。
- `requirements.txt`：项目所需的Python依赖。
- `database.db`：用于测试的示例SQLite数据库。
- `deploy_vllm.sh`：vLLM模型的部署脚本。
- `test_api.py`：用于测试基于API的LLM连接器的脚本。

### 核心模块
#### `bird_evaluator_unified.py`
- **用途**：BIRD数据集的主要评估脚本。
- **主要功能**：
  - 支持混合SQL生成。
  - 处理并发评估。
  - 提供详细的日志记录和调试信息。

#### `enhanced_sql_generator.py`
- **用途**：实现基于经验的SQL生成逻辑。
- **主要功能**：
  - 动态选择少样本示例。
  - 稳健的SQL提取与验证。
  - 与LLM连接器集成。

#### `llm_connector_*.py`
- **用途**：与LLM交互的连接器。
- **变体**：
  - `llm_connector_local.py`：用于本地LLM部署。
  - `llm_connector_api.py`：用于基于API的LLM。
  - `llm_connector_unified.py`：统一接口，支持本地和基于API的LLM。

#### `master_sql_postprocessor.py`
- **用途**：后处理生成的SQL查询，确保语法和语义的正确性。

#### `sql_executor.py`
- **用途**：在SQLite数据库上执行SQL查询并返回结果。

### 配置
#### `config/ilex_config.yaml`
- **用途**：框架的中央配置文件。
- **主要参数**：
  - `experience_mode`：基于经验的SQL生成设置。
  - `exploration_mode`：基于探索的SQL生成设置。
  - `llm_connector`：LLM连接器的配置。

### 数据
#### `data/`
- **用途**：包含用于评估的数据集和数据库文件。
- **结构**：
  - `dev.json`：开发数据集。
  - `dev.sql`：开发数据集的标准SQL查询。
  - `dev_databases/`：包含SQLite数据库文件的目录。

### 文档
#### `docs/subproblem_fields_explanation.md`
- **用途**：详细解释子问题分解字段及其用法。

### 源代码
#### `src/ilex_core/`
- **模块**：
  - `execution_memory.py`：管理探索模式的执行历史。
  - `experience_retriever.py`：检索少样本学习的相关示例。
  - `experience_upgrader.py`：分析并升级经验知识库。
  - `exploration_engine_llm.py`：实现基于探索的SQL生成引擎。
  - `problem_decomposer.py`：将复杂问题分解为子问题。
  - `sql_generator_hybrid.py`：结合经验和探索模式生成SQL。

### 知识库
#### `knowledge_base/`
- **文件**：
  - `correct_set.json`：存储用于基于经验学习的正确示例。
  - `init_correct_set.json`：初始正确示例。
  - `init_mistake_set.json`：初始错误示例。
  - `qwen2-72b_4_True_True_rate_0.5/`：包含示例向量化表示的目录。

## 安装
1. 克隆仓库：
   ```bash
   git clone https://github.com/czbnlp/ILEX-SQL.git
   ```
2. 进入项目目录：
   ```bash
   cd ILEX-SQL
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法
### 运行评估器
评估BIRD数据集：
```bash
python bird_evaluator_unified.py --concurrency 5 --max-questions 10 --output results.json --api-model
```
