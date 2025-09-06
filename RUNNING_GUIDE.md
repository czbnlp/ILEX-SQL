# ILEX-SQL 运行指南

本文档介绍如何设置和运行ILEX-SQL双模式智能SQL生成系统。

## 1. 环境准备

### 1.1 系统要求
- Python 3.8 或更高版本
- pip 包管理器
- Git

### 1.2 安装依赖
```bash
# 克隆项目
git clone https://github.com/czbnlp/ILEX-SQL.git
cd ILEX-SQL

# 安装Python依赖
pip install -r requirements.txt
```

如果requirements.txt不存在，请手动安装以下依赖：
```bash
pip install pyyaml requests openai sqlalchemy pandas numpy
```

## 2. 配置设置

### 2.1 模型API配置

ILEX-SQL支持多种LLM模型，你需要在配置文件中设置API密钥和相关参数。

#### 2.1.1 OpenAI API配置
创建一个`.env`文件在项目根目录：
```bash
# 创建环境变量文件
touch .env
```

在`.env`文件中添加：
```env
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4  # 或 gpt-3.5-turbo
OPENAI_BASE_URL=https://api.openai.com/v1
```

#### 2.1.2 其他LLM服务配置
如果你使用其他LLM服务（如Claude、Gemini等），请相应修改配置：

```env
# Claude API配置
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# 或其他自定义API
CUSTOM_API_URL=your_custom_api_url
CUSTOM_API_KEY=your_custom_api_key
```

### 2.2 数据库配置

在`.env`文件中添加数据库连接信息：
```env
# 数据库配置
DATABASE_URL=sqlite:///your_database.db
# 或者使用PostgreSQL
# DATABASE_URL=postgresql://username:password@localhost:5432/your_database
```

### 2.3 创建自定义配置文件

复制并修改配置文件：
```bash
cp config/ilex_config.yaml config/ilex_config_local.yaml
```

在`config/ilex_config_local.yaml`中你可以调整：
- 复杂度阈值
- 探索步数
- 记忆大小
- 日志级别等参数

## 3. 创建LLM连接器

创建一个`llm_connector.py`文件在项目根目录：

```python
import os
import requests
import json
from typing import Optional, Dict, Any

class LLMConnector:
    """LLM连接器类"""
    
    def __init__(self, config_path: str = "config/ilex_config.yaml"):
        self.config = self._load_config(config_path)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        import yaml
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        调用LLM API
        
        Args:
            prompt: 输入提示词
            
        Returns:
            LLM生成的响应文本
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个专业的SQL查询生成助手。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"LLM API调用失败: {e}")
            return "抱歉，我暂时无法处理您的请求。"
```

## 4. 创建SQL执行器

创建一个`sql_executor.py`文件在项目根目录：

```python
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine, text
from typing import Tuple, Optional, Any
import pandas as pd

class SQLExecutor:
    """SQL执行器类"""
    
    def __init__(self, database_url: str = "sqlite:///database.db"):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        
    def __call__(self, sql: str, db_path: Optional[str] = None) -> Tuple[Any, Optional[str]]:
        """
        执行SQL查询
        
        Args:
            sql: SQL查询语句
            db_path: 数据库路径（可选）
            
        Returns:
            (查询结果, 错误信息)
        """
        try:
            if db_path:
                # 如果指定了数据库路径，使用该路径
                engine = create_engine(f"sqlite:///{db_path}")
            else:
                engine = self.engine
                
            with engine.connect() as connection:
                result = pd.read_sql_query(text(sql), connection)
                return result.to_dict('records'), None
                
        except Exception as e:
            return None, str(e)
    
    def get_schema(self, table_name: Optional[str] = None) -> str:
        """
        获取数据库schema信息
        
        Args:
            table_name: 表名（可选）
            
        Returns:
            Schema信息字符串
        """
        try:
            with self.engine.connect() as connection:
                if table_name:
                    query = f"PRAGMA table_info({table_name});"
                    result = pd.read_sql_query(query, connection)
                    return f"表 {table_name} 的结构:\n{result.to_string()}"
                else:
                    query = "SELECT name FROM sqlite_master WHERE type='table';"
                    tables = pd.read_sql_query(query, connection)
                    schema_info = "数据库中的表:\n"
                    for table in tables['name']:
                        schema_info += f"- {table}\n"
                    return schema_info
        except Exception as e:
            return f"获取schema失败: {e}"
```

## 5. 运行示例

创建一个`run_example.py`文件：

```python
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径到Python路径
sys.path.append('src')

from ilex_core.mode_selector import ModeSelector
from ilex_core.exploration_engine import ExplorationEngine
from llm_connector import LLMConnector
from sql_executor import SQLExecutor

def main():
    """主运行函数"""
    print("=== ILEX-SQL 双模式智能SQL生成系统 ===")
    
    # 初始化组件
    print("初始化组件...")
    llm_connector = LLMConnector()
    sql_executor = SQLExecutor()
    
    # 初始化模式选择器
    mode_selector = ModeSelector()
    
    # 初始化探索引擎
    exploration_engine = ExplorationEngine(
        llm_connector=llm_connector,
        sql_executor=sql_executor
    )
    
    # 示例问题
    test_questions = [
        "What is the average salary of employees?",
        "Find the name of the employee who has the highest salary in the sales department.",
        "First, find the manager with the highest salary, then find all employees in the same department."
    ]
    
    # 获取数据库schema
    print("\n获取数据库schema...")
    schema = sql_executor.get_schema()
    print(schema)
    
    # 处理每个问题
    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"问题: {question}")
        print(f"{'='*50}")
        
        # 1. 评估问题复杂度并选择模式
        print("步骤1: 评估问题复杂度...")
        mode_decision = mode_selector.get_mode_decision(question)
        print(f"复杂度分数: {mode_decision['complexity_score']:.3f}")
        print(f"选择模式: {mode_decision['mode']}")
        print(f"推理: {mode_decision['reasoning']}")
        
        # 2. 根据选择的模式处理问题
        if mode_decision['use_exploration_mode']:
            print("步骤2: 使用探索模式处理...")
            final_sql, success, details = exploration_engine.solve_complex_question(
                question,
                "database.db",  # 数据库路径
                schema
            )
        else:
            print("步骤2: 使用经验模式处理...")
            # 这里可以实现经验模式的逻辑
            final_sql, success, details = "SELECT * FROM employees LIMIT 10;", True, {"mode": "experience"}
        
        # 3. 输出结果
        if success:
            print(f"✓ 生成的SQL: {final_sql}")
            print("✓ 执行结果:")
            result, error = sql_executor(final_sql)
            if error:
                print(f"✗ 执行错误: {error}")
            else:
                for i, row in enumerate(result[:5]):  # 只显示前5行
                    print(f"  {i+1}. {row}")
        else:
            print("✗ 处理失败")
            print(f"详细信息: {details}")

if __name__ == "__main__":
    main()
```

## 6. 运行项目

### 6.1 设置环境变量
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env文件，填入你的API密钥
nano .env
```

### 6.2 安装依赖
```bash
pip install python-dotenv
```

### 6.3 运行示例
```bash
python run_example.py
```

## 7. 故障排除

### 7.1 常见问题

**问题1: ModuleNotFoundError**
```bash
# 确保在项目根目录运行
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_example.py
```

**问题2: API密钥错误**
- 检查`.env`文件中的API密钥是否正确
- 确认API密钥有足够的权限

**问题3: 数据库连接失败**
- 检查数据库文件是否存在
- 确认数据库路径配置正确

### 7.2 调试模式

在配置文件中启用调试模式：
```yaml
debug:
  enable_logging: true
  log_level: "DEBUG"
```

## 8. 扩展和自定义

### 8.1 添加新的LLM支持
修改`llm_connector.py`中的`__call__`方法来支持新的LLM服务。

### 8.2 添加新的数据库支持
修改`sql_executor.py`来支持更多数据库类型。

### 8.3 自定义提示词
修改`prompts/ilex/`目录下的提示词文件来优化系统性能。

## 9. 性能优化

### 9.1 缓存配置
在配置文件中启用缓存：
```yaml
performance:
  enable_caching: true
  cache_size: 1000
```

### 9.2 并行处理
启用并行处理以提高性能：
```yaml
performance:
  enable_parallel_processing: true
  max_workers: 4
```

## 10. 获取帮助

如果遇到问题，请：
1. 检查日志输出
2. 查看配置文件
3. 确认API密钥和数据库连接
4. 参考本文档的故障排除部分

祝使用愉快！