# ILEX-SQL 运行指南

本文档介绍如何设置和运行ILEX-SQL双模式智能SQL生成系统，支持本地vLLM部署和云端API两种模式。

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

### 1.3 可选：vLLM安装（本地模型）
如果计划使用本地模型，需要安装vLLM：
```bash
pip install vllm
```

## 2. 配置设置

### 2.1 环境变量配置

复制环境变量模板：
```bash
cp .env.example .env
```

根据使用的模型类型，选择相应的配置：

#### 2.1.1 本地vLLM模型配置
```env
# 本地模型配置（vLLM部署）
LOCAL_MODEL=Qwen3-8B_v2_p2
LOCAL_API_KEY=EMPTY
LOCAL_TIMEOUT=1000
LOCAL_BASE_URLS=http://localhost:8882/v1,http://localhost:8883/v1

# 数据库配置
DATABASE_URL=sqlite:///database.db

# 日志级别
LOG_LEVEL=INFO
```

#### 2.1.2 OpenAI API配置（可选）
```env
# OpenAI API配置（可选，如果使用本地模型则不需要）
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_BASE_URL=https://api.openai.com/v1

# 数据库配置
DATABASE_URL=sqlite:///database.db

# 日志级别
LOG_LEVEL=INFO
```

### 2.2 数据库配置

系统支持多种数据库类型：

#### SQLite（默认）
```env
DATABASE_URL=sqlite:///database.db
```

#### PostgreSQL
```env
DATABASE_URL=postgresql://username:password@localhost:5432/your_database
```

#### MySQL
```env
DATABASE_URL=mysql://username:password@localhost:3306/your_database
```

## 3. 模型部署

### 3.1 本地vLLM部署（推荐）

#### 3.1.1 下载模型
```bash
# 示例：下载Qwen模型
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-8B-v2
```

#### 3.1.2 启动vLLM服务
使用提供的部署脚本：
```bash
# 给脚本执行权限
chmod +x deploy_vllm.sh

# 启动服务
./deploy_vllm.sh
```

或手动启动：
```bash
export CUDA_VISIBLE_DEVICES="2,3"

vllm serve /path/to/your/model \
  --port 8883 \
  --served-model-name Qwen3-8B_v2_p2 \
  --tensor-parallel-size 2 \
  --max-model-len 16384
```

#### 3.1.3 验证服务
```bash
# 检查服务状态
curl http://localhost:8883/v1/models

# 测试连接
python -c "from llm_connector_local import LocalLLMConnector; connector = LocalLLMConnector(); print('连接成功' if connector.test_connection() else '连接失败')"
```

### 3.2 多服务部署（负载均衡）

启动多个vLLM实例：
```bash
# 终端1：启动第一个服务
export CUDA_VISIBLE_DEVICES="0,1"
vllm serve /path/to/model --port 8882 --tensor-parallel-size 2

# 终端2：启动第二个服务
export CUDA_VISIBLE_DEVICES="2,3"
vllm serve /path/to/model --port 8883 --tensor-parallel-size 2
```

在`.env`文件中配置：
```env
LOCAL_BASE_URLS=http://localhost:8882/v1,http://localhost:8883/v1
```

## 4. 运行项目

### 4.1 基本运行

```bash
# 运行示例
python run_example.py
```

### 4.2 交互模式

```bash
# 启动交互模式
python run_example.py --interactive
```

在交互模式下，你可以：
- 输入自然语言问题
- 查看生成的SQL查询
- 获取执行结果

### 4.3 测试组件

```bash
# 测试各个组件
python run_example.py --test
```

### 4.4 直接测试连接器

```bash
# 测试本地LLM连接器
python llm_connector_local.py

# 测试OpenAI连接器（如果配置了）
python llm_connector.py
```

## 5. 项目结构

```
ILEX-SQL/
├── config/
│   └── ilex_config.yaml          # 系统配置文件
├── src/
│   └── ilex_core/                # 核心模块
│       ├── mode_selector.py       # 模式选择器
│       ├── exploration_engine.py  # 探索引擎
│       ├── problem_decomposer.py  # 问题分解器
│       └── execution_memory.py    # 执行记忆
├── prompts/ilex/                  # 提示词模板
├── llm_connector.py              # OpenAI API连接器
├── llm_connector_local.py        # 本地vLLM连接器
├── sql_executor.py               # SQL执行器
├── run_example.py                # 主程序
├── deploy_vllm.sh                # vLLM部署脚本
├── requirements.txt              # 依赖列表
├── .env.example                  # 环境变量模板
├── LOCAL_MODEL_GUIDE.md          # 本地模型详细指南
└── RUNNING_GUIDE.md             # 本文件
```

## 6. 使用示例

### 6.1 简单查询
```python
from llm_connector_local import LocalLLMConnector
from sql_executor import SQLExecutor

# 初始化组件
connector = LocalLLMConnector()
executor = SQLExecutor()

# 生成SQL
prompt = "找出薪水最高的员工"
sql = connector(prompt)

# 执行SQL
result, error = executor(sql)
print(result)
```

### 6.2 批量处理
```python
from llm_connector_local import LocalLLMConnector

connector = LocalLLMConnector()

# 批量处理任务
items = [
    {"item_title": "问题1", "item_content": "内容1", "nid": 1},
    {"item_title": "问题2", "item_content": "内容2", "nid": 2}
]

connector.batch_process(
    items, 
    "output.jsonl", 
    "progress.jsonl", 
    concurrency=10
)
```

## 7. 配置优化

### 7.1 性能配置
在`config/ilex_config.yaml`中调整：

```yaml
ilex:
  performance:
    enable_caching: true          # 启用缓存
    cache_size: 1000             # 缓存大小
    enable_parallel_processing: false  # 并行处理
    max_workers: 4               # 最大工作线程数
  
  exploration:
    max_exploration_steps: 5     # 最大探索步数
    timeout_per_step: 30         # 每步超时时间
    max_retries_per_subquestion: 3 # 最大重试次数
```

### 7.2 模式选择配置
```yaml
ilex:
  mode_selection:
    complexity_threshold: 0.7    # 复杂度阈值
    enable_exploration: true     # 启用探索模式
    enable_complexity_analysis: true  # 启用复杂度分析
```

## 8. 故障排除

### 8.1 常见问题

**问题1: 本地模型连接失败**
```bash
# 检查vLLM服务状态
ps aux | grep vllm

# 检查端口占用
lsof -i :8883

# 检查GPU状态
nvidia-smi
```

**问题2: CUDA内存不足**
- 减少tensor-parallel-size
- 使用更小的模型
- 减少max-model-len

**问题3: 导入错误**
```bash
# 确保在项目根目录运行
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_example.py
```

**问题4: 数据库连接失败**
- 检查数据库文件是否存在
- 确认数据库路径配置正确
- 检查数据库服务是否运行

### 8.2 调试模式

启用详细日志：
```env
LOG_LEVEL=DEBUG
```

或在配置文件中：
```yaml
ilex:
  debug:
    enable_logging: true
    log_level: "DEBUG"
    save_exploration_history: true
    save_execution_traces: true
```

### 8.3 性能监控

查看GPU使用情况：
```bash
watch -n 1 nvidia-smi
```

监控vLLM服务：
```bash
# 查看服务日志
journalctl -u vllm -f

# 或直接查看进程输出
ps aux | grep vllm
```

## 9. 高级功能

### 9.1 自定义提示词
修改`prompts/ilex/`目录下的提示词文件：
- `complexity_analysis.txt` - 复杂度分析提示词
- `problem_decompose.txt` - 问题分解提示词
- `subquestion_solve.txt` - 子问题求解提示词
- `result_synthesize.txt` - 结果合成提示词

### 9.2 知识库管理
在`src/knowledge_base/`目录下：
- `init_correct_set.json` - 正确案例集合
- `init_mistake_set.json` - 错误案例集合

### 9.3 扩展新模型
1. 在`llm_connector_local.py`中添加新模型支持
2. 更新`deploy_vllm.sh`脚本
3. 修改`LOCAL_MODEL_GUIDE.md`

## 10. 最佳实践

### 10.1 硬件配置推荐
- **开发环境**: 单GPU（24GB显存）
- **生产环境**: 多GPU（推荐4x A100/H100）
- **内存**: 32GB以上
- **存储**: SSD，足够存放模型文件

### 10.2 性能优化建议
1. **批量处理**: 使用`batch_process`方法提高吞吐量
2. **缓存配置**: 启用缓存减少重复计算
3. **并发控制**: 根据GPU内存调整并发数
4. **模型选择**: 根据任务复杂度选择合适大小的模型

### 10.3 安全建议
1. **API密钥**: 不要在代码中硬编码API密钥
2. **网络安全**: 在生产环境中使用HTTPS
3. **访问控制**: 限制vLLM服务的访问权限
4. **日志安全**: 避免在日志中记录敏感信息

## 11. 获取帮助

### 11.1 文档资源
- `LOCAL_MODEL_GUIDE.md` - 本地模型详细指南
- `README.md` - 项目概述
- `config/ilex_config.yaml` - 配置文件说明

### 11.2 社区支持
- GitHub Issues: https://github.com/czbnlp/ILEX-SQL/issues
- Wiki: https://github.com/czbnlp/ILEX-SQL/wiki

### 11.3 联系方式
如有问题或建议，请通过GitHub Issues联系我们。

---

祝使用愉快！