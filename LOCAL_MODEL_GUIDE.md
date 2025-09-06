# 本地模型部署指南

本指南介绍如何在ILEX-SQL项目中使用本地部署的开源模型替代OpenAI API。

## 概述

项目现在支持通过vLLM部署本地开源模型，提供以下优势：
- 完全本地化，无需外部API调用
- 更高的隐私保护
- 可定制的模型选择
- 更快的响应速度（取决于硬件配置）

## 前置要求

### 硬件要求
- GPU: 至少24GB显存（推荐）
- 内存: 32GB以上
- 存储: 足够存放模型文件的空间

### 软件要求
- Python 3.8+
- CUDA 11.0+
- vLLM库
- PyTorch

## 安装步骤

### 1. 安装vLLM

```bash
pip install vllm
```

### 2. 下载模型

```bash
# 示例：下载Qwen模型
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-8B-v2
```

### 3. 配置环境变量

复制并编辑环境变量文件：

```bash
cp .env.example .env
```

编辑`.env`文件，设置本地模型配置：

```env
# 本地模型配置（vLLM部署）
LOCAL_MODEL=Qwen3-8B_v2_p2
LOCAL_API_KEY=EMPTY
LOCAL_TIMEOUT=1000
LOCAL_BASE_URLS=http://localhost:8882/v1,http://localhost:8883/v1
```

## 启动服务

### 1. 启动vLLM服务

使用提供的部署脚本：

```bash
./deploy_vllm.sh
```

或者手动启动：

```bash
export CUDA_VISIBLE_DEVICES="2,3"

vllm serve /path/to/your/model \
  --port 8883 \
  --served-model-name Qwen3-8B_v2_p2 \
  --tensor-parallel-size 2 \
  --max-model-len 16384
```

### 2. 验证服务状态

```bash
# 检查服务是否运行
curl http://localhost:8883/v1/models

# 或者运行测试脚本
python -c "from llm_connector_local import LocalLLMConnector; connector = LocalLLMConnector(); print('连接成功' if connector.test_connection() else '连接失败')"
```

## 使用本地模型

### 1. 运行示例

```bash
# 使用本地模型运行示例
python run_example.py
```

### 2. 交互模式

```bash
# 启动交互模式
python run_example.py --interactive
```

### 3. 测试组件

```bash
# 测试各个组件
python run_example.py --test
```

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `LOCAL_MODEL` | 模型名称 | `Qwen3-8B_v2_p2` |
| `LOCAL_API_KEY` | API密钥 | `EMPTY` |
| `LOCAL_TIMEOUT` | 请求超时时间（秒） | `1000` |
| `LOCAL_BASE_URLS` | vLLM服务URL列表 | `http://localhost:8882/v1,http://localhost:8883/v1` |

### 多服务部署

支持同时运行多个vLLM实例以实现负载均衡：

```bash
# 终端1：启动第一个服务
export CUDA_VISIBLE_DEVICES="0,1"
vllm serve /path/to/model --port 8882 --tensor-parallel-size 2

# 终端2：启动第二个服务
export CUDA_VISIBLE_DEVICES="2,3"
vllm serve /path/to/model --port 8883 --tensor-parallel-size 2
```

然后在`.env`文件中配置：

```env
LOCAL_BASE_URLS=http://localhost:8882/v1,http://localhost:8883/v1
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少tensor-parallel-size
   - 使用更小的模型
   - 减少max-model-len

2. **连接失败**
   - 检查vLLM服务是否启动
   - 验证端口是否被占用
   - 检查防火墙设置

3. **模型加载失败**
   - 验证模型路径是否正确
   - 检查模型文件完整性
   - 确保有足够的磁盘空间

### 调试命令

```bash
# 检查端口占用
lsof -i :8883

# 查看GPU使用情况
nvidia-smi

# 测试API连接
curl -X POST http://localhost:8883/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-8B_v2_p2", "messages": [{"role": "user", "content": "Hello"}]}'
```

## 性能优化

### 1. 批量处理

使用批量处理功能提高吞吐量：

```python
from llm_connector_local import LocalLLMConnector

connector = LocalLLMConnector()
items = [...]  # 你的数据列表
connector.batch_process(items, "output.jsonl", "progress.jsonl", concurrency=10)
```

### 2. 缓存配置

在`config/ilex_config.yaml`中启用缓存：

```yaml
ilex:
  performance:
    enable_caching: true
    cache_size: 1000
```

### 3. 并发设置

根据GPU内存调整并发数：

```python
# 高并发（需要更多GPU内存）
connector.batch_process(items, output_file, progress_file, concurrency=20)

# 低并发（节省GPU内存）
connector.batch_process(items, output_file, progress_file, concurrency=5)
```

## 迁移指南

### 从OpenAI API迁移

1. **安装依赖**
   ```bash
   pip install vllm
   ```

2. **更新环境变量**
   - 设置`LOCAL_MODEL`等变量
   - 可选：保留OpenAI配置作为备用

3. **修改代码**
   - 将`from llm_connector import LLMConnector`改为`from llm_connector_local import LocalLLMConnector`
   - 将`LLMConnector()`改为`LocalLLMConnector()`

4. **启动服务**
   ```bash
   ./deploy_vllm.sh
   ```

5. **测试验证**
   ```bash
   python run_example.py --test
   ```

## 支持的模型

目前测试支持的模型：
- Qwen3系列
- Llama系列
- Mistral系列

添加其他模型需要：
1. 确保模型格式兼容
2. 更新`LOCAL_MODEL`环境变量
3. 可能需要调整提示词模板

## 贡献指南

如需添加对新模型的支持或改进现有功能，请：
1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目遵循MIT许可证。详见LICENSE文件。