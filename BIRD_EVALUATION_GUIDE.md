# BIRD数据集评估指南

本指南介绍如何使用ILEX-SQL系统评估BIRD数据集，包括模拟测试和真实评估两种模式。

## 概述

BIRD（Big Bench for Interactive Evaluation of Large Language Models on Text-to-SQL）是一个大规模的文本到SQL数据集，包含12,751个自然语言问题和相应的SQL查询，涉及95个不同的数据库。

## 系统要求

### 必需组件
1. **BIRD数据集文件**：
   - `dev.json` - 开发集问题
   - `dev.sql` - 开发集标准SQL答案
   - `dev_databases/` - 数据库文件目录

2. **Python依赖**：
   - 已安装所有项目依赖（见requirements.txt）

3. **可选组件**：
   - vLLM服务（用于真实评估）
   - GPU支持（用于本地模型部署）

## 快速开始

### 1. 环境检查

首先检查您的环境是否准备好：

```bash
# 运行环境检查脚本
python test_bird_final.py
```

这个脚本会检查：
- 数据文件是否存在
- 数据库文件是否完整
- 组件是否正常初始化

### 2. 模拟评估（推荐用于测试）

如果您还没有部署vLLM服务，可以使用模拟版本进行测试：

```bash
# 模拟评估 - 测试5个问题
python bird_evaluator_mock.py --max-questions 5 --output test_results.json

# 模拟评估 - 完整开发集
python bird_evaluator_mock.py --output full_test_results.json
```

### 3. 真实评估（需要vLLM服务）

当您的vLLM服务部署完成后，可以使用真实评估：

```bash
# 真实评估 - 测试5个问题
python bird_evaluator_working.py --max-questions 5 --output real_test_results.json

# 真实评估 - 完整开发集
python bird_evaluator_working.py --output real_full_results.json

# 真实评估 - 使用OpenAI API
python bird_evaluator_working.py --use-openai --output openai_results.json
```

### 4. 通过主程序使用

您也可以通过主程序启动BIRD评估：

```bash
# 使用主程序进行BIRD评估
python run_example.py --bird
```

## 详细使用说明

### 评估器参数

所有评估器都支持以下参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--split` | 数据集分割 (dev/test/train) | `dev` |
| `--max-questions` | 最大评估问题数量 | `None` (全部) |
| `--output` | 输出文件路径 | `bird_evaluation_results.json` |
| `--timeout` | SQL执行超时时间（秒） | `30` |
| `--use-openai` | 使用OpenAI API而非本地模型 | `False` |

### 使用示例

#### 基础测试
```bash
# 测试前10个问题
python bird_evaluator_mock.py --max-questions 10 --output quick_test.json
```

#### 完整评估
```bash
# 评估整个开发集
python bird_evaluator_working.py --split dev --output dev_evaluation.json
```

#### 自定义输出
```bash
# 自定义输出文件和超时时间
python bird_evaluator_working.py --max-questions 50 --timeout 60 --output custom_results.json
```

## 评估结果解读

### 输出文件结构

评估结果会保存为JSON格式，包含以下结构：

```json
{
  "statistics": {
    "total": 总问题数,
    "correct": 正确数,
    "timeout": 超时数,
    "error": 错误数,
    "by_difficulty": {
      "simple": {"total": 简单问题数, "correct": 正确数},
      "moderate": {"total": 中等问题数, "correct": 正确数},
      "challenging": {"total": 困难问题数, "correct": 正确数}
    }
  },
  "results": [
    {
      "question_id": 问题ID,
      "question": 问题文本,
      "db_id": 数据库ID,
      "difficulty": 难度级别,
      "gold_sql": 标准SQL,
      "predicted_sql": 预测SQL,
      "correct": 是否正确,
      "error": 错误信息,
      "timeout": 是否超时,
      "execution_time": 执行时间,
      "mode": 使用的模式
    }
  ]
}
```

### 性能指标

#### 总体准确率
```
总准确率 = 正确数 / 总问题数
```

#### 按难度分类准确率
```
简单问题准确率 = 简单问题正确数 / 简单问题总数
中等问题准确率 = 中等问题正确数 / 中等问题总数
困难问题准确率 = 困难问题正确数 / 困难问题总数
```

#### 错误分析
- **超时数**：SQL执行超时的问题数量
- **错误数**：SQL生成或执行失败的问题数量

## 故障排除

### 常见问题

#### 1. 数据文件不存在
```
错误: FileNotFoundError: BIRD数据集文件不存在: LPE-SQL/data/dev.json
```
**解决方案**：
- 确保BIRD数据集文件已正确下载
- 检查文件路径是否正确

#### 2. 数据库文件缺失
```
错误: FileNotFoundError: 数据库文件不存在: /path/to/database.sqlite
```
**解决方案**：
- 运行数据库下载脚本：`python download_bird_databases.py --create-sample`
- 或手动下载完整的BIRD数据集

#### 3. vLLM连接失败
```
错误: 无法连接到任何本地模型服务
```
**解决方案**：
- 检查vLLM服务是否启动：`ps aux | grep vllm`
- 检查端口是否被占用：`lsof -i :8883`
- 使用模拟版本进行测试：`python bird_evaluator_mock.py`

#### 4. 内存不足
```
错误: CUDA out of memory
```
**解决方案**：
- 减少并发数：`--max-questions 10`
- 使用更小的模型
- 增加GPU内存

### 调试技巧

#### 1. 单步调试
```bash
# 运行单个问题测试
python debug_schema.py
```

#### 2. 查看详细日志
```bash
# 启用详细日志
export LOG_LEVEL=DEBUG
python bird_evaluator_working.py --max-questions 1
```

#### 3. 检查中间结果
```bash
# 查看生成的SQL
python -c "
import json
with open('results.json', 'r') as f:
    data = json.load(f)
    for result in data['results'][:3]:
        print(f'问题: {result[\"question\"][:50]}...')
        print(f'预测SQL: {result[\"predicted_sql\"]}')
        print(f'标准SQL: {result[\"gold_sql\"]}')
        print(f'正确: {result[\"correct\"]}')
        print('-' * 50)
"
```

## 高级用法

### 1. 自定义评估脚本

您可以创建自定义的评估脚本：

```python
#!/usr/bin/env python3
from bird_evaluator_working import BIRDEvaluator

# 创建评估器
evaluator = BIRDEvaluator()

# 自定义评估逻辑
def custom_evaluation():
    # 加载数据
    bird_data = evaluator.load_bird_data("dev")
    gold_sqls = evaluator.load_gold_sql("dev")
    
    # 选择特定难度的问题
    simple_questions = [q for q in bird_data if q.get('difficulty') == 'simple']
    
    # 评估
    results = []
    for question in simple_questions[:10]:  # 只评估前10个简单问题
        idx = question.get('question_id')
        gold_sql = gold_sqls.get(str(idx))
        result = evaluator.evaluate_single_question(question, gold_sql, idx)
        results.append(result)
    
    return results

# 运行自定义评估
results = custom_evaluation()
print(f"评估了 {len(results)} 个问题")
```

### 2. 批量评估

```bash
# 评估不同数据集分割
for split in dev test train; do
    python bird_evaluator_working.py --split $split --output ${split}_results.json
done
```

### 3. 性能基准测试

```bash
# 测试不同并发数
for concurrency in 1 5 10 20; do
    echo "测试并发数: $concurrency"
    time python bird_evaluator_working.py --max-questions 100 --output benchmark_${concurrency}.json
done
```

## 最佳实践

### 1. 评估流程建议

1. **先运行模拟测试**：
   ```bash
   python bird_evaluator_mock.py --max-questions 10
   ```

2. **然后运行小规模真实测试**：
   ```bash
   python bird_evaluator_working.py --max-questions 50
   ```

3. **最后运行完整评估**：
   ```bash
   python bird_evaluator_working.py
   ```

### 2. 结果分析

1. **查看总体准确率**：
   ```bash
   python -c "
   import json
   with open('results.json', 'r') as f:
       data = json.load(f)
       stats = data['statistics']
       print(f'总体准确率: {stats[\"correct\"]/stats[\"total\"]*100:.2f}%')
   "
   ```

2. **分析错误案例**：
   ```bash
   python -c "
   import json
   with open('results.json', 'r') as f:
       data = json.load(f)
       errors = [r for r in data['results'] if not r['correct']]
       print(f'错误案例数: {len(errors)}')
       for error in errors[:3]:
           print(f'问题: {error[\"question\"][:50]}...')
           print(f'错误: {error.get(\"error\", \"未知错误\")}')
   "
   ```

### 3. 性能优化

1. **使用更快的硬件**：GPU加速的vLLM服务
2. **优化数据库查询**：确保数据库文件在SSD上
3. **调整超时设置**：根据问题复杂度调整`--timeout`参数

## 参考资源

- [BIRD数据集官方页面](https://bird-bench.github.io/)
- [vLLM部署指南](LOCAL_MODEL_GUIDE.md)
- [项目运行指南](RUNNING_GUIDE.md)

## 技术支持

如果您在使用过程中遇到问题，请：

1. 检查本文档的故障排除部分
2. 查看项目的GitHub Issues
3. 运行调试脚本获取更多信息

---

祝您使用愉快！