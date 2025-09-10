# 子问题分解字段详解

## 1. dependencies: 依赖的子问题ID列表

### 含义
表示当前子问题执行前必须先完成的其他子问题。这是一个整数列表，包含所有前置依赖子问题的ID。

### 示例
```json
{
  "id": 3,
  "description": "计算每个部门的平均薪资",
  "dependencies": [1, 2],
  "priority": 4,
  "estimated_complexity": 0.6
}
```

### 实际应用场景
- **空列表 `[]`**：表示该子问题可以独立执行，无需前置条件
- **`[1]`**：表示必须先完成ID为1的子问题
- **`[1, 2]`**：表示必须先完成ID为1和ID为2的子问题

### 执行逻辑
```python
def can_execute(subproblem, solved_ids):
    """检查子问题是否可以执行"""
    return all(dep_id in solved_ids for dep_id in subproblem.dependencies)
```

---

## 2. priority: 优先级(1-5)

### 含义
表示子问题的重要性和执行优先级，数值越高表示优先级越高。

### 优先级分级
| 数值 | 优先级 | 描述 | 典型场景 |
|------|--------|------|----------|
| 1 | 最低 | 可选或辅助性任务 | 数据验证、格式化输出 |
| 2 | 低 | 次要任务 | 数据清洗、简单过滤 |
| 3 | 中等 | 常规任务 | 基础查询、简单计算 |
| 4 | 高 | 重要任务 | 核心计算、关键数据获取 |
| 5 | 最高 | 关键任务 | 主要业务逻辑、最终结果生成 |

### 示例
```json
{
  "id": 1,
  "description": "获取所有销售记录",
  "dependencies": [],
  "priority": 5,  # 最高优先级，必须先获取基础数据
  "estimated_complexity": 0.3
}
```

### 执行策略
```python
def get_next_subproblem(available_subproblems):
    """获取下一个要执行的子问题"""
    # 优先执行高优先级任务
    return max(available_subproblems, key=lambda x: x.priority)
```

---

## 3. estimated_complexity: 预估复杂度(0-1)

### 含义
表示子问题的计算复杂度和执行难度，数值越高表示越复杂。

### 复杂度分级
| 范围 | 复杂度 | 描述 | 典型特征 |
|------|--------|------|----------|
| 0.0-0.2 | 极低 | 简单查询 | 单表查询、无计算 |
| 0.2-0.4 | 低 | 基础查询 | 简单连接、基础聚合 |
| 0.4-0.6 | 中等 | 复杂查询 | 多表连接、复杂条件 |
| 0.6-0.8 | 高 | 高级查询 | 子查询、复杂计算 |
| 0.8-1.0 | 极高 | 超复杂查询 | 多层嵌套、大数据量 |

### 示例
```json
{
  "id": 2,
  "description": "计算各部门薪资排名",
  "dependencies": [1],
  "priority": 4,
  "estimated_complexity": 0.7  # 需要窗口函数或复杂排序
}
```

### 实际应用
```python
def estimate_execution_time(subproblem):
    """根据复杂度预估执行时间"""
    base_time = 1.0  # 基础时间（秒）
    return base_time * (1 + subproblem.estimated_complexity * 10)
```

---

## 综合示例

### 问题：找出销售额最高的产品类别及其平均利润率

```json
{
  "subproblems": [
    {
      "id": 1,
      "description": "获取所有销售记录",
      "dependencies": [],
      "priority": 5,
      "estimated_complexity": 0.3
    },
    {
      "id": 2,
      "description": "按产品类别计算总销售额",
      "dependencies": [1],
      "priority": 4,
      "estimated_complexity": 0.5
    },
    {
      "id": 3,
      "description": "找出销售额最高的产品类别",
      "dependencies": [2],
      "priority": 4,
      "estimated_complexity": 0.4
    },
    {
      "id": 4,
      "description": "计算该类别的平均利润率",
      "dependencies": [1, 3],
      "priority": 3,
      "estimated_complexity": 0.6
    }
  ]
}
```

### 执行顺序分析
1. **第一步**：执行ID=1（无依赖，最高优先级）
2. **第二步**：执行ID=2（依赖ID=1，高优先级）
3. **第三步**：执行ID=3（依赖ID=2，高优先级）
4. **第四步**：执行ID=4（依赖ID=1和ID=3，中等优先级）

### 总复杂度评估
- 简单任务（0.3）：1个
- 中等任务（0.4-0.5）：2个
- 复杂任务（0.6）：1个
- 平均复杂度：(0.3 + 0.5 + 0.4 + 0.6) / 4 = 0.45

这个分解方案确保了任务执行的逻辑性和高效性。