#!/usr/bin/env python3
"""
问题分解器模块 (LLM-Based)
将复杂问题分解为可管理的子问题，完全由大语言模型驱动。
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from llm_connector_unified import UnifiedLLMConnector

@dataclass
class SubProblem:
    """子问题数据类（简化版）"""
    id: int
    description: str
    dependencies: List[int]
    sql_template: Optional[str] = None

DECOMPOSITION_PROMPT = """
作为专业数据库问题分解专家，请将以下问题分解为逻辑子问题：

**数据库Schema**:
{schema}

**执行上下文**:
{context}

**原始问题**:
{question}

请生成包含以下字段的JSON数组：
1. description: 子问题描述
2. dependencies: 依赖的子问题ID列表
3. sql_template: (可选) SQL模板

示例输出格式：
```json
{
  "sub_problems": [
    {
      "description": "计算每个部门的平均薪资",
      "dependencies": [],
      "sql_template": "SELECT department, AVG(salary) FROM employees GROUP BY department"
    }
  ]
}
```
"""

class LLMProblemDecomposer:
    def __init__(self, llm_connector=None):
        self.llm_connector = llm_connector or UnifiedLLMConnector()
        self.logger = logging.getLogger(__name__)
        self.counter = 0

    def decompose_problem(self, question: str, context: str = "", schema: str = "") -> List[SubProblem]:
        """使用LLM分解复杂问题"""
        prompt = DECOMPOSITION_PROMPT.format(
            question=question,
            context=context,
            schema=schema
        )
        
        print(f"\n{'*'*80}")
        print(f"开始执行问题分解器")
        print(f"{'*'*80}\n")
        
        try:
            response = self.llm_connector(prompt)
            result = self._parse_response(response)
            return self._create_subproblems(result)
        except Exception as e:
            self.logger.error(f"分解失败: {e}")
            return [SubProblem(
                id=0,
                description=question,
                dependencies=[],
                sql_template=None
            )]

    def _parse_response(self, response: str) -> List[Dict]:
        """解析LLM响应（更健壮的版本）"""
        try:
            # 尝试提取JSON部分
            json_str = response
            for marker in ['```json', '```']:
                if marker in json_str:
                    json_str = json_str.split(marker)[1] if marker == '```json' else json_str.split(marker)[0]
            
            # 清理可能的语法错误
            json_str = json_str.strip().replace('\n', '').replace('\\"', '"')
            
            # 尝试解析
            data = json.loads(json_str)
            if isinstance(data, dict) and "sub_problems" in data:
                return data["sub_problems"]
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败，尝试修复格式... 原始响应: {response[:200]}")
            try:
                # 尝试修复常见格式错误
                fixed = response.replace("'", '"').replace("True", "true").replace("False", "false")
                return json.loads(fixed)["sub_problems"]
            except:
                return []
        except Exception as e:
            self.logger.error(f"响应解析错误: {e}")
            return []

    def _create_subproblems(self, problems: List[Dict]) -> List[SubProblem]:
        """创建子问题对象（简化版）"""
        subproblems = []
        for p in problems:
            subproblems.append(SubProblem(
                id=self.counter,
                description=p["description"],
                dependencies=p["dependencies"],
                sql_template=p.get("sql_template")
            ))
            self.counter += 1
        return subproblems

# 保持接口兼容性
ProblemDecomposer = LLMProblemDecomposer