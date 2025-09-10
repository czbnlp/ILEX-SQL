"""
LLM-based Exploration Engine
Completely refactored to use LLM-based processing instead of rule-based components
"""

import time
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from .execution_memory import ExecutionMemory, ExecutionRecord
import sys
import os
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent.parent.parent))
from enhanced_sql_generator import EnhancedSQLGeneratorLPE


@dataclass
class LLMSubProblem:
    """LLM生成的子问题数据类（简化版）"""
    id: int
    description: str
    dependencies: List[int]
    sql_template: Optional[str] = None
    reasoning: str = ""  # LLM生成的推理过程
    expected_output: str = ""  # LLM预期的输出描述


class LLMProblemDecomposer:
    """LLM-based问题分解器"""
    
    def __init__(self, llm_connector=None):
        """
        初始化LLM问题分解器
        
        Args:
            llm_connector: LLM连接器
        """
        self.llm_connector = llm_connector
        self.subproblem_counter = 0
        
        # 分解提示模板
        self.decomposition_prompt_template = """
You are an intelligent SQL query decomposition expert. Given a complex natural language question and database schema information, decompose it into smaller, manageable subproblems that can be solved step by step.

### Database Schema:
{schema}

### Original Question:
{question}

### Execution Context (Previous Results):
{context}

### Task:
1. Analyze the question type (sequential, conditional, comparative, or complex multi-step)
2. Identify the key components and dependencies
3. Decompose into logical subproblems
4. Determine the optimal execution order
5. Estimate complexity for each subproblem

### Requirements:
- Each subproblem should be self-contained and solvable
- Clearly specify dependencies between subproblems
- Provide SQL templates where applicable
- Include reasoning for each decomposition

### Output Format (JSON):
{{
    "question_type": "sequential|conditional|comparative|complex",
    "decomposition_reasoning": "Brief explanation of decomposition approach",
    "subproblems": [
        {{
            "id": 1,
            "description": "Clear description of what this subproblem solves",
            "dependencies": [0],
            "sql_template": "Optional SQL template or approach",
            "reasoning": "Why this subproblem is needed",
            "expected_output": "What result this should produce"
        }}
    ],
    "synthesis_approach": "How to combine results for final answer"
}}

### Response:
"""

        self.subproblem_selection_prompt = """
You are managing the execution of a complex SQL query exploration. Based on the current state, determine the next best subproblem to solve.

### Original Question:
{original_question}

### All Subproblems:
{all_subproblems}

### Solved Subproblems:
{solved_subproblems}

### Execution Context (Previous Results):
{context}

### Task:
1. Identify which subproblems are ready to be solved (dependencies satisfied)
2. Consider the execution order and priorities
3. Select the most appropriate next subproblem
4. Provide reasoning for the selection

### Requirements:
- Only select subproblems whose dependencies are all solved
- Consider priority and logical flow
- If multiple subproblems are ready, choose the one that provides most value
- If no subproblems can be solved, indicate completion

### Output Format (JSON):
{{
    "next_subproblem_id": 1,
    "reasoning": "Why this subproblem should be solved next", 
    "ready_subproblems": [1, 2, 3],
    "completion_status": "in_progress|completed|blocked"
}}

### Response:
"""

    def decompose_problem(self, question: str, context: str, db_schema: str) -> List[LLMSubProblem]:
        """
        使用LLM分解复杂问题
        
        Args:
            question: 原始问题
            context: 执行记忆上下文
            db_schema: 数据库schema
            
        Returns:
            子问题列表
        """
        prompt = self.decomposition_prompt_template.format(
            schema=db_schema,
            question=question,
            context=context
        )
        
        try:
            response = self.llm_connector(prompt)
            decomposition_result = self._parse_decomposition_response(response)
            
            subproblems = []
            for subproblem_data in decomposition_result.get('subproblems', []):
                subproblem = LLMSubProblem(
                    id=subproblem_data['id'],
                    description=subproblem_data['description'],
                    dependencies=subproblem_data.get('dependencies', []),
                    sql_template=subproblem_data.get('sql_template'),
                    reasoning=subproblem_data.get('reasoning', ''),
                    expected_output=subproblem_data.get('expected_output', '')
                )
                    subproblems.append(subproblem)
                
                return subproblems
            
        except Exception as e:
            logging.error(f"LLM问题分解失败: {e}")
            # 回退到简单的通用分解
            return self._fallback_decomposition(question, context, db_schema)
    
    def get_next_subproblem(self, original_question: str, context: str, solved_subproblems: List[int], all_subproblems: List[LLMSubProblem]) -> Optional[LLMSubProblem]:
        """
        使用LLM选择下一个要解决的子问题
        
        Args:
            original_question: 原始问题
            context: 执行上下文
            solved_subproblems: 已解决的子问题ID列表
            all_subproblems: 所有子问题
            
        Returns:
            下一个子问题，如果没有则返回None
        """
        all_subproblems_json = []
        for sp in all_subproblems:
            all_subproblems_json.append({
                'id': sp.id,
                'description': sp.description,
                'dependencies': sp.dependencies
            })
        
        prompt = self.subproblem_selection_prompt.format(
            original_question=original_question,
            all_subproblems=json.dumps(all_subproblems_json, ensure_ascii=False, indent=2),
            solved_subproblems=json.dumps(solved_subproblems),
            context=context
        )
        
        try:
            response = self.llm_connector(prompt)
            selection_result = self._parse_selection_response(response)
            
            if selection_result['completion_status'] == 'completed':
                return None
                
            next_id = selection_result['next_subproblem_id']
            for subproblem in all_subproblems:
                if subproblem.id == next_id:
                    return subproblem
                    
            return None
            
        except Exception as e:
            logging.error(f"LLM子问题选择失败: {e}")
            # 回退到简单的优先级选择
            return self._fallback_subproblem_selection(all_subproblems, solved_subproblems)
    
    def _parse_decomposition_response(self, response: str) -> Dict[str, Any]:
        """解析LLM分解响应"""
        try:
            # 尝试提取JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # 如果没有找到JSON，返回空结果
                return {'subproblems': []}
                
        except json.JSONDecodeError as e:
            logging.error(f"解析分解响应失败: {e}")
            return {'subproblems': []}
    
    def _parse_selection_response(self, response: str) -> Dict[str, Any]:
        """解析LLM选择响应"""
        try:
            # 尝试提取JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # 如果没有找到JSON，返回默认结果
                return {
                    'next_subproblem_id': 1,
                    'reasoning': 'Default selection',
                    'ready_subproblems': [],
                    'completion_status': 'in_progress'
                }
                
        except json.JSONDecodeError as e:
            logging.error(f"解析选择响应失败: {e}")
            return {
                'next_subproblem_id': 1,
                'reasoning': 'Default selection',
                'ready_subproblems': [],
                'completion_status': 'in_progress'
            }
    
    def _fallback_decomposition(self, question: str, context: str, db_schema: str) -> List[LLMSubProblem]:
        """回退分解方法"""
        # 简单的两阶段分解：先理解问题，再生成SQL
        return [
            LLMSubProblem(
                id=1,
                description=f"Understand the requirements and identify relevant tables/columns for: {question}",
                dependencies=[],
                sql_template=None,
                reasoning="First understand what data is needed",
                expected_output="List of relevant tables and columns"
            ),
            LLMSubProblem(
                id=2,
                description=f"Generate SQL query for: {question}",
                dependencies=[1],
                sql_template=None,
                reasoning="Then create the SQL based on understanding",
                expected_output="Final SQL query"
            )
        ]
    
    def _fallback_subproblem_selection(self, all_subproblems: List[LLMSubProblem], solved_subproblems: List[int]) -> Optional[LLMSubProblem]:
        """回退子问题选择"""
        # 选择第一个依赖已满足的未解决问题
        for subproblem in all_subproblems:
            if subproblem.id not in solved_subproblems:
                # 检查依赖是否都满足
                dependencies_satisfied = all(dep in solved_subproblems for dep in subproblem.dependencies)
                if dependencies_satisfied:
                    return subproblem
        return None


class LLMExecutionMemory:
    """LLM-based执行记忆管理"""
    
    def __init__(self, max_size: int = 10):
        """
        初始化LLM执行记忆
        
        Args:
            max_size: 最大记忆容量
        """
        self.max_size = max_size
        self.execution_records = []
        
        # LLM上下文生成提示
        self.context_prompt = """
You are managing the execution memory for a complex SQL query exploration system. Format the execution history into a clear context for the next exploration step.

### Execution History:
{execution_history}

### Task:
1. Summarize what has been accomplished so far
2. Identify what information is still needed
3. Highlight any errors or issues to learn from
4. Provide context for the next exploration step

### Requirements:
- Be concise but comprehensive
- Focus on actionable insights
- Highlight dependencies and relationships
- Note any patterns or learnings

### Output Format:
Provide a clear, structured summary that can be used as context for the next exploration step.

### Summary:
"""

        self.completion_prompt = """
You are evaluating whether a complex SQL query exploration has successfully answered the original question.

### Original Question:
{original_question}

### Execution History:
{execution_history}

### Task:
1. Analyze if the original question has been answered
2. Consider the quality and completeness of results
3. Evaluate if further exploration is needed
4. Provide confidence assessment

### Requirements:
- Consider both successful and failed attempts
- Evaluate if the results address the core question
- Assess if additional information is needed
- Be conservative in declaring completion

### Output Format (JSON):
{{
    "is_complete": true/false,
    "reasoning": "Explanation of completion assessment",
    "remaining_gaps": ["Any remaining information gaps"],
    "suggestions": ["Suggestions for improvement if needed"]
}}

### Response:
"""

    def get_context_for_prompt(self, llm_connector) -> str:
        """使用LLM生成优化的上下文"""
        if not self.execution_records:
            return "No previous execution history available."
        
        history_text = []
        for i, record in enumerate(self.execution_records):
            status = "✓" if record.success else "✗"
            history_text.append(f"{status} Step {i+1}: {record.subquestion}")
            history_text.append(f"  Result: {record.result}")
            if record.sql:
                history_text.append(f"  SQL: {record.sql}")
            if record.error:
                history_text.append(f"  Error: {record.error}")
        
        prompt = self.context_prompt.format(
            execution_history="\n".join(history_text)
        )
        
        try:
            response = llm_connector(prompt)
            return response.strip()
        except Exception as e:
            logging.error(f"LLM上下文生成失败: {e}")
            # 回退到简单的格式化
            return "\n".join(history_text)
    
    def is_exploration_complete(self, original_question: str, llm_connector) -> Tuple[bool, float]:
        """使用LLM判断探索是否完成"""
        if not self.execution_records:
            return False, 0.0
        
        context = self.get_context_for_prompt(llm_connector)
        prompt = self.completion_prompt.format(
            original_question=original_question,
            execution_history=context
        )
        
        try:
            response = llm_connector(prompt)
            result = self._parse_completion_response(response)
            
            return result.get('is_complete', False), result.get('confidence', 0.0)
            
        except Exception as e:
            logging.error(f"LLM完成判断失败: {e}")
            # 回退到基于成功率的简单判断
            return self._fallback_completion_check(original_question)
    
    def _parse_completion_response(self, response: str) -> Dict[str, Any]:
        """解析LLM完成判断响应"""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # 简单的关键词判断
                response_lower = response.lower()
                is_complete = 'complete' in response_lower and 'true' in response_lower
                confidence = 0.7 if is_complete else 0.3
                
                return {
                    'is_complete': is_complete,
                    'confidence': confidence,
                    'reasoning': 'Simple keyword-based assessment'
                }
                
        except json.JSONDecodeError as e:
            logging.error(f"解析完成响应失败: {e}")
            return {'is_complete': False, 'confidence': 0.0}
    
    def _fallback_completion_check(self, original_question: str) -> Tuple[bool, float]:
        """回退完成判断"""
        if not self.execution_records:
            return False, 0.0
        
        # 基于最近的成功记录
        recent_records = self.execution_records[-3:]
        success_count = sum(1 for record in recent_records if record.success)
        
        # 如果最近3次都成功，认为基本完成
        if success_count >= 3:
            return True, 0.8
        elif success_count >= 2:
            return True, 0.6
        else:
            return False, 0.3
    
    def add(self, subquestion: str, result: str, sql: str = None, error: str = None, execution_time: float = 0):
        """添加执行记录"""
        record = ExecutionRecord(
            subquestion=subquestion,
            result=result,
            sql=sql,
            error=error,
            execution_time=execution_time,
            success=error is None and result is not None
        )
        
        self.execution_records.append(record)
        
        # 保持记忆大小限制
        if len(self.execution_records) > self.max_size:
            self.execution_records.pop(0)
    
    def clear_memory(self):
        """清空执行记忆"""
        self.execution_records.clear()


class LLMExplorationEngine:
    """LLM-based探索引擎"""
    
    def __init__(self, 
                 config_path: str = "config/ilex_config.yaml",
                 llm_connector=None,
                 sql_executor=None):
        """
        初始化LLM探索引擎
        
        Args:
            config_path: 配置文件路径
            llm_connector: LLM连接器
            sql_executor: SQL执行器
        """
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.config = config.get('ilex', {}).get('exploration', {})
        except Exception as e:
            self.config = {}
            self.logger.warning(f"加载配置文件失败: {e}，使用默认配置")
        
        # 初始化LLM组件
        self.execution_memory = LLMExecutionMemory(
            max_size=self.config.get('execution_memory_size', 10)
        )
        self.problem_decomposer = LLMProblemDecomposer(llm_connector)  # 使用实现类
        self.llm_connector = llm_connector
        self.sql_executor = sql_executor
        
        # 配置参数
        self.max_exploration_steps = self.config.get('max_exploration_steps', 5)
        self.timeout_per_step = self.config.get('timeout_per_step', 30)
        self.max_retries_per_subquestion = self.config.get('max_retries_per_subquestion', 3)
        self.enable_intermediate_validation = self.config.get('enable_intermediate_validation', True)
        
        # 统计信息
        self.exploration_stats = {
            'total_explorations': 0,
            'successful_explorations': 0,
            'failed_explorations': 0,
            'average_steps': 0,
            'average_time': 0
        }
        
        # SQL生成提示
        self.sql_generation_prompt = """
You are an expert SQL query generator working within an exploration system. Generate a SQL query to solve the given subproblem.

### Database Schema:
{schema}

### Subproblem:
{subproblem_description}

### Execution Context (Previous Results):
{context}

### Reasoning:
{reasoning}

### Expected Output:
{expected_output}

### Task:
Generate a syntactically correct SQL query that solves this subproblem.

### Requirements:
- Use proper SQL syntax for the database type
- Reference the correct table and column names from the schema
- Handle any special characters in column names appropriately
- Make the query efficient and readable
- Only return the SQL query, no explanations

### SQL Query:
"""

        self.synthesis_prompt = """
You are synthesizing the final SQL query from multiple exploration steps. Combine the results from solved subproblems to answer the original question.

### Original Question:
{original_question}

### Database Schema:
{schema}

### Execution History (All Subproblem Results):
{execution_history}

### Task:
1. Analyze what has been discovered through exploration
2. Identify the key information needed to answer the original question
3. Synthesize a final SQL query that combines all relevant results
4. Ensure the query directly addresses the original question

### Requirements:
- Use information from all successful subproblems
- Create a coherent final query
- Handle any necessary joins, aggregations, or filtering
- Ensure the query is syntactically correct

### Final SQL Query:
"""

    def solve_complex_question(self, 
                             original_question: str, 
                             db_path: str,
                             db_schema: str = "") -> Tuple[str, bool, Dict[str, Any]]:
        """
        使用LLM解决复杂问题的主流程
        
        Args:
            original_question: 原始问题
            db_path: 数据库路径
            db_schema: 数据库schema信息
            
        Returns:
            (最终SQL, 是否成功, 详细信息)
        """
        start_time = time.time()
        self.exploration_stats['total_explorations'] += 1
        
        self.logger.info(f"开始LLM探索模式解决复杂问题: {original_question[:100]}...")
        
        # 重置执行记忆
        self.execution_memory.clear_memory()
        
        try:
            # 使用LLM分解问题
            self.logger.info("使用LLM分解复杂问题...")
            all_subproblems = self.problem_decomposer.decompose_problem(
                original_question,
                "",  # 初始上下文为空
                db_schema
            )
            
            self.logger.info(f"问题分解完成，共{len(all_subproblems)}个子问题")
            
            # 探索主循环
            step = 0
            solved_subproblems = []
            
            while step < self.max_exploration_steps:
                self.logger.info(f"LLM探索步骤 {step + 1}/{self.max_exploration_steps}")
                
                # 使用LLM选择下一个子问题
                context = self.execution_memory.get_context_for_prompt(self.llm_connector)
                next_subproblem = self.problem_decomposer.get_next_subproblem(
                    original_question,
                    context,
                    solved_subproblems,
                    all_subproblems
                )
                
                if next_subproblem is None:
                    self.logger.info("LLM确定没有更多可解决的子问题")
                    break
                
                # 解决子问题
                subproblem_result = self._solve_subproblem(
                    next_subproblem,
                    db_path,
                    db_schema
                )
                
                # 记录执行结果
                self.execution_memory.add(
                    subquestion=next_subproblem.description,
                    result=subproblem_result['result'],
                    sql=subproblem_result.get('sql'),
                    error=subproblem_result.get('error'),
                    execution_time=subproblem_result.get('execution_time', 0)
                )
                
                if subproblem_result['success']:
                    solved_subproblems.append(next_subproblem.id)
                    self.logger.info(f"LLM子问题解决成功: {next_subproblem.description[:50]}...")
                else:
                    self.logger.warning(f"LLM子问题解决失败: {next_subproblem.description[:50]}...")
                
                # 使用LLM检查是否完成
                is_complete, confidence = self.execution_memory.is_exploration_complete(
                    original_question,
                    self.llm_connector
                )
                
                if is_complete and confidence > 0.7:
                    self.logger.info(f"LLM判断探索完成 (置信度: {confidence:.2f})")
                    break
                
                step += 1
            
            # 使用LLM合成最终SQL
            final_sql, synthesis_success = self._synthesize_final_sql(
                original_question, db_path, db_schema
            )
            
            # 计算执行时间
            execution_time = time.time() - start_time
            
            if synthesis_success:
                self.exploration_stats['successful_explorations'] += 1
                self.logger.info("LLM探索模式成功生成最终SQL")
                
                return final_sql, True, {
                    'exploration_steps': step + 1,
                    'solved_subproblems': len(solved_subproblems),
                    'total_subproblems': len(all_subproblems),
                    'execution_time': execution_time,
                    'final_sql': final_sql,
                    'synthesis_success': True,
                    'exploration_mode': 'llm_based'
                }
            else:
                self.exploration_stats['failed_explorations'] += 1
                self.logger.error("LLM探索模式合成最终SQL失败")
                
                return "", False, {
                    'error': 'Failed to synthesize final SQL',
                    'execution_time': execution_time,
                    'exploration_steps': step + 1,
                    'solved_subproblems': len(solved_subproblems),
                    'synthesis_success': False
                }
                
        except Exception as e:
            self.exploration_stats['failed_explorations'] += 1
            self.logger.error(f"LLM探索模式发生错误: {e}")
            
            return "", False, {
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _solve_subproblem(self, subproblem: LLMSubProblem, db_path: str, db_schema: str) -> Dict[str, Any]:
        """
        使用LLM解决子问题
        
        Args:
            subproblem: 子问题
            db_path: 数据库路径
            db_schema: 数据库schema
            
        Returns:
            解决结果
        """
        start_time = time.time()
        
        try:
            # 生成上下文
            context = self.execution_memory.get_context_for_prompt(self.llm_connector)
            
            # 构建SQL生成提示
            prompt = self.sql_generation_prompt.format(
                schema=db_schema,
                subproblem_description=subproblem.description,
                context=context,
                reasoning=subproblem.reasoning,
                expected_output=subproblem.expected_output
            )
            
            # 生成SQL
            sql_response = self.llm_connector(prompt)
            sql = self._extract_sql_from_response(sql_response)
            
            if not sql:
                return {
                    'success': False,
                    'error': 'Failed to generate SQL for subproblem',
                    'execution_time': time.time() - start_time
                }
            
            # 执行SQL并验证
            if self.sql_executor:
                execution_result = self.sql_executor.execute_query(sql, db_path)
                
                if execution_result['success']:
                    return {
                        'success': True,
                        'result': str(execution_result['data']),
                        'sql': sql,
                        'execution_time': time.time() - start_time
                    }
                else:
                    return {
                        'success': False,
                        'error': execution_result['error'],
                        'sql': sql,
                        'execution_time': time.time() - start_time
                    }
            else:
                # 如果没有SQL执行器，只返回生成的SQL
                return {
                    'success': True,
                    'result': f"Generated SQL: {sql}",
                    'sql': sql,
                    'execution_time': time.time() - start_time
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _synthesize_final_sql(self, original_question: str, db_path: str, db_schema: str) -> Tuple[str, bool]:
        """
        使用LLM合成最终SQL
        
        Args:
            original_question: 原始问题
            db_path: 数据库路径
            db_schema: 数据库schema
            
        Returns:
            (最终SQL, 是否成功)
        """
        try:
            # 生成执行历史上下文
            execution_history = self.execution_memory.get_context_for_prompt(self.llm_connector)
            
            # 构建合成提示
            prompt = self.synthesis_prompt.format(
                original_question=original_question,
                schema=db_schema,
                execution_history=execution_history
            )
            
            # 生成最终SQL
            final_sql_response = self.llm_connector(prompt)
            final_sql = self._extract_sql_from_response(final_sql_response)
            
            if final_sql:
                return final_sql, True
            else:
                return "", False
                
        except Exception as e:
            logging.error(f"LLM最终SQL合成失败: {e}")
            return "", False
    
    def _extract_sql_from_response(self, response: str) -> str:
        """从LLM响应中提取SQL"""
        import re
        
        # 尝试多种SQL提取模式
        sql_patterns = [
            r'SELECT.*?(?:;|$)',
            r'select.*?(?:;|$)',
            r'```sql\s*(SELECT.*?)\s*```',
            r'```(SELECT.*?)```'
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            if matches:
                sql = matches[0].strip()
                if not sql.endswith(';'):
                    sql += ';'
                return sql
        
        # 如果没有找到SQL模式，返回整个响应
        return response.strip()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取探索统计信息"""
        return self.exploration_stats.copy()