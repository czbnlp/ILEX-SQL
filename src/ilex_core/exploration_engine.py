"""
探索引擎模块
执行探索模式的完整流程
"""

import time
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
from .execution_memory import ExecutionMemory, ExecutionRecord
from .problem_decomposer_fixed import ProblemDecomposer, SubProblem
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))
from enhanced_sql_generator import EnhancedSQLGenerator

class ExplorationEngine:
    """探索引擎类"""
    
    def __init__(self, 
                 config_path: str = "config/ilex_config.yaml",
                 llm_connector=None,
                 sql_executor=None):
        """
        初始化探索引擎
        
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
        
        # 初始化组件
        self.execution_memory = ExecutionMemory(
            max_size=self.config.get('execution_memory_size', 10),
            config_path=config_path
        )
        self.problem_decomposer = ProblemDecomposer(config_path)
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
    
    def solve_complex_question(self, 
                             original_question: str, 
                             db_path: str,
                             db_schema: str = "") -> Tuple[str, bool, Dict[str, Any]]:
        """
        解决复杂问题的主流程
        
        Args:
            original_question: 原始问题
            db_path: 数据库路径
            db_schema: 数据库schema信息
            
        Returns:
            (最终SQL, 是否成功, 详细信息)
        """
        start_time = time.time()
        self.exploration_stats['total_explorations'] += 1
        
        self.logger.info(f"开始探索模式解决复杂问题: {original_question[:100]}...")
        
        # 重置执行记忆
        self.execution_memory.clear_memory()
        
        try:
            # 探索主循环
            step = 0
            solved_subproblems = []
            
            while step < self.max_exploration_steps:
                self.logger.info(f"探索步骤 {step + 1}/{self.max_exploration_steps}")
                
                # 获取下一个子问题
                next_subproblem = self.problem_decomposer.get_next_subproblem(
                    original_question,
                    self.execution_memory.get_context_for_prompt(),
                    solved_subproblems
                )
                
                if next_subproblem is None:
                    self.logger.info("没有更多可解决的子问题")
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
                    self.logger.info(f"子问题解决成功: {next_subproblem.description[:50]}...")
                else:
                    self.logger.warning(f"子问题解决失败: {next_subproblem.description[:50]}...")
                
                # 检查是否完成
                if self._is_exploration_complete(original_question):
                    self.logger.info("探索完成")
                    break
                
                step += 1
            
            # 生成最终SQL
            final_sql, synthesis_success = self._synthesize_final_sql(
                original_question, db_path, db_schema
            )
            
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 更新统计信息
            if synthesis_success:
                self.exploration_stats['successful_explorations'] += 1
            else:
                self.exploration_stats['failed_explorations'] += 1
            
            self.exploration_stats['average_steps'] = (
                (self.exploration_stats['average_steps'] * (self.exploration_stats['total_explorations'] - 1) + step) 
                / self.exploration_stats['total_explorations']
            )
            self.exploration_stats['average_time'] = (
                (self.exploration_stats['average_time'] * (self.exploration_stats['total_explorations'] - 1) + execution_time)
                / self.exploration_stats['total_explorations']
            )
            
            details = {
                'steps_taken': step,
                'execution_time': execution_time,
                'solved_subproblems': solved_subproblems,
                'memory_summary': self.execution_memory.get_summary(),
                'exploration_path': self._get_exploration_path()
            }
            
            return final_sql, synthesis_success, details
            
        except Exception as e:
            self.logger.error(f"探索过程中发生错误: {e}")
            error_time = time.time() - start_time
            self.exploration_stats['failed_explorations'] += 1
            
            return "", False, {
                'error': str(e),
                'execution_time': error_time,
                'steps_taken': step
            }
    def _solve_subproblem(self, 
                         subproblem: SubProblem, 
                         db_path: str,
                         db_schema: str) -> Dict[str, Any]:
        """
        解决单个子问题
        
        Args:
            subproblem: 子问题
            db_path: 数据库路径
            db_schema: 数据库schema信息
            
        Returns:
            解决结果字典
        """
        self.logger.info(f"解决子问题: {subproblem.description}")
        
        # 重试机制
        max_retries = getattr(self, 'max_retries_per_subquestion', 3)
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # 使用增强的SQL生成器
                if self.llm_connector:
                    enhanced_sql_generator = EnhancedSQLGenerator(self.llm_connector)
                    detailed_schema = enhanced_sql_generator.get_detailed_schema(db_path)
                    sql = enhanced_sql_generator.generate_sql_with_schema(subproblem.description, detailed_schema, db_path)
                else:
                    # 如果没有LLM连接器，使用简单的SQL生成
                    sql = self._generate_simple_sql(subproblem.description)
                
                # 执行SQL
                if self.sql_executor and sql:
                    result, error = self.sql_executor(sql, db_path)
                else:
                    # 模拟执行
                    result = f"模拟结果: {subproblem.description}"
                    error = None
                
                execution_time = time.time() - start_time
                
                # 检查执行结果
                success = error is None
                
                if success or attempt == self.max_retries_per_subquestion - 1:
                    return {
                        'success': success,
                        'result': result,
                        'sql': sql,
                        'error': error,
                        'execution_time': execution_time,
                        'attempts': attempt + 1
                    }
                
                # 如果失败且还有重试机会，等待后重试
                time.sleep(1)
                
            except Exception as e:
                self.logger.warning(f"子问题解决尝试 {attempt + 1} 失败: {e}")
                if attempt == self.max_retries_per_subproblem - 1:
                    return {
                        'success': False,
                        'result': None,
                        'sql': None,
                        'error': str(e),
                        'execution_time': 0,
                        'attempts': attempt + 1
                    }
        
        return {
            'success': False,
            'result': None,
            'sql': None,
            'error': 'Max retries exceeded',
            'execution_time': 0,
            'attempts': self.max_retries_per_subproblem
        }
    
    def _generate_subproblem_prompt(self, subproblem: SubProblem, db_schema: str) -> str:
        """
        生成子问题解决提示
        
        Args:
            subproblem: 子问题
            db_schema: 数据库schema信息
            
        Returns:
            提示字符串
        """
        memory_context = self.execution_memory.get_context_for_prompt()
        
        prompt = f"""基于当前执行记忆，解决以下子问题：

子问题: {subproblem.description}
"""
        
        if memory_context:
            prompt += f"""
执行记忆中的相关信息:
{memory_context}
"""
        
        if db_schema:
            prompt += f"""
数据库Schema:
{db_schema}
"""
        
        prompt += """
请生成SQL查询来解决这个子问题。注意考虑执行记忆中的中间结果。
只返回SQL语句，不要包含其他解释。
"""
        
        return prompt
    
    def _extract_sql_from_response(self, response: str) -> str:
        """
        从LLM响应中提取SQL
        
        Args:
            response: LLM响应
            
        Returns:
            SQL字符串
        """
        # 查找SELECT语句
        import re
        
        # 尝试匹配SQL语句
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
                # 确保以分号结尾
                if not sql.endswith(';'):
                    sql += ';'
                return sql
        
        # 如果没有找到SQL模式，返回整个响应（可能已经是SQL）
        return response.strip()
    
    def _generate_simple_sql(self, subproblem_description: str) -> str:
        """
        生成简单SQL（备用方案）
        
        Args:
            subproblem_description: 子问题描述
            
        Returns:
            简单SQL语句
        """
        # 这是一个非常简化的SQL生成器
        # 在实际应用中，应该使用更复杂的逻辑或调用LLM
        
        description_lower = subproblem_description.lower()
        
        if 'count' in description_lower:
            return "SELECT COUNT(*) FROM table_name;"
        elif 'average' in description_lower or 'avg' in description_lower:
            return "SELECT AVG(column_name) FROM table_name;"
        elif 'maximum' in description_lower or 'max' in description_lower:
            return "SELECT MAX(column_name) FROM table_name;"
        elif 'minimum' in description_lower or 'min' in description_lower:
            return "SELECT MIN(column_name) FROM table_name;"
        else:
            return "SELECT * FROM table_name LIMIT 10;"
    
    def _synthesize_final_sql(self, 
                             original_question: str, 
                             db_path: str,
                             db_schema: str) -> Tuple[str, bool]:
        """
        合成最终SQL
        
        Args:
            original_question: 原始问题
            db_path: 数据库路径
            db_schema: 数据库schema信息
            
        Returns:
            (最终SQL, 是否成功)
        """
        self.logger.info("合成最终SQL")
        
        # 生成合成提示
        prompt = self._generate_synthesis_prompt(original_question, db_schema)
        
        try:
            if self.llm_connector:
                response = self.llm_connector(prompt)
                final_sql = self._extract_sql_from_response(response)
            else:
                # 简单的合成逻辑
                final_sql = self._simple_synthesis(original_question)
            
            # 验证生成的SQL
            if self.sql_executor:
                result, error = self.sql_executor(final_sql, db_path)
                success = error is None
            else:
                success = True  # 模拟成功
            
            return final_sql, success
            
        except Exception as e:
            self.logger.error(f"SQL合成失败: {e}")
            return "", False
    
    def _generate_synthesis_prompt(self, original_question: str, db_schema: str) -> str:
        """
        生成SQL合成提示
        
        Args:
            original_question: 原始问题
            db_schema: 数据库schema信息
            
        Returns:
            提示字符串
        """
        memory_context = self.execution_memory.get_context_for_prompt()
        
        prompt = f"""基于完整的执行记忆，为原始问题生成最终的SQL查询。

原始问题: {original_question}
"""
        
        if memory_context:
            prompt += f"""
完整的执行记忆:
{memory_context}
"""
        
        if db_schema:
            prompt += f"""
数据库Schema:
{db_schema}
"""
        
        prompt += """
请整合执行记忆中的所有中间结果，生成解决原始问题的最终SQL查询。
确保SQL语句语法正确，并且能够充分利用前面步骤的结果。
只返回SQL语句，不要包含其他解释。
"""
        
        return prompt
    
    def _simple_synthesis(self, original_question: str) -> str:
        """
        简单的SQL合成（备用方案）
        
        Args:
            original_question: 原始问题
            
        Returns:
            合成的SQL
        """
        # 这是一个简化的合成逻辑
        # 在实际应用中，应该使用更复杂的逻辑或调用LLM
        
        # 基于执行记忆生成一个通用的SQL
        return "SELECT * FROM table_name WHERE condition;"
    
    def _is_exploration_complete(self, original_question: str) -> bool:
        """
        判断探索是否完成
        
        Args:
            original_question: 原始问题
            
        Returns:
            是否完成
        """
        return self.execution_memory.is_question_resolved(original_question)
    
    def _get_exploration_path(self) -> List[Dict[str, Any]]:
        """
        获取探索路径
        
        Returns:
            探索路径列表
        """
        path = []
        for record in self.execution_memory.memory:
            path.append({
                'step': record.step_number,
                'subquestion': record.subquestion,
                'result': record.result,
                'success': record.success,
                'execution_time': record.execution_time
            })
        return path
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取探索引擎统计信息
        
        Returns:
            统计信息字典
        """
        return {
            **self.exploration_stats,
            'current_memory_size': len(self.execution_memory),
            'memory_stats': self.execution_memory.get_execution_statistics()
        }
    
    def reset(self):
        """重置探索引擎状态"""
        self.execution_memory.clear_memory()
        self.problem_decomposer.reset_counter()
        self.logger.info("探索引擎已重置")
    
    def save_exploration_history(self, filepath: str):
        """
        保存探索历史
        
        Args:
            filepath: 文件路径
        """
        history = {
            'exploration_stats': self.exploration_stats,
            'execution_memory': [record.__dict__ for record in self.execution_memory.memory],
            'timestamp': time.time()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            self.logger.info(f"探索历史已保存到: {filepath}")
        except Exception as e:
            self.logger.error(f"保存探索历史失败: {e}")


# 测试函数
def test_exploration_engine():
    """测试探索引擎功能"""
    print("=== 探索引擎测试 ===")
    
    # 创建探索引擎
    engine = ExplorationEngine()
    
    # 模拟LLM连接器
    def mock_llm_connector(prompt):
        return "SELECT * FROM table_name WHERE condition;"
    
    # 模拟SQL执行器
    def mock_sql_executor(sql, db_path):
        return f"执行结果: {sql}", None
    
    engine.llm_connector = mock_llm_connector
    engine.sql_executor = mock_sql_executor
    
    # 测试复杂问题
    test_question = "First, find the manager with the highest salary, then find all employees in the same department."
    
    print(f"测试问题: {test_question}")
    
    # 执行探索
    final_sql, success, details = engine.solve_complex_question(
        test_question, 
        "test_db_path", 
        "test_schema"
    )
    
    print(f"最终SQL: {final_sql}")
    print(f"是否成功: {success}")
    print(f"详细信息: {details}")
    print(f"统计信息: {engine.get_statistics()}")


if __name__ == "__main__":
    test_exploration_engine()