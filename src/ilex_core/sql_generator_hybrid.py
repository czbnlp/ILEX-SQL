"""
混合SQL生成器
先使用经验模式处理，失败后自动切换到探索模式
"""

import time
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
from .execution_memory import ExecutionMemory
from .problem_decomposer import ProblemDecomposer, SubProblem
from .exploration_engine_llm import LLMExplorationEngine
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))
from enhanced_sql_generator import EnhancedSQLGeneratorLPE
from master_sql_postprocessor import MasterSQLPostProcessor
from llm_connector_unified import UnifiedLLMConnector

class HybridSQLGenerator:
    """混合SQL生成器"""
    
    def __init__(self, 
                 config_path: str = "config/ilex_config.yaml",
                 llm_connector=None,
                 sql_executor=None,
                 validate_execution: bool = False):
        """
        初始化混合SQL生成器
        
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
                self.config = config.get('ilex', {}).get('hybrid', {})
        except Exception as e:
            self.config = {}
            self.logger.warning(f"加载配置文件失败: {e}，使用默认配置")
        
        # 初始化组件
        if llm_connector is None:
            # 如果没有提供连接器，使用统一连接器
            self.llm_connector = UnifiedLLMConnector(config_path)
            self.logger.info("使用统一LLM连接器")
        else:
            self.llm_connector = llm_connector
        
        # 初始化SQL执行器
        if sql_executor is None:
            raise ValueError("必须提供一个有效的SQL执行器(sql_executor)。")
        self.sql_executor = sql_executor
        
        # 初始化经验模式组件 (使用新的LPE-SQL风格实现)
        self.enhanced_sql_generator = EnhancedSQLGeneratorLPE(self.llm_connector)
        self.sql_postprocessor = MasterSQLPostProcessor()
        
        # 配置经验模式参数
        self.experience_config = self.config.get('experience_mode', {})
        self.top_k = self.experience_config.get('top_k', 4)
        self.correct_rate = self.experience_config.get('correct_rate', 0.5)
        self.engine = self.experience_config.get('engine', 'qwen2-72b')
        self.accumulate_knowledge_base = self.experience_config.get('accumulate_knowledge_base', True)
        self.use_init_knowledge_base = self.experience_config.get('use_init_knowledge_base', True)
        
        # 初始化探索模式组件 (支持LLM-based和rule-based两种模式)
        exploration_mode = self.config.get('exploration_mode', 'llm_based')
        
        self.exploration_engine = LLMExplorationEngine(
            config_path=config_path,
            llm_connector=self.llm_connector,
            sql_executor=sql_executor
        )
        self.logger.info("使用LLM-based探索模式")

        
        # 配置参数
        self.enable_exploration_fallback = self.config.get('enable_exploration_fallback', True)
        self.max_retries_experience_mode = self.config.get('max_retries_experience_mode', 2)
        self.exploration_timeout = self.config.get('exploration_timeout', 60)
        
        # 统计信息
        self.stats = {
            'total_questions': 0,
            'experience_mode_success': 0,
            'exploration_mode_used': 0,
            'exploration_mode_success': 0,
            'total_failures': 0
        }
    
    def generate_sql(self, 
                    question: str, 
                    db_path: str,
                    db_schema: Dict[str, Any] = None,
                    gold_sql: str = None) -> Tuple[str, bool, Dict[str, Any]]:
        """
        生成SQL查询（混合模式）
        
        Args:
            question: 自然语言问题
            db_path: 数据库路径
            db_schema: 数据库schema信息
            gold_sql: 标准SQL（用于结果验证）
            
        Returns:
            (最终SQL, 是否成功[基于执行结果], 详细信息)
        """
        self.stats['total_questions'] += 1
        start_time = time.time()
        
        self.logger.info(f"开始处理问题: {question[:100]}...")
        
        # 步骤1: 尝试经验模式
        self.logger.info("步骤1: 尝试使用经验模式生成SQL...")
        sql, success, details = self._try_experience_mode(question, db_path, db_schema)
        
        if success:
            self.stats['experience_mode_success'] += 1
            execution_time = time.time() - start_time
            details['execution_time'] = execution_time
            details['mode_used'] = 'experience'
            details['fallback_to_exploration'] = False
            self.logger.info("✓ 经验模式成功生成SQL")
            return sql, success, details
        
        # 步骤2: 如果经验模式失败，尝试探索模式
        if self.enable_exploration_fallback:
            self.logger.info("步骤2: 经验模式失败，切换到探索模式...")
            self.stats['exploration_mode_used'] += 1
            
            sql, success, details = self._try_exploration_mode(question, db_path, db_schema)
            
            if success:
                self.stats['exploration_mode_success'] += 1
                execution_time = time.time() - start_time
                details['execution_time'] = execution_time
                details['mode_used'] = 'exploration'
                details['fallback_to_exploration'] = True
                details['experience_mode_error'] = details.get('experience_mode_error')
                self.logger.info("✓ 探索模式成功生成SQL")
                return sql, success, details
            else:
                self.stats['total_failures'] += 1
                execution_time = time.time() - start_time
                details['execution_time'] = execution_time
                details['mode_used'] = 'exploration'
                details['fallback_to_exploration'] = True
                details['experience_mode_error'] = details.get('experience_mode_error')
                self.logger.error("✗ 探索模式也失败了")
                return sql, success, details
        else:
            # 如果不启用探索模式回退，直接返回经验模式的结果
            self.stats['total_failures'] += 1
            execution_time = time.time() - start_time
            details['execution_time'] = execution_time
            details['mode_used'] = 'experience'
            details['fallback_to_exploration'] = False
            self.logger.error("✗ 经验模式失败，且未启用探索模式回退")
            return sql, success, details
    
    def _try_experience_mode(self, 
                            question: str, 
                            db_path: str,
                            db_schema: Dict[str, Any] = None,
                            gold_sql: str = None) -> Tuple[str, bool, Dict[str, Any]]:
        """
        尝试使用经验模式生成SQL (LPE-SQL风格实现)
        
        Args:
            question: 自然语言问题
            db_path: 数据库路径
            db_schema: 数据库schema信息 (可选，新实现中不需要)
            gold_sql: 标准SQL（用于结果验证）
            
        Returns:
            (SQL, 是否成功[基于执行结果], 详细信息)
        """
        try:
            # 使用新的LPE-SQL风格生成器
            # 注意：新的生成器内部会处理经验检索、schema获取和SQL生成
            sql = self.enhanced_sql_generator.generate_sql(
                question=question,
                db_path=db_path,
                knowledge=None,  # 可以从配置或问题分析中获取
                correct_rate=self.correct_rate
            )
            
            print(f"\n{'*'*80}")
            print(f"LPE-SQL经验模式生成的SQL: {sql}")
            print(f"{'*'*80}\n")
            if sql and sql.strip():
                # 验证SQL语法
                validation = self.sql_postprocessor.validate_sql_syntax(sql, db_path)
                
                if validation['is_valid']:
                    # 执行SQL验证结果一致性
                    exec_result, exec_error = self.sql_executor(sql, db_path)
                    gold_result, _ = self.sql_executor(gold_sql, db_path)  # 需要从调用链获取gold_sql
                    is_correct = (exec_result == gold_result) if not exec_error else False
                    
                    return sql, is_correct, {
                        'raw_sql': sql,
                        'processed_sql': sql,
                        'validation': validation,
                        'execution_correct': is_correct,
                        'retries_used': 0,
                        'experience_mode': 'lpe_sql_style',
                        'retrieval_stats': self.enhanced_sql_generator.get_retrieval_stats()
                    }
                else:
                    # SQL语法验证失败
                    return "", False, {
                        'error': f"LPE-SQL经验模式生成的SQL验证失败: {validation['error']}",
                        'validation': validation,
                        'retrieval_stats': self.enhanced_sql_generator.get_retrieval_stats()
                    }
            else:
                # 生成的SQL为空
                return "", False, {
                    'error': "LPE-SQL经验模式未能生成有效的SQL查询",
                    'retrieval_stats': self.enhanced_sql_generator.get_retrieval_stats()
                }
                
        except Exception as e:
            self.logger.error(f"LPE-SQL经验模式生成SQL时发生错误: {e}")
            error_details = {
                'error': str(e),
                'retrieval_stats': self.enhanced_sql_generator.get_retrieval_stats() if hasattr(self, 'enhanced_sql_generator') else 'N/A'
            }
            return "", False, error_details
    
    def _try_exploration_mode(self, 
                             question: str, 
                             db_path: str,
                             db_schema: Dict[str, Any] = None,
                             gold_sql: str = None) -> Tuple[str, bool, Dict[str, Any]]:
        """
        尝试使用探索模式生成SQL
        
        Args:
            question: 自然语言问题
            db_path: 数据库路径
            db_schema: 数据库schema信息
            gold_sql: 标准SQL（用于结果验证）
            
        Returns:
            (SQL, 是否成功[基于执行结果], 详细信息)
        """
        try:
            # 如果没有提供schema，获取schema信息
            if db_schema is None:
                db_schema = self.enhanced_sql_generator.get_detailed_schema(db_path)
            
            # 将schema转换为文本格式
            schema_text = self.enhanced_sql_generator._build_schema_text(
                db_schema, list(db_schema['tables'].keys())
            )
            
            # 使用探索引擎解决问题
            sql, success, details = self.exploration_engine.solve_complex_question(
                question, db_path, schema_text
            )
            
            return sql, success, details
        
        except Exception as e:
            self.logger.error(f"探索模式生成SQL时发生错误: {e}")
            return "", False, {'error': str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取生成器统计信息
        
        Returns:
            统计信息字典
        """
        total = self.stats['total_questions']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'experience_mode_success_rate': self.stats['experience_mode_success'] / total,
            'exploration_mode_usage_rate': self.stats['exploration_mode_used'] / total,
            'exploration_mode_success_rate': (
                self.stats['exploration_mode_success'] / self.stats['exploration_mode_used']
                if self.stats['exploration_mode_used'] > 0 else 0
            ),
            'overall_success_rate': (
                (self.stats['experience_mode_success'] + self.stats['exploration_mode_success']) / total
            ),
            'failure_rate': self.stats['total_failures'] / total
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("\n=== 混合SQL生成器统计信息 ===")
        print(f"总问题数: {stats['total_questions']}")
        print(f"经验模式成功: {stats['experience_mode_success']}")
        print(f"探索模式使用次数: {stats['exploration_mode_used']}")
        print(f"探索模式成功: {stats['exploration_mode_success']}")
        print(f"总失败数: {stats['total_failures']}")
        print(f"经验模式成功率: {stats['experience_mode_success_rate']:.2%}")
        print(f"探索模式使用率: {stats['exploration_mode_usage_rate']:.2%}")
        print(f"探索模式成功率: {stats['exploration_mode_success_rate']:.2%}")
        print(f"总体成功率: {stats['overall_success_rate']:.2%}")
        print(f"失败率: {stats['failure_rate']:.2%}")
        print("=" * 40)


# 测试函数
def test_hybrid_sql_generator():
    """测试混合SQL生成器"""
    print("=== 测试混合SQL生成器 ===")
    
    # 这里需要实际的LLM连接器和SQL执行器
    # 为了测试，可以使用模拟版本
    from llm_connector_local import LocalLLMConnector
    from sql_executor import SQLExecutor
    
    try:
        llm_connector = LocalLLMConnector()
        sql_executor = SQLExecutor()
        
        generator = HybridSQLGenerator(
            llm_connector=llm_connector,
            sql_executor=sql_executor
        )
        
        # 测试问题
        test_questions = [
            "查询所有员工的信息",
            "找出薪水最高的员工",
            "First, find the department with the highest average salary, then list all employees in that department"
        ]
        
        for question in test_questions:
            print(f"\n测试问题: {question}")
            sql, success, details = generator.generate_sql(
                question, 
                "database.db"  # 需要实际的数据库路径
            )
            
            if success:
                print(f"✓ 成功生成SQL: {sql}")
                print(f"  使用模式: {details['mode_used']}")
                print(f"  执行时间: {details['execution_time']:.2f}秒")
            else:
                print(f"✗ 生成失败: {details.get('error', '未知错误')}")
        
        # 打印统计信息
        generator.print_statistics()
        
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    print(f"\n{'*'*80}")
    print(f"开始执行混合SQL生成器")
    print(f"{'*'*80}\n")
    test_hybrid_sql_generator()