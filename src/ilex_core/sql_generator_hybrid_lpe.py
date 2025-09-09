"""
混合SQL生成器 - LPE-SQL 风格实现
先使用LPE经验模式处理，失败后自动切换到探索模式
"""

import time
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
from .execution_memory import ExecutionMemory
from .problem_decomposer_fixed import ProblemDecomposer, SubProblem
from .exploration_engine import ExplorationEngine
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))
from enhanced_sql_generator_lpe import EnhancedSQLGeneratorLPE
from master_sql_postprocessor import MasterSQLPostProcessor

class HybridSQLGeneratorLPE:
    """混合SQL生成器 - LPE版本"""
    
    def __init__(self, 
                 config_path: str = "config/ilex_config.yaml",
                 llm_connector=None,
                 sql_executor=None,
                 experience_retriever=None):
        """
        初始化混合SQL生成器
        
        Args:
            config_path: 配置文件路径
            llm_connector: LLM连接器
            sql_executor: SQL执行器
            experience_retriever: 经验检索器
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
        self.llm_connector = llm_connector
        self.sql_executor = sql_executor
        
        # 初始化LPE经验模式组件
        self.enhanced_sql_generator = EnhancedSQLGeneratorLPE(
            llm_connector=llm_connector,
            experience_retriever=experience_retriever
        )
        self.sql_postprocessor = MasterSQLPostProcessor()
        
        # 初始化探索模式组件
        self.exploration_engine = ExplorationEngine(
            config_path=config_path,
            llm_connector=llm_connector,
            sql_executor=sql_executor
        )
        
        # 配置参数
        self.enable_exploration_fallback = self.config.get('enable_exploration_fallback', True)
        self.max_retries_experience_mode = self.config.get('max_retries_experience_mode', 2)
        self.exploration_timeout = self.config.get('exploration_timeout', 60)
        self.accumulate_knowledge = self.config.get('accumulate_knowledge', True)
        
        # 统计信息
        self.stats = {
            'total_questions': 0,
            'experience_mode_success': 0,
            'exploration_mode_used': 0,
            'exploration_mode_success': 0,
            'total_failures': 0,
            'knowledge_accumulated': 0
        }
    
    def generate_sql(self, 
                    question: str, 
                    db_path: str,
                    db_schema: Dict[str, Any] = None) -> Tuple[str, bool, Dict[str, Any]]:
        """
        生成SQL查询（混合模式 - LPE版本）
        
        Args:
            question: 自然语言问题
            db_path: 数据库路径
            db_schema: 数据库schema信息
            
        Returns:
            (最终SQL, 是否成功, 详细信息)
        """
        self.stats['total_questions'] += 1
        start_time = time.time()
        
        self.logger.info(f"开始处理问题: {question[:100]}...")
        
        # 步骤1: 尝试LPE经验模式
        self.logger.info("步骤1: 尝试使用LPE经验模式生成SQL...")
        sql, success, details = self._try_lpe_experience_mode(question, db_path, db_schema)
        
        if success:
            self.stats['experience_mode_success'] += 1
            execution_time = time.time() - start_time
            details['execution_time'] = execution_time
            details['mode_used'] = 'lpe_experience'
            details['fallback_to_exploration'] = False
            self.logger.info("✓ LPE经验模式成功生成SQL")
            return sql, success, details
        
        # 步骤2: 如果经验模式失败，尝试探索模式
        if self.enable_exploration_fallback:
            self.logger.info("步骤2: LPE经验模式失败，切换到探索模式...")
            self.stats['exploration_mode_used'] += 1
            
            sql, success, details = self._try_exploration_mode(question, db_path, db_schema)
            
            if success:
                self.stats['exploration_mode_success'] += 1
                execution_time = time.time() - start_time
                details['execution_time'] = execution_time
                details['mode_used'] = 'exploration'
                details['fallback_to_exploration'] = True
                details['lpe_experience_error'] = details.get('lpe_experience_error')
                self.logger.info("✓ 探索模式成功生成SQL")
                
                # 如果探索模式成功，可以考虑将这个成功经验添加到知识库
                if self.accumulate_knowledge:
                    self._accumulate_successful_experience(question, sql, details)
                
                return sql, success, details
            else:
                self.stats['total_failures'] += 1
                execution_time = time.time() - start_time
                details['execution_time'] = execution_time
                details['mode_used'] = 'exploration'
                details['fallback_to_exploration'] = True
                details['lpe_experience_error'] = details.get('lpe_experience_error')
                self.logger.error("✗ 探索模式也失败了")
                return sql, success, details
        else:
            # 如果不启用探索模式回退，直接返回经验模式的结果
            self.stats['total_failures'] += 1
            execution_time = time.time() - start_time
            details['execution_time'] = execution_time
            details['mode_used'] = 'lpe_experience'
            details['fallback_to_exploration'] = False
            self.logger.error("✗ LPE经验模式失败，且未启用探索模式回退")
            return sql, success, details
    
    def _try_lpe_experience_mode(self, 
                               question: str, 
                               db_path: str,
                               db_schema: Dict[str, Any] = None) -> Tuple[str, bool, Dict[str, Any]]:
        """
        尝试使用LPE经验模式生成SQL
        
        Args:
            question: 自然语言问题
            db_path: 数据库路径
            db_schema: 数据库schema信息
            
        Returns:
            (SQL, 是否成功, 详细信息)
        """
        try:
            # 获取详细的schema信息
            if db_schema is None:
                db_schema = self.enhanced_sql_generator.get_detailed_schema(db_path)
            
            # 使用LPE SQL生成器生成SQL
            sql = self.enhanced_sql_generator.generate_sql_with_schema(
                question, db_schema, db_path, max_retries=self.max_retries_experience_mode
            )
            
            # 验证SQL语法
            validation = self.sql_postprocessor.validate_sql_syntax(sql, db_path)
            
            if validation['is_valid']:
                return sql, True, {
                    'raw_sql': sql,
                    'validation': validation,
                    'retries_used': 0,
                    'lpe_experience_success': True
                }
            else:
                # 如果验证失败，记录错误信息
                return sql, False, {
                    'error': f"LPE经验模式生成SQL验证失败: {validation['error']}",
                    'validation': validation,
                    'lpe_experience_error': validation['error']
                }
        
        except Exception as e:
            self.logger.error(f"LPE经验模式生成SQL时发生错误: {e}")
            return "", False, {'error': str(e), 'lpe_experience_error': str(e)}
    
    def _try_exploration_mode(self, 
                             question: str, 
                             db_path: str,
                             db_schema: Dict[str, Any] = None) -> Tuple[str, bool, Dict[str, Any]]:
        """
        尝试使用探索模式生成SQL（与之前相同）
        
        Args:
            question: 自然语言问题
            db_path: 数据库路径
            db_schema: 数据库schema信息
            
        Returns:
            (SQL, 是否成功, 详细信息)
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
    
    def _accumulate_successful_experience(self, question: str, sql: str, details: Dict[str, Any]):
        """
        积累成功经验到知识库
        
        Args:
            question: 问题文本
            sql: 成功的SQL查询
            details: 详细信息
        """
        try:
            # 提取有用的信息用于知识积累
            knowledge = details.get('knowledge', '')
            thought_process = details.get('thought_process', '')
            difficulty = details.get('difficulty', 'unknown')
            
            # 添加到知识库
            self.enhanced_sql_generator.add_experience(
                question=question,
                sql=sql,
                correct=True,
                knowledge=knowledge,
                thought_process=thought_process,
                difficulty=difficulty
            )
            
            self.stats['knowledge_accumulated'] += 1
            self.logger.info("✓ 成功经验已积累到知识库")
            
        except Exception as e:
            self.logger.warning(f"积累成功经验失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取生成器统计信息"""
        return {
            **self.stats,
            'experience_generator_stats': self.enhanced_sql_generator.get_statistics(),
            'exploration_engine_stats': self.exploration_engine.exploration_stats if hasattr(self.exploration_engine, 'exploration_stats') else {}
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("\n=== LPE混合SQL生成器统计信息 ===")
        print(f"总问题数: {stats['total_questions']}")
        print(f"LPE经验模式成功: {stats['experience_mode_success']}")
        print(f"探索模式使用次数: {stats['exploration_mode_used']}")
        print(f"探索模式成功: {stats['exploration_mode_success']}")
        print(f"总失败数: {stats['total_failures']}")
        print(f"知识积累次数: {stats['knowledge_accumulated']}")
        
        if stats['total_questions'] > 0:
            lpe_success_rate = stats['experience_mode_success'] / stats['total_questions']
            exploration_usage_rate = stats['exploration_mode_used'] / stats['total_questions']
            print(f"LPE经验模式成功率: {lpe_success_rate:.2%}")
            print(f"探索模式使用率: {exploration_usage_rate:.2%}")
        
        # 打印子组件统计信息
        self.enhanced_sql_generator.print_statistics()
        print("=" * 40)


# 测试函数
def test_hybrid_lpe_generator():
    """测试LPE混合SQL生成器"""
    print("=== 测试LPE混合SQL生成器 ===")
    
    try:
        # 初始化生成器
        generator = HybridSQLGeneratorLPE()
        
        # 测试问题
        test_questions = [
            "Find the employee with the highest salary",
            "List all customers who have placed more than 5 orders",
            "First, find the department with the highest average salary, then list all employees in that department"
        ]
        
        for question in test_questions:
            print(f"\n测试问题: {question}")
            
            final_sql, success, details = generator.generate_sql(question, "database.db")
            
            if success:
                print(f"✓ 成功生成SQL: {final_sql[:100]}...")
                print(f"  使用模式: {details['mode_used']}")
            else:
                print(f"✗ 生成失败: {details.get('error', '未知错误')}")
        
        # 打印统计信息
        generator.print_statistics()
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_hybrid_lpe_generator()