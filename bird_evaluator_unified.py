#!/usr/bin/env python3
"""
统一的BIRD数据集评估器
整合所有功能，支持混合模式（先经验模式，失败后探索模式）
支持并发执行和多种运行模式
"""

import json
import sqlite3
import time
import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import threading
from queue import Queue

# 添加项目路径
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from llm_connector_local import LocalLLMConnector
from llm_connector_api import APILLMConnector
from llm_connector_unified import UnifiedLLMConnector
from sql_executor import SQLExecutor
from src.ilex_core.sql_generator_hybrid import HybridSQLGenerator
from enhanced_sql_generator import EnhancedSQLGeneratorLPE
from master_sql_postprocessor import MasterSQLPostProcessor  # pyright: ignore[reportUnusedImport]


class UnifiedBIRDEvaluator:
    """统一的BIRD数据集评估器"""
    
    def __init__(self, 
                 data_dir: str = "data/",
                 db_root: str = "data/dev_databases",
                 max_concurrency: int = 1,
                 use_local_model: bool = True,
                 use_mock: bool = False,
                 timeout: int = 30,
                 use_api_model: bool = False):
        """
        初始化评估器
        
        Args:
            data_dir: BIRD数据集目录
            db_root: 数据库根目录
            max_concurrency: 最大并发度
            use_local_model: 是否使用本地模型
            use_mock: 是否使用模拟模式
            timeout: SQL执行超时时间
            use_api_model: 是否使用API模型（如百度千帆）
        """
        self.data_dir = Path(data_dir)
        self.db_root = Path(db_root)
        self.max_concurrency = max_concurrency
        self.use_local_model = use_local_model
        self.use_mock = use_mock
        self.timeout = timeout
        self.use_api_model = use_api_model
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        if self.use_mock:
            self.llm_connector = MockLLMConnector()
            self.sql_executor = MockSQLExecutor()
        else:
            if use_api_model:
                # 使用API模型（如百度千帆）
                self.llm_connector = UnifiedLLMConnector()
                self.logger.info("使用API模型连接器")
            elif use_local_model:
                # 使用本地vLLM模型
                self.llm_connector = LocalLLMConnector()
                self.logger.info("使用本地vLLM模型连接器")
            else:
                # 使用外部API模型（如OpenAI）
                from llm_connector import LLMConnector
                self.llm_connector = LLMConnector()
                self.logger.info("使用外部API模型连接器")
            
            self.sql_executor = SQLExecutor()
        
        # 初始化混合SQL生成器
        self.sql_generator = HybridSQLGenerator(
            llm_connector=self.llm_connector,
            sql_executor=self.sql_executor
        )
        
        # 线程安全的统计信息
        self.stats_lock = threading.Lock()
        self.stats = {
            'total': 0,
            'correct': 0,
            'timeout': 0,
            'error': 0,
            'by_difficulty': {
                'simple': {'total': 0, 'correct': 0},
                'moderate': {'total': 0, 'correct': 0},
                'challenging': {'total': 0, 'correct': 0}
            }
        }
        
        # 结果记录
        self.results = []
        self.results_lock = threading.Lock()
        
        # 创建数据库连接池（用于并发模式）
        self.db_connections = {}
        self.db_locks = {}
    
    def load_bird_data(self, split: str = "dev") -> List[Dict]:
        """加载BIRD数据集"""
        json_file = self.data_dir / f"{split}.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"BIRD数据集文件不存在: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"加载了 {len(data)} 个BIRD数据集问题")
        return data
    
    def load_gold_sql(self, split: str = "dev") -> Dict[str, str]:
        """加载标准SQL答案"""
        sql_file = self.data_dir / f"{split}.sql"
        
        if not sql_file.exists():
            raise FileNotFoundError(f"标准SQL文件不存在: {sql_file}")
        
        gold_sqls = {}
        with open(sql_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if '\t' in line:
                sql, db_id = line.strip().split('\t')
                gold_sqls[str(i)] = sql.strip()
        
        return gold_sqls
    
    def get_db_path(self, db_id: str) -> str:
        """获取数据库文件路径"""
        db_path = self.db_root / db_id / f"{db_id}.sqlite"
        
        if not db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
        
        return str(db_path)
    
    def get_db_connection(self, db_path: str) -> sqlite3.Connection:
        """获取数据库连接（线程安全）"""
        if db_path not in self.db_connections:
            with self.stats_lock:
                if db_path not in self.db_connections:
                    self.db_connections[db_path] = sqlite3.connect(db_path, check_same_thread=False)
                    self.db_locks[db_path] = threading.Lock()
        
        return self.db_connections[db_path]
    
    def generate_sql_hybrid(self, question: str, db_id: str) -> Tuple[str, bool, Dict]:
        """
        使用混合模式生成SQL（先经验模式，失败后探索模式）
        
        Args:
            question: 自然语言问题
            db_id: 数据库ID
            
        Returns:
            (生成的SQL, 是否成功, 详细信息)
        """
        try:
            # 获取数据库路径
            db_path = self.get_db_path(db_id)
            
            # 使用混合SQL生成器
            final_sql, success, details = self.sql_generator.generate_sql(question, db_path)
            
            return final_sql, success, details
            
        except Exception as e:
            return "", False, {"error": str(e), "mode": "error"}
    
    def execute_sql_with_timeout(self, sql: str, db_path: str) -> Tuple[Any, Optional[str]]:
        """带超时的SQL执行（线程安全）"""
        try:
            if self.use_mock:
                # 模拟模式直接使用sql_executor
                return self.sql_executor(sql, db_path)
            else:
                if self.max_concurrency > 1:
                    # 并发模式使用连接池
                    conn = self.get_db_connection(db_path)
                    lock = self.db_locks[db_path]
                    
                    with lock:
                        cursor = conn.cursor()
                        cursor.execute(f"EXPLAIN {sql}")
                        
                        # 获取实际结果
                        cursor.execute(sql)
                        result = cursor.fetchall()
                        
                        return result, None
                else:
                    # 单线程模式直接使用sql_executor
                    return self.sql_executor(sql, db_path)
            
        except sqlite3.Error as e:
            if "timeout" in str(e).lower():
                return None, "timeout"
            return None, str(e)
        except Exception as e:
            return None, str(e)
    
    def calculate_accuracy(self, predicted_result: List, gold_result: List) -> bool:
        """计算SQL执行结果的准确率"""
        try:
            # 将结果转换为集合进行比较
            pred_set = set()
            gold_set = set()
            
            for row in predicted_result:
                if isinstance(row, tuple):
                    pred_set.add(row)
                elif isinstance(row, list):
                    pred_set.add(tuple(row))
                else:
                    pred_set.add((row,))
            
            for row in gold_result:
                if isinstance(row, tuple):
                    gold_set.add(row)
                elif isinstance(row, list):
                    gold_set.add(tuple(row))
                else:
                    gold_set.add((row,))
            
            return pred_set == gold_set
            
        except Exception as e:
            print(f"结果比较错误: {e}")
            return False
    
    def evaluate_single_question(self, 
                                question_data: Dict, 
                                gold_sql: str, 
                                idx: int) -> Dict:
        """
        评估单个问题（线程安全版本）
        
        Args:
            question_data: 问题数据
            gold_sql: 标准SQL
            idx: 问题索引
            
        Returns:
            评估结果
        """
        question = question_data['question']
        db_id = question_data['db_id']
        difficulty = question_data.get('difficulty', 'simple')
        question_id = question_data.get('question_id', idx)
        
        result = {
            'question_id': question_id,
            'question': question,
            'db_id': db_id,
            'difficulty': difficulty,
            'gold_sql': gold_sql,
            'predicted_sql': '',
            'correct': False,
            'error': None,
            'timeout': False,
            'execution_time': 0,
            'mode': 'hybrid'
        }
        
        try:
            start_time = time.time()
            
            # 生成SQL
            predicted_sql, success, details = self.generate_sql_hybrid(question, db_id)
            result['predicted_sql'] = predicted_sql
            result['mode'] = details.get('mode_used', 'unknown')
            result['details'] = details
            
            if not success:
                result['error'] = details.get('error', 'SQL生成失败')
                return result
            
            # 执行标准SQL
            db_path = self.get_db_path(db_id)
            gold_result, gold_error = self.execute_sql_with_timeout(gold_sql, db_path)
            
            if gold_error:
                result['error'] = f"标准SQL执行失败: {gold_error}"
                return result
            
            # 执行预测SQL
            predicted_result, pred_error = self.execute_sql_with_timeout(predicted_sql, db_path)
            
            if pred_error:
                if 'timeout' in pred_error.lower():
                    result['timeout'] = True
                    result['error'] = '执行超时'
                else:
                    result['error'] = f"预测SQL执行失败: {pred_error}"
                return result
            
            # 比较结果
            result['correct'] = self.calculate_accuracy(predicted_result, gold_result)
            result['execution_time'] = time.time() - start_time
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def update_stats(self, result: Dict):
        """更新统计信息（线程安全）"""
        with self.stats_lock:
            self.stats['total'] += 1
            
            if result['correct']:
                self.stats['correct'] += 1
                self.stats['by_difficulty'][result['difficulty']]['correct'] += 1
            
            self.stats['by_difficulty'][result['difficulty']]['total'] += 1
            
            if result['timeout']:
                self.stats['timeout'] += 1
            
            if result['error']:
                self.stats['error'] += 1
    
    def add_result(self, result: Dict):
        """添加结果（线程安全）"""
        with self.results_lock:
            self.results.append(result)
    
    def evaluate_dataset(self, 
                        split: str = "dev",
                        max_questions: Optional[int] = None,
                        output_file: Optional[str] = None) -> Dict:
        """
        评估整个数据集
        
        Args:
            split: 数据集分割
            max_questions: 最大问题数量
            output_file: 输出文件路径
            
        Returns:
            评估结果
        """
        if self.max_concurrency > 1:
            return self.evaluate_dataset_concurrent(split, max_questions, output_file)
        else:
            return self.evaluate_dataset_sequential(split, max_questions, output_file)
    
    def evaluate_dataset_sequential(self, 
                                   split: str = "dev",
                                   max_questions: Optional[int] = None,
                                   output_file: Optional[str] = None) -> Dict:
        """顺序评估整个数据集"""
        print(f"开始顺序评估BIRD数据集 ({split})...")
        
        # 加载数据
        bird_data = self.load_bird_data(split)
        gold_sqls = self.load_gold_sql(split)
        
        # 限制问题数量
        if max_questions:
            bird_data = bird_data[:max_questions]
            print(f"限制评估问题数量: {max_questions}")
        
        self.stats['total'] = len(bird_data)
        
        # 评估每个问题
        results = []
        for i, question_data in enumerate(tqdm(bird_data, desc="评估进度")):
            idx = question_data.get('question_id', i)
            gold_sql = gold_sqls.get(str(idx), '')
            
            if not gold_sql:
                print(f"警告: 问题 {idx} 没有找到标准SQL")
                continue
            
            result = self.evaluate_single_question(question_data, gold_sql, idx)
            results.append(result)
            
            # 更新统计信息
            self.update_stats(result)
        
        self.results = results
        
        # 保存结果
        if output_file:
            self.save_results(output_file)
        
        # 打印统计信息
        self.print_statistics()
        
        # 打印混合模式统计信息
        if hasattr(self.sql_generator, 'print_statistics'):
            self.sql_generator.print_statistics()
        
        return self.stats
    
    def evaluate_dataset_concurrent(self, 
                                  split: str = "dev",
                                  max_questions: Optional[int] = None,
                                  output_file: Optional[str] = None) -> Dict:
        """并发评估整个数据集"""
        print(f"开始并发评估BIRD数据集 ({split})，并发度: {self.max_concurrency}...")
        
        # 加载数据
        bird_data = self.load_bird_data(split)
        gold_sqls = self.load_gold_sql(split)
        
        # 限制问题数量
        if max_questions:
            bird_data = bird_data[:max_questions]
            print(f"限制评估问题数量: {max_questions}")
        
        # 创建任务队列
        tasks = []
        for i, question_data in enumerate(bird_data):
            idx = question_data.get('question_id', i)
            gold_sql = gold_sqls.get(str(idx), '')
            
            if not gold_sql:
                print(f"警告: 问题 {idx} 没有找到标准SQL")
                continue
            
            tasks.append((question_data, gold_sql, idx))
        
        # 并发执行
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self.evaluate_single_question, question_data, gold_sql, idx): (question_data, gold_sql, idx)
                for question_data, gold_sql, idx in tasks
            }
            
            # 收集结果
            with tqdm(total=len(tasks), desc="并发评估进度") as pbar:
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        self.add_result(result)
                        self.update_stats(result)
                        pbar.update(1)
                    except Exception as e:
                        print(f"评估任务失败: {e}")
                        pbar.update(1)
        
        # 保存结果
        if output_file:
            self.save_results(output_file)
        
        # 打印统计信息
        self.print_statistics()
        
        # 打印混合模式统计信息
        if hasattr(self.sql_generator, 'print_statistics'):
            self.sql_generator.print_statistics()
        
        return self.stats
    
    def save_results(self, output_file: str):
        """保存评估结果"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'results': self.results,
                'statistics': self.stats,
                'config': {
                    'max_concurrency': self.max_concurrency,
                    'use_local_model': self.use_local_model,
                    'use_mock': self.use_mock,
                    'timeout': self.timeout
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存到: {output_path}")
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.stats
        
        print("\n=== BIRD数据集评估统计信息 ===")
        print(f"总问题数: {stats['total']}")
        print(f"正确数: {stats['correct']}")
        print(f"超时数: {stats['timeout']}")
        print(f"错误数: {stats['error']}")
        
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            print(f"准确率: {accuracy:.2%}")
        
        print("\n按难度分类:")
        for difficulty, diff_stats in stats['by_difficulty'].items():
            total = diff_stats['total']
            correct = diff_stats['correct']
            if total > 0:
                acc = correct / total
                print(f"  {difficulty}: {correct}/{total} ({acc:.2%})")
        
        print("=" * 40)
    
    def close(self):
        """关闭资源"""
        # 关闭数据库连接
        for conn in self.db_connections.values():
            try:
                conn.close()
            except:
                pass


# 模拟组件
class MockLLMConnector:
    """模拟LLM连接器"""
    
    def __init__(self):
        self.model = "mock-model"
    
    def __call__(self, prompt: str) -> str:
        """模拟LLM响应"""
        # 简单的SQL生成逻辑
        if "highest" in prompt.lower() and "free rate" in prompt.lower():
            return "SELECT MAX(`Free Meal Count (K-12)` / `Enrollment (K-12)`) FROM frpm WHERE `County Name` = 'Alameda'"
        elif "lowest" in prompt.lower() and "free rate" in prompt.lower():
            return "SELECT MIN(`Free Meal Count (K-12)` / `Enrollment (K-12)`) FROM frpm WHERE `Educational Option Type` = 'Continuation School'"
        elif "zip code" in prompt.lower() and "charter schools" in prompt.lower():
            return "SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`District Name` = 'Fresno County Office of Education' AND T1.`Charter School (Y/N)` = 1"
        else:
            # 默认返回一个简单的SELECT查询
            return "SELECT * FROM frpm LIMIT 10"
    
    def test_connection(self) -> bool:
        """模拟连接测试"""
        return True


class MockSQLExecutor:
    """模拟SQL执行器"""
    
    def __call__(self, sql: str, db_path: str = None) -> Tuple[Any, Optional[str]]:
        """模拟SQL执行"""
        try:
            # 真实执行SQL
            if db_path:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(sql)
                result = cursor.fetchall()
                conn.close()
                return result, None
            else:
                return [], None
        except Exception as e:
            return None, str(e)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="统一的BIRD数据集评估器")
    parser.add_argument("--split", default="dev", help="数据集分割 (dev/test/train)")
    parser.add_argument("--max-questions", type=int, help="最大问题数量")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--concurrency", type=int, default=1, help="并发度")
    parser.add_argument("--mock", action="store_true", help="使用模拟模式")
    parser.add_argument("--remote-model", action="store_true", help="使用远程模型")
    parser.add_argument("--api-model", action="store_true", help="使用API模型（如百度千帆）")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = UnifiedBIRDEvaluator(
        max_concurrency=args.concurrency,
        use_local_model=not args.remote_model and not args.api_model,
        use_mock=args.mock,
        use_api_model=args.api_model
    )
    
    try:
        # 执行评估
        stats = evaluator.evaluate_dataset(
            split=args.split,
            max_questions=args.max_questions,
            output_file=args.output
        )
        
        print(f"\n✓ 评估完成！")
        
    except Exception as e:
        print(f"❌ 评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()