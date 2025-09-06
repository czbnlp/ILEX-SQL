#!/usr/bin/env python3
"""
BIRD数据集评估器
用于评估ILEX-SQL系统在BIRD数据集上的性能
"""

import json
import sqlite3
import time
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import pandas as pd

# 添加项目路径
sys.path.append('src')

from llm_connector_local import LocalLLMConnector
from sql_executor import SQLExecutor
from ilex_core.mode_selector import ModeSelector
from ilex_core.exploration_engine import ExplorationEngine


class BIRDEvaluator:
    """BIRD数据集评估器"""
    
    def __init__(self, 
                 data_dir: str = "LPE-SQL/data",
                 db_root: str = "LPE-SQL/data",
                 use_local_model: bool = True,
                 timeout: int = 30):
        """
        初始化评估器
        
        Args:
            data_dir: BIRD数据集目录
            db_root: 数据库根目录
            use_local_model: 是否使用本地模型
            timeout: SQL执行超时时间
        """
        self.data_dir = Path(data_dir)
        self.db_root = Path(db_root)
        self.use_local_model = use_local_model
        self.timeout = timeout
        
        # 初始化组件
        if use_local_model:
            self.llm_connector = LocalLLMConnector()
        else:
            from llm_connector import LLMConnector
            self.llm_connector = LLMConnector()
        
        self.sql_executor = SQLExecutor()
        self.mode_selector = ModeSelector()
        self.exploration_engine = ExplorationEngine(
            llm_connector=self.llm_connector,
            sql_executor=self.sql_executor
        )
        
        # 统计信息
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
    
    def load_bird_data(self, split: str = "dev") -> List[Dict]:
        """
        加载BIRD数据集
        
        Args:
            split: 数据集分割 (dev, test, train)
            
        Returns:
            数据集列表
        """
        json_file = self.data_dir / f"{split}.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"BIRD数据集文件不存在: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"加载了 {len(data)} 个BIRD数据集问题")
        return data
    
    def load_gold_sql(self, split: str = "dev") -> Dict[str, str]:
        """
        加载标准SQL答案
        
        Args:
            split: 数据集分割
            
        Returns:
            问题ID到SQL的映射
        """
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
        """
        获取数据库文件路径
        
        Args:
            db_id: 数据库ID
            
        Returns:
            数据库文件路径
        """
        db_path = self.db_root / db_id / f"{db_id}.sqlite"
        
        if not db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
        
        return str(db_path)
    
    def generate_sql(self, question: str, db_id: str) -> Tuple[str, bool, Dict]:
        """
        生成SQL查询
        
        Args:
            question: 自然语言问题
            db_id: 数据库ID
            
        Returns:
            (生成的SQL, 是否成功, 详细信息)
        """
        try:
            # 获取数据库schema
            db_path = self.get_db_path(db_id)
            schema = self._get_db_schema(db_path)
            
            # 评估问题复杂度
            mode_decision = self.mode_selector.get_mode_decision(question)
            
            # 根据模式选择处理方式
            if mode_decision['use_exploration_mode']:
                # 使用探索模式
                final_sql, success, details = self.exploration_engine.solve_complex_question(
                    question,
                    db_path,
                    schema
                )
            else:
                # 使用经验模式
                prompt = f"""
                基于以下数据库schema，为问题生成SQL查询：
                
                问题: {question}
                
                数据库Schema:
                {schema}
                
                请只返回SQL语句，不要包含其他解释。
                """
                
                final_sql = self.llm_connector(prompt)
                success = True
                details = {"mode": "experience"}
            
            return final_sql, success, details
            
        except Exception as e:
            return "", False, {"error": str(e)}
    
    def _get_db_schema(self, db_path: str) -> str:
        """
        获取数据库schema信息
        
        Args:
            db_path: 数据库路径
            
        Returns:
            Schema信息字符串
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 获取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_info = []
            for table in tables:
                table_name = table[0]
                schema_info.append(f"表: {table_name}")
                
                # 获取表结构
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                for col in columns:
                    col_name, col_type, not_null, default_val, is_pk = col
                    constraints = []
                    if not_null:
                        constraints.append("NOT NULL")
                    if is_pk:
                        constraints.append("PRIMARY KEY")
                    
                    constraint_str = f" {', '.join(constraints)}" if constraints else ""
                    schema_info.append(f"  {col_name} {col_type}{constraint_str}")
                
                schema_info.append("")
            
            conn.close()
            return "\n".join(schema_info)
            
        except Exception as e:
            return f"获取schema失败: {e}"
    
    def execute_sql_with_timeout(self, sql: str, db_path: str) -> Tuple[Any, Optional[str]]:
        """
        带超时的SQL执行
        
        Args:
            sql: SQL查询
            db_path: 数据库路径
            
        Returns:
            (查询结果, 错误信息)
        """
        try:
            # 使用sql_executor执行
            result, error = self.sql_executor(sql, db_path)
            
            if error:
                return None, error
            
            return result, None
            
        except Exception as e:
            return None, str(e)
    
    def calculate_accuracy(self, predicted_result: List, gold_result: List) -> bool:
        """
        计算SQL执行结果的准确率
        
        Args:
            predicted_result: 预测结果
            gold_result: 标准结果
            
        Returns:
            是否正确
        """
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
        评估单个问题
        
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
            'mode': 'unknown'
        }
        
        try:
            start_time = time.time()
            
            # 生成SQL
            predicted_sql, success, details = self.generate_sql(question, db_id)
            result['predicted_sql'] = predicted_sql
            result['mode'] = details.get('mode', 'unknown')
            
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
    
    def evaluate_dataset(self, 
                        split: str = "dev",
                        max_questions: Optional[int] = None,
                        output_file: Optional[str] = None) -> Dict:
        """
        评估整个数据集
        
        Args:
            split: 数据集分割
            max_questions: 最大问题数量（用于测试）
            output_file: 输出文件路径
            
        Returns:
            评估结果
        """
        print(f"开始评估BIRD数据集 ({split})...")
        
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
            if result['correct']:
                self.stats['correct'] += 1
                self.stats['by_difficulty'][result['difficulty']]['correct'] += 1
            
            self.stats['by_difficulty'][result['difficulty']]['total'] += 1
            
            if result['timeout']:
                self.stats['timeout'] += 1
            
            if result['error']:
                self.stats['error'] += 1
        
        self.results = results
        
        # 保存结果
        if output_file:
            self.save_results(output_file)
        
        # 打印统计信息
        self.print_statistics()
        
        return self.stats
    
    def save_results(self, output_file: str):
        """
        保存评估结果
        
        Args:
            output_file: 输出文件路径
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'statistics': self.stats,
                'results': self.results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {output_file}")
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("BIRD数据集评估结果")
        print("="*60)
        
        # 总体统计
        total = self.stats['total']
        correct = self.stats['correct']
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"总问题数: {total}")
        print(f"正确数: {correct}")
        print(f"准确率: {accuracy:.2f}%")
        print(f"超时数: {self.stats['timeout']}")
        print(f"错误数: {self.stats['error']}")
        
        # 按难度统计
        print("\n按难度分类统计:")
        print("{:15} {:>8} {:>8} {:>10}".format("难度", "总数", "正确", "准确率"))
        print("-" * 45)
        
        for difficulty in ['simple', 'moderate', 'challenging']:
            stats = self.stats['by_difficulty'][difficulty]
            diff_total = stats['total']
            diff_correct = stats['correct']
            diff_accuracy = (diff_correct / diff_total * 100) if diff_total > 0 else 0
            
            print("{:15} {:>8} {:>8} {:>9.1f}%".format(
                difficulty, diff_total, diff_correct, diff_accuracy
            ))
        
        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BIRD数据集评估器")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test", "train"],
                       help="数据集分割")
    parser.add_argument("--max-questions", type=int, default=None,
                       help="最大评估问题数量（用于测试）")
    parser.add_argument("--output", type=str, default="bird_evaluation_results.json",
                       help="输出文件路径")
    parser.add_argument("--use-openai", action="store_true",
                       help="使用OpenAI API而不是本地模型")
    parser.add_argument("--timeout", type=int, default=30,
                       help="SQL执行超时时间（秒）")
    
    args = parser.parse_args()
    
    try:
        # 创建评估器
        evaluator = BIRDEvaluator(
            use_local_model=not args.use_openai,
            timeout=args.timeout
        )
        
        # 执行评估
        stats = evaluator.evaluate_dataset(
            split=args.split,
            max_questions=args.max_questions,
            output_file=args.output
        )
        
        print(f"\n评估完成！结果已保存到: {args.output}")
        
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()