#!/usr/bin/env python3
"""
增强的SQL生成器 - LPE-SQL 风格实现
基于经验知识库和检索的few-shot示例生成SQL查询
"""

import re
import json
import sqlite3
import logging
from typing import Dict, List, Tuple, Any, Optional
from llm_connector_local import LocalLLMConnector
from master_sql_postprocessor import MasterSQLPostProcessor
from src.ilex_core.experience_retriever import ExperienceRetriever


class EnhancedSQLGeneratorLPE:
    """基于LPE-SQL方法的增强SQL生成器"""
    
    def __init__(self, llm_connector=None, experience_retriever=None):
        """
        初始化SQL生成器
        
        Args:
            llm_connector: LLM连接器
            experience_retriever: 经验检索器，如果为None则创建默认实例
        """
        self.llm_connector = llm_connector or LocalLLMConnector()
        self.experience_retriever = experience_retriever or ExperienceRetriever()
        self.sql_postprocessor = MasterSQLPostProcessor()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 统计信息
        self.stats = {
            'total_questions': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'experience_mode_used': 0,
            'retrieval_success_count': 0
        }
    
    def get_detailed_schema(self, db_path: str) -> Dict[str, Any]:
        """
        获取详细的数据库schema信息（与之前相同）
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            schema_info = {
                'tables': {},
                'relationships': [],
                'sample_data': {}
            }
            
            # 获取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                
                # 获取表结构
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                table_info = {
                    'name': table_name,
                    'columns': [],
                    'primary_keys': [],
                    'foreign_keys': []
                }
                
                for col in columns:
                    if len(col) >= 6:
                        col_name, col_type, not_null, default_val, is_pk = col[1], col[2], bool(col[3]), col[4], bool(col[5])
                    elif len(col) >= 5:
                        col_name, col_type, not_null, default_val, is_pk = col[1], col[2], bool(col[3]), col[4], False
                    else:
                        col_name, col_type, not_null, default_val, is_pk = col + (None,) * (5 - len(col))
                    
                    column_info = {
                        'name': col_name,
                        'type': col_type,
                        'not_null': bool(not_null),
                        'primary_key': bool(is_pk),
                        'default_value': default_val
                    }
                    table_info['columns'].append(column_info)
                    
                    if is_pk:
                        table_info['primary_keys'].append(col_name)
                
                # 获取外键信息
                cursor.execute(f"PRAGMA foreign_key_list({table_name});")
                foreign_keys = cursor.fetchall()
                
                for fk in foreign_keys:
                    if len(fk) >= 5:
                        fk_info = {
                            'from_table': table_name,
                            'from_column': fk[3],
                            'to_table': fk[2],
                            'to_column': fk[4]
                        }
                        table_info['foreign_keys'].append(fk_info)
                        schema_info['relationships'].append(fk_info)
                
                # 获取示例数据
                try:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                    sample_data = cursor.fetchall()
                    if sample_data:
                        schema_info['sample_data'][table_name] = sample_data
                except:
                    pass
                
                schema_info['tables'][table_name] = table_info
            
            conn.close()
            return schema_info
            
        except Exception as e:
            self.logger.error(f"获取详细schema失败: {e}")
            return {'tables': {}, 'relationships': [], 'sample_data': {}}
    
    def generate_sql_with_schema(self, question: str, schema: Dict[str, Any], db_path: str, 
                               max_retries: int = 2, error_history: list = None, 
                               accumulate_knowledge: bool = True) -> str:
        """
        基于schema生成SQL，使用LPE-SQL方法的经验检索
        
        Args:
            question: 问题文本
            schema: 数据库schema
            db_path: 数据库路径
            max_retries: 最大重试次数
            error_history: 错误历史记录
            accumulate_knowledge: 是否积累知识库
            
        Returns:
            生成的SQL语句
        """
        self.stats['total_questions'] += 1
        
        if error_history is None:
            error_history = []
        
        # 步骤1: 从经验知识库检索相似示例
        self.logger.info(f"正在检索相似示例 for: {question[:50]}...")
        correct_examples, mistake_examples = self.experience_retriever.retrieve_similar_examples(question)
        
        if correct_examples or mistake_examples:
            self.stats['retrieval_success_count'] += 1
            self.logger.info(f"✓ 检索到 {len(correct_examples)} 个正确示例, {len(mistake_examples)} 个错误示例")
        else:
            self.logger.warning("⚠ 未检索到相似示例，将使用基础方法")
        
        # 步骤2: 构建schema文本
        relevant_tables = self._find_relevant_tables(question, schema)
        schema_text = self._build_schema_text(schema, relevant_tables)
        
        # 步骤3: 构建LPE-SQL风格的提示词
        prompt = self._build_lpe_prompt(
            question, schema_text, correct_examples, mistake_examples, 
            db_path, error_history, accumulate_knowledge
        )
        
        # 步骤4: 调用LLM生成SQL
        self.logger.info("正在生成SQL...")
        response = self.llm_connector(prompt)
        
        # 步骤5: 提取SQL
        sql = self._extract_sql_from_response(response)
        
        # 步骤6: 后处理SQL
        processed_sql = self.sql_postprocessor.fix_sql_syntax(sql, schema)
        
        # 步骤7: 验证SQL
        validation = self.sql_postprocessor.validate_sql_syntax(processed_sql, db_path)
        
        if validation['is_valid']:
            self.stats['successful_generations'] += 1
            self.logger.info("✓ SQL生成成功")
            return processed_sql
        else:
            # 如果验证失败，尝试重试
            if max_retries > 0:
                self.logger.warning(f"SQL验证失败: {validation['error']}，正在重试...")
                error_history.append({
                    'sql': processed_sql,
                    'error': validation['error'],
                    'suggested_fix': validation.get('suggested_fix', '')
                })
                return self.generate_sql_with_schema(question, schema, db_path, max_retries - 1, error_history, accumulate_knowledge)
            else:
                self.stats['failed_generations'] += 1
                self.logger.error("SQL生成失败，已达到最大重试次数")
                # 返回经过后处理的SQL，即使验证失败
                return processed_sql
    
    def _find_relevant_tables(self, question: str, schema: Dict[str, Any]) -> List[str]:
        """找到相关的表（与之前相同）"""
        question_lower = question.lower()
        relevant_tables = []
        
        # 基于关键词匹配表名
        for table_name in schema['tables'].keys():
            table_lower = table_name.lower()
            
            # 检查表名是否与问题中的关键词匹配
            if any(keyword in table_lower for keyword in question_lower.split()):
                relevant_tables.append(table_name)
                continue
            
            # 检查列名是否与问题中的关键词匹配
            columns = schema['tables'][table_name]['columns']
            for col in columns:
                col_lower = col['name'].lower()
                if any(keyword in col_lower for keyword in question_lower.split()):
                    relevant_tables.append(table_name)
                    break
        
        # 如果没有找到相关表，返回所有表
        if not relevant_tables:
            relevant_tables = list(schema['tables'].keys())
        
        return relevant_tables
    
    def _build_schema_text(self, schema: Dict[str, Any], relevant_tables: List[str]) -> str:
        """构建schema文本（与之前相同）"""
        schema_parts = []
        schema_parts.append("### Schema of the database with sample rows:")
        schema_parts.append("=" * 60)
        
        for table_name in relevant_tables:
            if table_name not in schema['tables']:
                continue
                
            table_info = schema['tables'][table_name]
            
            # 构建CREATE TABLE语句
            columns_def = []
            for col in table_info['columns']:
                col_def = f"{col['name']} {col['type']}"
                if col['primary_key']:
                    col_def += " PRIMARY KEY"
                if col['not_null']:
                    col_def += " NOT NULL"
                columns_def.append(col_def)
            
            create_table = f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(columns_def) + "\n);"
            schema_parts.append(create_table)
            
            # 添加示例数据
            if table_name in schema['sample_data']:
                sample_rows = schema['sample_data'][table_name]
                if sample_rows:
                    column_names = [col['name'] for col in table_info['columns']]
                    
                    schema_parts.append(f"\n/*")
                    schema_parts.append(f" {len(sample_rows)} example rows:")
                    schema_parts.append(f" SELECT * FROM {table_name} LIMIT {len(sample_rows)};")
                    
                    # 格式化显示示例数据
                    col_widths = [max(len(str(col)), max(len(str(row[i])) for row in sample_rows)) for i, col in enumerate(column_names)]
                    
                    header = " | ".join(col.ljust(width) for col, width in zip(column_names, col_widths))
                    schema_parts.append(f" {header}")
                    
                    separator = " | ".join("-" * width for width in col_widths)
                    schema_parts.append(f" {separator}")
                    
                    for row in sample_rows:
                        data_row = " | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths))
                        schema_parts.append(f" {data_row}")
                    
                    schema_parts.append(f" */")
            
            schema_parts.append("")  # 空行分隔
        
        return "\n".join(schema_parts)
    
    def _build_lpe_prompt(self, question: str, schema_text: str, 
                         correct_examples: List[Dict], mistake_examples: List[Dict],
                         db_path: str, error_history: List[Dict], 
                         accumulate_knowledge: bool) -> str:
        """
        构建LPE-SQL风格的提示词
        
        Args:
            question: 问题文本
            schema_text: schema文本
            correct_examples: 正确示例列表
            mistake_examples: 错误示例列表
            db_path: 数据库路径
            error_history: 错误历史
            accumulate_knowledge: 是否积累知识
            
        Returns:
            完整的提示词
        """
        # 构建示例部分
        examples_prompt = self._build_examples_prompt(correct_examples, mistake_examples)
        
        # 基础提示词结构
        prompt = f"""You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to answer the question, then look at the results of the query and return the answer.

{schema_text}

"""
        
        # 添加示例
        if examples_prompt:
            prompt += f"{examples_prompt}\n\n"
        
        # 添加错误历史
        if error_history:
            prompt += "Previous attempts and errors:\n"
            for i, error_info in enumerate(error_history, 1):
                prompt += f"Attempt {i}: SQL = {error_info['sql']}\n"
                prompt += f"Error: {error_info['error']}\n"
                if error_info.get('suggested_fix'):
                    prompt += f"Suggested fix: {error_info['suggested_fix']}\n"
            prompt += "\nPlease learn from these errors and generate a correct SQL query.\n\n"
        
        # 添加问题和指令
        prompt += f"""Question: {question}

Please follow these instructions:
1. Only return the SQL query, no explanations
2. Make sure to quote column names with special characters using double quotes
3. Use proper table aliases when joining multiple tables
4. Ensure the SQL is syntactically correct for SQLite
5. Start your response with SELECT

Now, generate the SQL query for the question above."""
        
        return prompt
    
    def _build_examples_prompt(self, correct_examples: List[Dict], mistake_examples: List[Dict]) -> str:
        """构建示例提示词部分"""
        prompt_parts = []
        
        # 添加正确示例
        if correct_examples:
            correct_prompt_parts = []
            for index, example in enumerate(correct_examples):
                example_text = f"example{index+1}: {{\n"
                example_text += f"## Question: {example.get('question', '')}\n"
                example_text += f"## SQL: {example.get('sql', '')}\n"
                if example.get('hint'):
                    example_text += f"## Hint: {example['hint']}\n"
                if example.get('thought process'):
                    example_text += f"## Thought process: {example['thought process']}\n"
                example_text += "\n}"
                correct_prompt_parts.append(example_text)
            
            correct_section = "### For your reference, here are some examples of Questions, SQL queries, and thought processes related to the Question you're working with\n\n"
            correct_section += "\n\n".join(correct_prompt_parts)
            prompt_parts.append(correct_section)
        
        # 添加错误示例
        if mistake_examples:
            mistake_prompt_parts = []
            for index, example in enumerate(mistake_examples):
                example_text = f"example{index+1}: {{\n"
                example_text += f"## Question: {example.get('question', '')}\n"
                example_text += f"## Error SQL: {example.get('error_sql', '')}\n"
                if example.get('compiler_hint'):
                    example_text += f"## Compiler hint: {example['compiler_hint']}\n"
                if example.get('reflective_cot'):
                    example_text += f"## Reflection: {example['reflective_cot']}\n"
                if example.get('ground_truth_sql'):
                    example_text += f"## Correct SQL: {example['ground_truth_sql']}\n"
                example_text += "\n}"
                mistake_prompt_parts.append(example_text)
            
            mistake_section = "### Below are examples of mistakes you've made before that are similar to the question you're about to tackle, so please refer to not making the same mistake!\n\n"
            mistake_section += "\n\n".join(mistake_prompt_parts)
            prompt_parts.append(mistake_section)
        
        return "\n\n".join(prompt_parts)
    
    def _extract_sql_from_response(self, response: str) -> str:
        """从LLM响应中提取SQL（与之前相同）"""
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
    
    def add_experience(self, question: str, sql: str, correct: bool = True, **kwargs):
        """
        添加新的经验到知识库
        
        Args:
            question: 问题文本
            sql: SQL查询
            correct: 是否正确
            **kwargs: 额外的元数据
        """
        try:
            self.experience_retriever.add_to_sets(question, sql, correct, **kwargs)
            self.logger.info(f"✓ 已将{'正确' if correct else '错误'}示例添加到知识库")
        except Exception as e:
            self.logger.error(f"添加经验失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取生成器统计信息"""
        return {
            **self.stats,
            'experience_retriever_stats': self.experience_retriever.get_statistics()
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("\n=== LPE-SQL 生成器统计信息 ===")
        print(f"总问题数: {stats['total_questions']}")
        print(f"成功生成: {stats['successful_generations']}")
        print(f"失败生成: {stats['failed_generations']}")
        print(f"经验模式使用: {stats['experience_mode_used']}")
        print(f"检索成功次数: {stats['retrieval_success_count']}")
        
        # 打印检索器统计信息
        self.experience_retriever.print_statistics()


# 测试函数
def test_lpe_sql_generator():
    """测试LPE-SQL生成器"""
    print("=== 测试LPE-SQL生成器 ===")
    
    try:
        # 初始化生成器
        generator = EnhancedSQLGeneratorLPE()
        
        # 测试schema获取
        schema = generator.get_detailed_schema("database.db")
        print(f"✓ 获取到 {len(schema['tables'])} 个表的schema")
        
        # 测试问题生成
        test_questions = [
            "Find the employee with the highest salary",
            "List all customers who have placed more than 5 orders",
            "What is the average rating for products in the electronics category?"
        ]
        
        for question in test_questions:
            print(f"\n测试问题: {question}")
            
            sql = generator.generate_sql_with_schema(question, schema, "database.db")
            
            if sql and not sql.startswith("SELECT"):
                print(f"⚠ 生成的SQL可能有问题: {sql[:100]}...")
            else:
                print(f"✓ 生成的SQL: {sql[:100]}...")
        
        # 打印统计信息
        generator.print_statistics()
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_lpe_sql_generator()