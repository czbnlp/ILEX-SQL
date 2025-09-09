#!/usr/bin/env python3
"""
增强的SQL生成器
基于数据库schema和问题分析生成准确的SQL查询
"""

import re
import json
import sqlite3
from typing import Dict, List, Tuple, Any, Optional
from llm_connector_local import LocalLLMConnector
from master_sql_postprocessor import MasterSQLPostProcessor


class EnhancedSQLGenerator:
    """增强的SQL生成器"""
    
    def __init__(self, llm_connector=None):
        """
        初始化SQL生成器
        
        Args:
            llm_connector: LLM连接器
        """
        self.llm_connector = llm_connector or LocalLLMConnector()
        
        # SQL关键词映射
        self.sql_keyword_mapping = {
            'count': 'COUNT',
            'sum': 'SUM',
            'average': 'AVG',
            'avg': 'AVG',
            'maximum': 'MAX',
            'max': 'MAX',
            'minimum': 'MIN',
            'min': 'MIN',
            'total': 'COUNT',
            'number': 'COUNT',
            'list': 'SELECT',
            'show': 'SELECT',
            'find': 'SELECT',
            'get': 'SELECT'
        }
        
        # 比较操作符映射
        self.comparison_mapping = {
            'greater than': '>',
            'more than': '>',
            'higher than': '>',
            'less than': '<',
            'fewer than': '<',
            'lower than': '<',
            'equal to': '=',
            'equals': '=',
            'same as': '=',
            'not equal': '!=',
            'different from': '!='
        }
    
    def get_detailed_schema(self, db_path: str) -> Dict[str, Any]:
        """
        获取详细的数据库schema信息
        
        Args:
            db_path: 数据库路径
            
        Returns:
            详细的schema信息
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
                    # 处理不同数量的列
                    if len(col) >= 6:
                        col_name, col_type, not_null, default_val, is_pk = col[1], col[2], bool(col[3]), col[4], bool(col[5])
                    elif len(col) >= 5:
                        col_name, col_type, not_null, default_val, is_pk = col[1], col[2], bool(col[3]), col[4], False
                    else:
                        # 处理返回列数不足的情况
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
            print(f"获取详细schema失败: {e}")
            return {'tables': {}, 'relationships': [], 'sample_data': {}}
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        分析问题，提取关键信息
        
        Args:
            question: 问题文本
            
        Returns:
            分析结果
        """
        question_lower = question.lower()
        
        analysis = {
            'question_type': 'simple',
            'requested_aggregations': [],
            'requested_columns': [],
            'filters': [],
            'joins_needed': False,
            'subqueries_needed': False,
            'comparison_operators': [],
            'target_tables': []
        }
        
        # 检测聚合操作
        for keyword, sql_func in self.sql_keyword_mapping.items():
            if keyword in question_lower:
                if sql_func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']:
                    analysis['requested_aggregations'].append(sql_func)
        
        # 检测比较操作
        for phrase, operator in self.comparison_mapping.items():
            if phrase in question_lower:
                analysis['comparison_operators'].append(operator)
        
        # 检测连接需求
        join_keywords = ['join', 'combine', 'together', 'related', 'associated']
        analysis['joins_needed'] = any(keyword in question_lower for keyword in join_keywords)
        
        # 检测子查询需求
        subquery_keywords = ['where', 'having', 'that', 'which', 'who', 'whose']
        analysis['subqueries_needed'] = any(keyword in question_lower for keyword in subquery_keywords)
        
        return analysis
    
    def find_relevant_tables(self, question: str, schema: Dict[str, Any]) -> List[str]:
        """
        根据问题找到相关的表
        
        Args:
            question: 问题文本
            schema: 数据库schema
            
        Returns:
            相关表名列表
        """
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
    
    def generate_sql_with_schema(self, question: str, schema: Dict[str, Any], db_path: str, max_retries: int = 2, error_history: list = None) -> str:
        """
        基于schema生成SQL，带错误反馈机制
        
        Args:
            question: 问题文本
            schema: 数据库schema
            db_path: 数据库路径
            max_retries: 最大重试次数
            error_history: 错误历史记录
            
        Returns:
            生成的SQL语句
        """
        if error_history is None:
            error_history = []
        
        # 分析问题
        analysis = self.analyze_question(question)
        
        # 找到相关表
        relevant_tables = self.find_relevant_tables(question, schema)
        
        # 构建完整的prompt（参考LPE-SQL的设计），包含错误反馈
        prompt = self._build_comprehensive_prompt(question, schema, db_path, relevant_tables, error_history)
        
        # 调用LLM生成SQL
        response = self.llm_connector(prompt)
        
        # 提取SQL语句
        sql = self._extract_sql_from_response(response)
        
        # 使用SQL后处理器修复语法错误
        postprocessor = MasterSQLPostProcessor()
        corrected_sql = postprocessor.fix_sql_syntax(sql, schema)
        
        # 验证修复后的SQL
        validation = postprocessor.validate_sql_syntax(corrected_sql, db_path)
        
        if not validation['is_valid']:
            error_msg = validation['error']
            print(f"SQL验证失败: {error_msg}")
            
            # 如果还有重试机会，将错误信息添加到历史中并重试
            if max_retries > 0:
                error_history.append({
                    'sql': corrected_sql,
                    'error': error_msg,
                    'suggested_fix': validation.get('suggested_fix', '')
                })
                print(f"重试生成SQL，剩余重试次数: {max_retries}")
                return self.generate_sql_with_schema(question, schema, db_path, max_retries - 1, error_history)
            else:
                # 如果没有重试机会了，尝试进一步修复
                corrected_sql = self._fallback_correction(corrected_sql, schema)
        
        return corrected_sql
    
    def _build_schema_text(self, schema: Dict[str, Any], relevant_tables: List[str]) -> str:
        """
        构建schema文本（参考LPE-SQL的格式）
        
        Args:
            schema: 数据库schema
            relevant_tables: 相关表列表
            
        Returns:
            schema文本
        """
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
            
            # 添加示例数据（格式化显示）
            if table_name in schema['sample_data']:
                sample_rows = schema['sample_data'][table_name]
                if sample_rows:
                    # 获取列名
                    column_names = [col['name'] for col in table_info['columns']]
                    
                    # 格式化示例数据
                    schema_parts.append(f"\n/*")
                    schema_parts.append(f" {len(sample_rows)} example rows:")
                    schema_parts.append(f" SELECT * FROM {table_name} LIMIT {len(sample_rows)};")
                    
                    # 计算列宽
                    col_widths = [max(len(str(col)), max(len(str(row[i])) for row in sample_rows)) for i, col in enumerate(column_names)]
                    
                    # 表头
                    header = " | ".join(col.ljust(width) for col, width in zip(column_names, col_widths))
                    schema_parts.append(f" {header}")
                    
                    # 分隔线
                    separator = " | ".join("-" * width for width in col_widths)
                    schema_parts.append(f" {separator}")
                    
                    # 数据行
                    for row in sample_rows:
                        data_row = " | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths))
                        schema_parts.append(f" {data_row}")
                    
                    schema_parts.append(f" */")
            
            schema_parts.append("")  # 空行分隔
        
        return "\n".join(schema_parts)
    
    def _get_few_shot_examples(self) -> str:
        """
        获取Few-shot示例（参考LPE-SQL的设计）
        
        Returns:
            Few-shot示例文本
        """
        examples = [
            """{
                question: What is the average rating for movie titled 'When Will I Be Loved'?
                hint: average rating = DIVIDE((SUM(rating_score where movie_title = 'When Will I Be Loved')), COUNT(rating_score));
                sql: SELECT AVG(T2.rating_score) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = 'When Will I Be Loved';
            }""",
            """{
                question: List all customers who have placed more than 10 orders.
                hint: count orders for each customer and filter where count > 10;
                sql: SELECT c.* FROM customers c INNER JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id HAVING COUNT(o.order_id) > 10;
            }""",
        ]
        
        return "\n\n".join(examples)
    
    def _build_comprehensive_prompt(self, question: str, schema: Dict[str, Any], db_path: str, relevant_tables: List[str], error_history: list) -> str:
        """
        构建完整的提示词
        
        Args:
            question: 问题文本
            schema: 数据库schema
            db_path: 数据库路径
            relevant_tables: 相关表列表
            error_history: 错误历史记录
            
        Returns:
            完整的提示词
        """
        schema_text = self._build_schema_text(schema, relevant_tables)
        few_shot_examples = self._get_few_shot_examples()
        
        prompt = f"""You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to answer the question, then look at the results of the query and return the answer.

{schema_text}

Question: {question}

"""
        
        # 添加错误反馈
        if error_history:
            prompt += "Previous attempts and errors:\n"
            for i, error_info in enumerate(error_history, 1):
                prompt += f"Attempt {i}: SQL = {error_info['sql']}\n"
                prompt += f"Error: {error_info['error']}\n"
                if error_info.get('suggested_fix'):
                    prompt += f"Suggested fix: {error_info['suggested_fix']}\n"
            prompt += "\nPlease learn from these errors and generate a correct SQL query.\n\n"
        
        prompt += f"""Please follow these instructions:
1. Only return the SQL query, no explanations
2. Make sure to quote column names with special characters using double quotes
3. Use proper table aliases when joining multiple tables
4. Ensure the SQL is syntactically correct for SQLite

Few-shot examples:
{few_shot_examples}

Now, generate the SQL query for the question above.
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
    
    def _fallback_correction(self, sql: str, schema: Dict[str, Any]) -> str:
        """
        后备修复方案
        
        Args:
            sql: SQL语句
            schema: 数据库schema
            
        Returns:
            修复后的SQL语句
        """
        # 简单的修复逻辑
        corrected_sql = sql
        
        # 确保以分号结尾
        if not corrected_sql.endswith(';'):
            corrected_sql += ';'
        
        return corrected_sql


# 测试函数
def test_enhanced_sql_generator():
    """测试增强SQL生成器"""
    print("=== 测试增强SQL生成器 ===")
    
    try:
        generator = EnhancedSQLGenerator()
        print("✓ 增强SQL生成器初始化成功")
        
        # 测试schema获取
        schema = generator.get_detailed_schema("database.db")
        print(f"✓ 获取到 {len(schema['tables'])} 个表的schema")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


if __name__ == "__main__":
    test_enhanced_sql_generator()