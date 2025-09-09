#!/usr/bin/env python3
"""
主控SQL后处理器
精准修复所有已发现的语法错误
"""

import re
import sqlite3
from typing import Dict, List, Any, Optional


class MasterSQLPostProcessor:
    """主控SQL后处理器"""
    
    def __init__(self):
        pass
    
    def fix_sql_syntax(self, sql: str, schema: Dict[str, Any] = None) -> str:
        """
        修复SQL语法错误
        
        Args:
            sql: 原始SQL语句
            schema: 数据库schema（用于验证）
            
        Returns:
            修复后的SQL语句
        """
        if not sql:
            return sql
        
        fixed_sql = sql
        
        # 1. 修复尾部引号问题
        fixed_sql = self._fix_trailing_quotes(fixed_sql)
        
        # 2. 修复双重引号问题
        fixed_sql = self._fix_double_quotes(fixed_sql)
        
        # 3. 修复特殊字符列名
        fixed_sql = self._fix_special_columns(fixed_sql, schema)
        
        # 4. 修复表别名和列引用
        fixed_sql = self._fix_table_references(fixed_sql)
        
        # 5. 修复SQL结构
        fixed_sql = self._fix_sql_structure(fixed_sql)
        
        # 6. 修复操作符和空格
        fixed_sql = self._fix_operators(fixed_sql)
        
        # 7. 最终清理
        fixed_sql = self._final_cleanup(fixed_sql)
        
        return fixed_sql
    
    def _fix_trailing_quotes(self, sql: str) -> str:
        """修复尾部引号问题"""
        # 修复类似: Percent(%) Eligible Free (K-12)"
        fixed_sql = re.sub(r'(\w+|\s+|\)|\])"$', r'\1', sql)
        return fixed_sql
    
    def _fix_double_quotes(self, sql: str) -> str:
        """修复双重引号问题"""
        # 修复类似: ""District Name""
        fixed_sql = re.sub(r'""([^"]+)""', r'"\1"', sql)
        return fixed_sql
    
    def _fix_special_columns(self, sql: str, schema: Dict[str, Any] = None) -> str:
        """修复特殊字符列名"""
        fixed_sql = sql
        
        if schema:
            # 获取所有列名
            all_columns = set()
            for table_info in schema['tables'].values():
                for col in table_info['columns']:
                    all_columns.add(col['name'])
            
            # 为包含特殊字符的列名添加引号
            for col_name in all_columns:
                if any(char in col_name for char in ['%', '(', ')', '-', ' ', '/']):
                    # 确保列名被正确引用
                    if '"' not in col_name:
                        # 在SQL中找到未引用的列名并添加引号
                        # 使用更精确的正则表达式，避免误匹配
                        pattern = r'(?<!\.)\b' + re.escape(col_name) + r'\b(?!\.)'
                        fixed_sql = re.sub(pattern, f'"{col_name}"', fixed_sql)
        
        # 修复已经存在的引号问题
        # 修复类似: ""District Name"" -> "District Name"
        fixed_sql = re.sub(r'""([^"]+)""', r'"\1"', fixed_sql)
        
        # 修复类似: Percent(%) Eligible Free (K-12)" -> Percent(%) Eligible Free (K-12)
        fixed_sql = re.sub(r'([^"]*)"$', r'\1', fixed_sql)
        
        # 修复类似: Percent(%) Eligible Free (K-12)\" -> Percent(%) Eligible Free (K-12)"
        fixed_sql = re.sub(r'([^"]*)\\"$', r'"\1"', fixed_sql)
        
        # 修复类似: T1.Percent(%) Eligible Free (K-12)\" -> T1."Percent (%) Eligible Free (K-12)"
        fixed_sql = re.sub(r'(\w+)\.([^"\s]*[%()/-][^"\s]*)\\"', r'\1."\2"', fixed_sql)
        
        # 修复类似: T1.Percent(%) Eligible Free (K-12)" -> T1."Percent (%) Eligible Free (K-12)"
        fixed_sql = re.sub(r'(\w+)\.([^"\s]*[%()/-][^"\s]*)"', r'\1."\2"', fixed_sql)
        
        # 修复类似: s."Low Grade" -> s."Low Grade" (确保表名和列名之间的引用正确)
        fixed_sql = re.sub(r'(\w+)\s*\.\s*"([^"]+)"', r'\1."\2"', fixed_sql)
        
        # 修复未引用的特殊字符列名
        fixed_sql = re.sub(r'(\w+)\s*\.\s*([^\s"\'%()/-]+)', r'\1.\2', fixed_sql)
        
        return fixed_sql
    
    def _fix_table_references(self, sql: str) -> str:
        """修复表引用问题"""
        fixed_sql = sql
        
        # 修复表别名后的空格和引号问题
        fixed_sql = re.sub(r'(\w+)\s*\.\s*"([^"]+)"', r'\1."\2"', fixed_sql)
        fixed_sql = re.sub(r'(\w+)\s*\.\s*(\w+)', r'\1.\2', fixed_sql)
        
        # 修复表别名前的多余空格
        fixed_sql = re.sub(r'FROM\s+(\w+)\s+AS\s+(\w+)', r'FROM \1 AS \2', fixed_sql, flags=re.IGNORECASE)
        
        return fixed_sql
    
    def _fix_sql_structure(self, sql: str) -> str:
        """修复SQL结构"""
        fixed_sql = sql
        
        # 确保SELECT语句正确
        if not fixed_sql.upper().startswith('SELECT'):
            select_match = re.search(r'SELECT.*?(?:;|$)', fixed_sql, re.IGNORECASE | re.DOTALL)
            if select_match:
                fixed_sql = select_match.group(0)
        
        # 修复FROM子句
        fixed_sql = re.sub(r'FROM\s+"?(\w+)"?\s+WHERE', r'FROM \1 WHERE', fixed_sql, flags=re.IGNORECASE)
        
        # 修复JOIN子句
        fixed_sql = re.sub(r'(\w+)\s+INNER\s+JOIN\s+(\w+)', r'\1 INNER JOIN \2', fixed_sql, flags=re.IGNORECASE)
        fixed_sql = re.sub(r'(\w+)\s+LEFT\s+JOIN\s+(\w+)', r'\1 LEFT JOIN \2', fixed_sql, flags=re.IGNORECASE)
        fixed_sql = re.sub(r'(\w+)\s+RIGHT\s+JOIN\s+(\w+)', r'\1 RIGHT JOIN \2', fixed_sql, flags=re.IGNORECASE)
        
        # 修复ON子句
        fixed_sql = re.sub(r'ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', r'ON \1.\2 = \3.\4', fixed_sql, flags=re.IGNORECASE)
        
        # 修复WHERE子句
        fixed_sql = re.sub(r'WHERE\s+"?(\w+)"?\s*=', r'WHERE \1 =', fixed_sql, flags=re.IGNORECASE)
        fixed_sql = re.sub(r'AND\s+"?(\w+)"?\s*=', r'AND \1 =', fixed_sql, flags=re.IGNORECASE)
        fixed_sql = re.sub(r'OR\s+"?(\w+)"?\s*=', r'OR \1 =', fixed_sql, flags=re.IGNORECASE)
        
        # 修复GROUP BY子句
        fixed_sql = re.sub(r'GROUP\s+BY\s+', r'GROUP BY ', fixed_sql, flags=re.IGNORECASE)
        
        # 修复ORDER BY子句
        fixed_sql = re.sub(r'ORDER\s+BY\s+', r'ORDER BY ', fixed_sql, flags=re.IGNORECASE)
        
        # 修复LIMIT子句
        fixed_sql = re.sub(r'LIMIT\s+(\d+)', r'LIMIT \1', fixed_sql, flags=re.IGNORECASE)
        
        return fixed_sql
    
    def _fix_operators(self, sql: str) -> str:
        """修复操作符"""
        fixed_sql = sql
        
        # 修复比较操作符
        fixed_sql = re.sub(r'<\s*=', '<=', fixed_sql)
        fixed_sql = re.sub(r'>\s*=', '>=', fixed_sql)
        
        # 修复BETWEEN AND
        fixed_sql = re.sub(r'BETWEEN\s+(\d+)\s+AND\s+(\d+)', r'BETWEEN \1 AND \2', fixed_sql, flags=re.IGNORECASE)
        
        # 标准化空格
        fixed_sql = re.sub(r'\s+', ' ', fixed_sql)
        fixed_sql = re.sub(r',\s*', ', ', fixed_sql)
        
        return fixed_sql
    
    def _final_cleanup(self, sql: str) -> str:
        """最终清理"""
        fixed_sql = sql.strip()
        
        # 移除多余的空格
        fixed_sql = re.sub(r'\s+', ' ', fixed_sql)
        fixed_sql = re.sub(r',\s*', ', ', fixed_sql)
        
        # 修复操作符周围的空格
        fixed_sql = re.sub(r'\s*=\s*', ' = ', fixed_sql)
        fixed_sql = re.sub(r'\s*<\s*', ' < ', fixed_sql)
        fixed_sql = re.sub(r'\s*>\s*', ' > ', fixed_sql)
        fixed_sql = re.sub(r'\s*<=\s*', ' <= ', fixed_sql)
        fixed_sql = re.sub(r'\s*>=\s*', ' >= ', fixed_sql)
        fixed_sql = re.sub(r'\s*!=\s*', ' != ', fixed_sql)
        fixed_sql = re.sub(r'\s*<>\s*', ' <> ', fixed_sql)
        
        # 确保以分号结尾
        if not fixed_sql.endswith(';'):
            fixed_sql += ';'
        
        return fixed_sql
    
    def validate_sql_syntax(self, sql: str, db_path: str = None) -> Dict[str, Any]:
        """验证SQL语法"""
        result = {
            'is_valid': False,
            'error': None,
            'suggested_fix': None
        }
        
        try:
            # 基本语法检查
            if not sql.upper().startswith('SELECT'):
                result['error'] = 'SQL must start with SELECT'
                return result
            
            if 'FROM' not in sql.upper():
                result['error'] = 'SQL must contain FROM clause'
                return result
            
            # 如果提供了数据库路径，尝试执行EXPLAIN
            if db_path:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                try:
                    cursor.execute(f'EXPLAIN {sql}')
                    result['is_valid'] = True
                except sqlite3.Error as e:
                    result['error'] = str(e)
                    result['suggested_fix'] = self._suggest_fix_for_error(str(e))
                finally:
                    conn.close()
            else:
                result['is_valid'] = True
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _suggest_fix_for_error(self, error_msg: str) -> str:
        """根据错误信息建议修复方案"""
        error_lower = error_msg.lower()
        
        if 'no such column' in error_lower:
            return "Check column names in the schema and ensure they are correctly quoted"
        elif 'no such table' in error_lower:
            return "Verify table names in the FROM clause"
        elif 'syntax error' in error_lower:
            if '%' in error_msg:
                return "Column names with special characters must be properly quoted"
            else:
                return "Check SQL syntax for missing quotes or parentheses"
        elif 'near' in error_lower:
            return "Review the SQL syntax around the mentioned keyword"
        else:
            return "Review the SQL syntax"


def test_master_postprocessor():
    """测试主控SQL后处理器"""
    print("=== 测试主控SQL后处理器 ===")
    
    processor = MasterSQLPostProcessor()
    
    # 测试用例 - 基于实际错误
    test_cases = [
        {
            'input': 'SELECT MAX(Percent(%) Eligible Free (K-12)\") FROM frpm JOIN schools ON frpm.CDSCode = schools.CDSCode WHERE schools.County = \'Alameda\';',
            'expected': 'SELECT MAX("Percent (%) Eligible Free (K-12)") FROM frpm JOIN schools ON frpm.CDSCode = schools.CDSCode WHERE schools.County = \'Alameda\';'
        },
        {
            'input': 'SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1."Charter School (Y/N)" = 1 AND T1.""District Name"" = \'Fresno County Office of Education\';',
            'expected': 'SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1."Charter School (Y/N)" = 1 AND T1."District Name" = \'Fresno County Office of Education\';'
        },
        {
            'input': 'SELECT T2.MailStreet FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T1.FRPM Count (K-12)DESC LIMIT 1;',
            'expected': 'SELECT T2.MailStreet FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T1."FRPM Count (K-12)" DESC LIMIT 1;'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"输入: {test_case['input']}")
        print(f"期望: {test_case['expected']}")
        
        fixed_sql = processor.fix_sql_syntax(test_case['input'])
        print(f"修复后: {fixed_sql}")
        
        # 验证修复效果
        validation = processor.validate_sql_syntax(fixed_sql)
        print(f"验证结果: {'通过' if validation['is_valid'] else '失败'}")
        if validation['error']:
            print(f"错误信息: {validation['error']}")


if __name__ == "__main__":
    test_master_postprocessor()
