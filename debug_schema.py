#!/usr/bin/env python3
"""
调试数据库Schema获取问题
"""

import sqlite3
import sys
from pathlib import Path

def debug_db_schema():
    """调试数据库Schema获取"""
    print("=== 调试数据库Schema获取 ===")
    
    db_path = "/Users/chuzhibo/Desktop/workspace_sql/data/dev_databases/california_schools/california_schools.sqlite"
    
    try:
        print(f"尝试连接数据库: {db_path}")
        
        # 测试基本连接
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("✓ 数据库连接成功")
        
        # 测试获取表名
        print("\n获取表名...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"✓ 找到 {len(tables)} 个表:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # 测试获取表结构
        if tables:
            table_name = tables[0][0]
            print(f"\n获取表 {table_name} 的结构...")
            
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            print(f"✓ 表 {table_name} 有 {len(columns)} 列:")
            for col in columns:
                col_name, col_type, not_null, default_val, is_pk = col
                print(f"  - {col_name} {col_type} (NOT NULL: {not_null}, PK: {is_pk})")
        
        conn.close()
        print("\n✓ 所有测试通过")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_db_schema()