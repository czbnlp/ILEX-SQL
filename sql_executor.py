import sqlite3
import sqlalchemy
from sqlalchemy import create_engine, text
from typing import Tuple, Optional, Any
import pandas as pd
import os

class SQLExecutor:
    """SQL执行器类"""
    
    def __init__(self, database_url: str = "sqlite:///database.db"):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        
        # 如果数据库文件不存在，创建示例数据库
        if not os.path.exists(database_url.replace("sqlite:///", "")):
            self._create_sample_database()
        
    def _create_sample_database(self):
        """创建示例数据库"""
        print("创建示例数据库...")
        
        # 创建员工表示例数据
        employees_data = [
            (1, "张三", 5000, "IT"),
            (2, "李四", 6000, "Sales"),
            (3, "王五", 7000, "IT"),
            (4, "赵六", 5500, "HR"),
            (5, "钱七", 8000, "Sales")
        ]
        
        # 创建部门表示例数据
        departments_data = [
            (1, "IT", "技术部"),
            (2, "Sales", "销售部"),
            (3, "HR", "人力资源部")
        ]
        
        try:
            with self.engine.connect() as conn:
                # 创建员工表
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS employees (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        salary REAL,
                        department_id INTEGER
                    )
                """))
                
                # 创建部门表
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS departments (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT
                    )
                """))
                
                # 插入示例数据
                conn.execute(text("""
                    INSERT OR IGNORE INTO employees (id, name, salary, department_id) 
                    VALUES (?, ?, ?, ?)
                """), employees_data)
                
                conn.execute(text("""
                    INSERT OR IGNORE INTO departments (id, name, description) 
                    VALUES (?, ?, ?)
                """), departments_data)
                
                conn.commit()
                print("✓ 示例数据库创建成功")
                
        except Exception as e:
            print(f"创建示例数据库失败: {e}")
    
    def __call__(self, sql: str, db_path: Optional[str] = None) -> Tuple[Any, Optional[str]]:
        """
        执行SQL查询
        
        Args:
            sql: SQL查询语句
            db_path: 数据库路径（可选）
            
        Returns:
            (查询结果, 错误信息)
        """
        try:
            if db_path:
                # 如果指定了数据库路径，使用该路径
                engine = create_engine(f"sqlite:///{db_path}")
            else:
                engine = self.engine
                
            with engine.connect() as connection:
                result = pd.read_sql_query(text(sql), connection)
                return result.to_dict('records'), None
                
        except Exception as e:
            return None, str(e)
    
    def get_schema(self, table_name: Optional[str] = None) -> str:
        """
        获取数据库schema信息
        
        Args:
            table_name: 表名（可选）
            
        Returns:
            Schema信息字符串
        """
        try:
            with self.engine.connect() as connection:
                if table_name:
                    query = f"PRAGMA table_info({table_name});"
                    result = pd.read_sql_query(query, connection)
                    schema_info = f"表 {table_name} 的结构:\n{result.to_string(index=False)}"
                else:
                    query = "SELECT name FROM sqlite_master WHERE type='table';"
                    tables = pd.read_sql_query(query, connection)
                    
                    if tables.empty:
                        return "数据库中没有表"
                    
                    schema_info = "数据库中的表:\n"
                    for table in tables['name']:
                        schema_info += f"- {table}\n"
                        
                        # 获取每个表的结构
                        table_info_query = f"PRAGMA table_info({table});"
                        table_info = pd.read_sql_query(table_info_query, connection)
                        schema_info += f"  表 {table} 结构:\n"
                        schema_info += f"  {table_info.to_string(index=False)}\n"
                        
        except Exception as e:
            return f"获取schema失败: {e}"
        
        return schema_info
    
    def execute_and_display(self, sql: str, db_path: Optional[str] = None) -> bool:
        """
        执行SQL并显示结果
        
        Args:
            sql: SQL查询语句
            db_path: 数据库路径（可选）
            
        Returns:
            执行是否成功
        """
        print(f"执行SQL: {sql}")
        
        result, error = self(sql, db_path)
        
        if error:
            print(f"✗ 执行失败: {error}")
            return False
        
        if not result:
            print("查询结果为空")
            return True
        
        print("✓ 执行成功，结果:")
        for i, row in enumerate(result):
            print(f"  {i+1}. {row}")
        
        return True


# 测试函数
def test_sql_executor():
    """测试SQL执行器"""
    print("=== 测试SQL执行器 ===")
    
    executor = SQLExecutor()
    
    # 显示数据库schema
    print("\n数据库Schema:")
    schema = executor.get_schema()
    print(schema)
    
    # 测试简单查询
    print("\n测试查询1: 查询所有员工")
    executor.execute_and_display("SELECT * FROM employees")
    
    # 测试带条件的查询
    print("\n测试查询2: 查询薪水大于6000的员工")
    executor.execute_and_display("SELECT * FROM employees WHERE salary > 6000")
    
    # 测试聚合查询
    print("\n测试查询3: 计算各部门平均薪水")
    executor.execute_and_display("SELECT department_id, AVG(salary) as avg_salary FROM employees GROUP BY department_id")
    
    # 测试连接查询
    print("\n测试查询4: 员工和部门连接查询")
    executor.execute_and_display("SELECT e.name, e.salary, d.name as department FROM employees e LEFT JOIN departments d ON e.department_id = d.id")


if __name__ == "__main__":
    test_sql_executor()