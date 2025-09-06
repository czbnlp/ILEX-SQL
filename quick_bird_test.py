#!/usr/bin/env python3
"""
BIRD数据集快速测试脚本
用于快速验证BIRD数据集评估功能
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append('src')

def test_bird_evaluation():
    """快速测试BIRD数据集评估"""
    print("=== BIRD数据集快速测试 ===")
    
    try:
        # 测试数据文件存在性
        data_dir = Path("LPE-SQL/data")
        dev_json = data_dir / "dev.json"
        dev_sql = data_dir / "dev.sql"
        
        print(f"检查数据文件...")
        print(f"  dev.json: {'✓' if dev_json.exists() else '✗'}")
        print(f"  dev.sql: {'✓' if dev_sql.exists() else '✗'}")
        
        if not dev_json.exists() or not dev_sql.exists():
            print("❌ 数据文件不完整")
            return
        
        # 测试数据库文件
        import json
        with open(dev_json, 'r') as f:
            data = json.load(f)
        
        sample_db_id = data[0]['db_id']
        db_path = data_dir / sample_db_id / f"{sample_db_id}.sqlite"
        print(f"  示例数据库 ({sample_db_id}): {'✓' if db_path.exists() else '✗'}")
        
        if not db_path.exists():
            print("❌ 数据库文件不存在，请先运行:")
            print("  python download_bird_databases.py --create-sample")
            return
        
        # 测试组件初始化
        print("\n测试组件初始化...")
        
        from llm_connector_local import LocalLLMConnector
        from sql_executor import SQLExecutor
        from ilex_core.mode_selector import ModeSelector
        
        print("  初始化LLM连接器...")
        connector = LocalLLMConnector()
        
        print("  初始化SQL执行器...")
        executor = SQLExecutor()
        
        print("  初始化模式选择器...")
        selector = ModeSelector(config_path="config/ilex_config.yaml")
        
        # 测试单个问题处理
        print("\n测试单个问题处理...")
        sample_question = data[0]['question']
        sample_db_id = data[0]['db_id']
        
        print(f"  问题: {sample_question}")
        print(f"  数据库: {sample_db_id}")
        
        # 获取数据库schema
        db_path_str = str(db_path)
        schema = get_db_schema(db_path_str)
        print(f"  Schema获取: {'✓' if schema else '✗'}")
        
        # 测试模式选择
        mode_decision = selector.get_mode_decision(sample_question)
        print(f"  模式选择: {mode_decision['mode']} (复杂度: {mode_decision['complexity_score']:.3f})")
        
        # 测试SQL生成
        if mode_decision['use_exploration_mode']:
            print("  使用探索模式...")
            print("  ⚠️  探索模式测试跳过（需要更长时间）")
        else:
            print("  使用经验模式...")
            prompt = f"""
            基于以下数据库schema，为问题生成SQL查询：
            
            问题: {sample_question}
            
            数据库Schema:
            {schema}
            
            请只返回SQL语句，不要包含其他解释。
            """
            
            print("  生成SQL...")
            sql = connector(prompt)
            print(f"  生成的SQL: {sql[:100]}...")
            
            # 测试SQL执行
            print("  执行SQL...")
            result, error = executor(sql, db_path_str)
            
            if error:
                print(f"  ❌ 执行失败: {error}")
            else:
                print(f"  ✓ 执行成功: {len(result)} 行结果")
        
        print("\n✅ 快速测试完成！")
        
        # 提供下一步建议
        print("\n下一步操作:")
        print("1. 完整评估: python bird_evaluator_fixed.py --max-questions 10")
        print("2. 交互模式: python run_example.py --interactive")
        print("3. BIRD评估: python run_example.py --bird")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def get_db_schema(db_path: str) -> str:
    """获取数据库schema信息"""
    try:
        import sqlite3
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


if __name__ == "__main__":
    test_bird_evaluation()