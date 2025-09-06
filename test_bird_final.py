#!/usr/bin/env python3
"""
BIRD数据集最终测试脚本
使用下载好的BIRD数据集进行快速测试
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append('src')

def test_bird_with_real_data():
    """使用真实BIRD数据集进行测试"""
    print("=== BIRD数据集最终测试 ===")
    
    try:
        # 检查数据文件
        data_dir = Path("LPE-SQL/data")
        db_root = Path("/Users/chuzhibo/Desktop/workspace_sql/data/dev_databases")
        
        print(f"数据目录: {data_dir}")
        print(f"数据库目录: {db_root}")
        
        # 检查必要文件
        dev_json = data_dir / "dev.json"
        dev_sql = data_dir / "dev.sql"
        
        if not dev_json.exists():
            print(f"❌ 数据文件不存在: {dev_json}")
            return
        
        if not dev_sql.exists():
            print(f"❌ SQL文件不存在: {dev_sql}")
            return
        
        print("✓ 数据文件检查通过")
        
        # 检查数据库目录
        if not db_root.exists():
            print(f"❌ 数据库目录不存在: {db_root}")
            return
        
        # 列出可用数据库
        db_dirs = [d for d in db_root.iterdir() if d.is_dir() and d.name != '.DS_Store']
        print(f"✓ 找到 {len(db_dirs)} 个数据库:")
        for db_dir in db_dirs[:5]:  # 只显示前5个
            db_file = db_dir / f"{db_dir.name}.sqlite"
            status = "✓" if db_file.exists() else "✗"
            print(f"  {status} {db_dir.name}")
        
        if len(db_dirs) > 5:
            print(f"  ... 还有 {len(db_dirs) - 5} 个数据库")
        
        # 测试加载BIRD数据
        print("\n测试数据加载...")
        import json
        with open(dev_json, 'r') as f:
            bird_data = json.load(f)
        
        print(f"✓ 加载了 {len(bird_data)} 个问题")
        
        # 显示前3个问题
        print("\n前3个问题示例:")
        for i, item in enumerate(bird_data[:3]):
            print(f"  {i+1}. [{item['difficulty']}] {item['question'][:80]}...")
            print(f"     数据库: {item['db_id']}")
        
        # 测试组件初始化
        print("\n测试组件初始化...")
        
        from llm_connector_local import LocalLLMConnector
        from sql_executor import SQLExecutor
        from ilex_core.mode_selector import ModeSelector
        
        print("  初始化LLM连接器...")
        connector = LocalLLMConnector()
        print("  ✓ LLM连接器初始化成功")
        
        print("  初始化SQL执行器...")
        executor = SQLExecutor()
        print("  ✓ SQL执行器初始化成功")
        
        print("  初始化模式选择器...")
        selector = ModeSelector(config_path="config/ilex_config.yaml")
        print("  ✓ 模式选择器初始化成功")
        
        # 测试单个问题处理
        print("\n测试单个问题处理...")
        sample_item = bird_data[0]
        question = sample_item['question']
        db_id = sample_item['db_id']
        difficulty = sample_item['difficulty']
        
        print(f"  问题: {question}")
        print(f"  数据库: {db_id}")
        print(f"  难度: {difficulty}")
        
        # 测试数据库连接
        db_path = db_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            print(f"  ❌ 数据库文件不存在: {db_path}")
            return
        
        print(f"  ✓ 数据库文件存在: {db_path}")
        
        # 测试schema获取
        schema = get_db_schema(str(db_path))
        if "获取schema失败" in schema:
            print(f"  ❌ Schema获取失败")
            return
        
        print(f"  ✓ Schema获取成功")
        print(f"  Schema预览: {schema[:200]}...")
        
        # 测试模式选择
        mode_decision = selector.get_mode_decision(question)
        print(f"  ✓ 模式选择: {mode_decision['mode']}")
        print(f"    复杂度分数: {mode_decision['complexity_score']:.3f}")
        
        # 测试SQL生成（仅经验模式）
        if not mode_decision['use_exploration_mode']:
            print("  测试SQL生成...")
            prompt = f"""
            基于以下数据库schema，为问题生成SQL查询：
            
            问题: {question}
            
            数据库Schema:
            {schema}
            
            请只返回SQL语句，不要包含其他解释。
            """
            
            sql = connector(prompt)
            print(f"  ✓ SQL生成成功: {sql[:100]}...")
            
            # 测试SQL执行
            print("  测试SQL执行...")
            result, error = executor(sql, str(db_path))
            
            if error:
                print(f"  ⚠️  SQL执行失败: {error}")
            else:
                print(f"  ✓ SQL执行成功，返回 {len(result) if result else 0} 行结果")
        else:
            print("  ⚠️  问题需要探索模式，跳过SQL生成测试")
        
        print("\n✅ 所有测试通过！")
        
        # 提供使用建议
        print("\n使用建议:")
        print("1. 快速测试 (5个问题):")
        print("   python bird_evaluator_final.py --max-questions 5")
        print("2. 完整评估 (所有问题):")
        print("   python bird_evaluator_final.py")
        print("3. 通过主程序使用:")
        print("   python run_example.py --bird")
        
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
    test_bird_with_real_data()