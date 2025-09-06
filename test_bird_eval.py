#!/usr/bin/env python3
"""
BIRD数据集评估测试脚本
快速测试ILEX-SQL系统在BIRD数据集上的性能
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append('src')

from bird_evaluator import BIRDEvaluator


def test_bird_evaluation():
    """测试BIRD数据集评估"""
    print("=== BIRD数据集评估测试 ===")
    
    try:
        # 创建评估器
        evaluator = BIRDEvaluator(
            data_dir="LPE-SQL/data",
            db_root="LPE-SQL/data",
            use_local_model=True,
            timeout=30
        )
        
        print("✓ 评估器初始化成功")
        
        # 测试加载单个问题
        bird_data = evaluator.load_bird_data("dev")
        if len(bird_data) > 0:
            sample_question = bird_data[0]
            print(f"\n示例问题: {sample_question['question']}")
            print(f"数据库: {sample_question['db_id']}")
            print(f"难度: {sample_question['difficulty']}")
            
            # 测试生成SQL
            predicted_sql, success, details = evaluator.generate_sql(
                sample_question['question'], 
                sample_question['db_id']
            )
            
            if success:
                print(f"✓ 生成的SQL: {predicted_sql}")
            else:
                print(f"✗ SQL生成失败: {details.get('error', '未知错误')}")
        
        # 执行小规模评估（前5个问题）
        print(f"\n开始小规模评估（前5个问题）...")
        stats = evaluator.evaluate_dataset(
            split="dev",
            max_questions=5,
            output_file="test_bird_results.json"
        )
        
        print(f"\n✓ 测试评估完成！")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("BIRD数据集评估测试开始...")
    print("=" * 50)
    
    # 检查必要文件
    required_files = [
        "LPE-SQL/data/dev.json",
        "LPE-SQL/data/dev.sql"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少必要文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n请确保LPE-SQL项目的data目录包含BIRD数据集文件")
        return
    
    test_bird_evaluation()


if __name__ == "__main__":
    main()