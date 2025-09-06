import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径到Python路径
sys.path.append('src')

from ilex_core.mode_selector import ModeSelector
from ilex_core.exploration_engine import ExplorationEngine
from llm_connector_local import LocalLLMConnector
from sql_executor import SQLExecutor

def main():
    """主运行函数"""
    print("=== ILEX-SQL 双模式智能SQL生成系统 ===")
    print("系统启动中...")
    
    # 初始化组件
    print("初始化LLM连接器...")
    llm_connector = LocalLLMConnector()
    
    print("初始化SQL执行器...")
    sql_executor = SQLExecutor()
    
    print("初始化模式选择器...")
    mode_selector = ModeSelector()
    
    print("初始化探索引擎...")
    exploration_engine = ExplorationEngine(
        llm_connector=llm_connector,
        sql_executor=sql_executor
    )
    
    # 测试LLM连接
    print("\n测试LLM连接...")
    if llm_connector.test_connection():
        print("✓ 本地LLM连接正常")
    else:
        print("✗ 本地LLM连接失败，请检查vLLM服务是否启动")
        return
    
    # 获取数据库schema
    print("\n获取数据库schema...")
    schema = sql_executor.get_schema()
    print(schema)
    
    # 示例问题
    test_questions = [
        "查询所有员工的信息",
        "找出薪水最高的员工",
        "计算IT部门的平均薪水",
        "找出薪水大于6000的员工及其部门信息",
        "First, find the department with the highest average salary, then list all employees in that department"
    ]
    
    # 处理每个问题
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"问题 {i}: {question}")
        print(f"{'='*60}")
        
        # 1. 评估问题复杂度并选择模式
        print("步骤1: 评估问题复杂度...")
        mode_decision = mode_selector.get_mode_decision(question)
        print(f"复杂度分数: {mode_decision['complexity_score']:.3f}")
        print(f"选择模式: {mode_decision['mode']}")
        print(f"推理: {mode_decision['reasoning']}")
        
        # 2. 根据选择的模式处理问题
        if mode_decision['use_exploration_mode']:
            print("步骤2: 使用探索模式处理...")
            final_sql, success, details = exploration_engine.solve_complex_question(
                question,
                "database.db",  # 数据库路径
                schema
            )
        else:
            print("步骤2: 使用经验模式处理...")
            # 简单的经验模式实现：直接调用LLM生成SQL
            prompt = f"""
            基于以下数据库schema，为问题生成SQL查询：
            
            问题: {question}
            
            数据库Schema:
            {schema}
            
            请只返回SQL语句，不要包含其他解释。
            """
            
            final_sql = llm_connector(prompt)
            success = True
            details = {"mode": "experience"}
        
        # 3. 输出结果
        if success and final_sql:
            print(f"✓ 生成的SQL: {final_sql}")
            print("步骤3: 执行SQL查询...")
            result, error = sql_executor(final_sql)
            if error:
                print(f"✗ 执行错误: {error}")
            else:
                print("✓ 执行结果:")
                if result:
                    for j, row in enumerate(result[:5]):  # 只显示前5行
                        print(f"  {j+1}. {row}")
                else:
                    print("  查询结果为空")
        else:
            print("✗ 处理失败")
            print(f"详细信息: {details}")
        
        print("-" * 60)

def test_individual_components():
    """测试各个组件"""
    print("\n=== 测试各个组件 ===")
    
    # 测试LLM连接器
    print("\n1. 测试LLM连接器...")
    llm_connector = LocalLLMConnector()
    if llm_connector.test_connection():
        print("✓ 本地LLM连接器正常")
    else:
        print("✗ 本地LLM连接器失败")
    
    # 测试SQL执行器
    print("\n2. 测试SQL执行器...")
    sql_executor = SQLExecutor()
    test_sql = "SELECT COUNT(*) as total_employees FROM employees"
    result, error = sql_executor(test_sql)
    if error:
        print(f"✗ SQL执行器失败: {error}")
    else:
        print(f"✓ SQL执行器正常，查询结果: {result}")
    
    # 测试模式选择器
    print("\n3. 测试模式选择器...")
    mode_selector = ModeSelector()
    test_question = "Find the employee with the highest salary"
    decision = mode_selector.get_mode_decision(test_question)
    print(f"✓ 模式选择器正常，决策: {decision['mode']}")
    
    print("\n=== 组件测试完成 ===")

def interactive_mode():
    """交互模式"""
    print("\n=== 交互模式 ===")
    print("输入你的问题（输入 'quit' 退出）:")
    
    # 初始化组件
    llm_connector = LocalLLMConnector()
    sql_executor = SQLExecutor()
    mode_selector = ModeSelector()
    exploration_engine = ExplorationEngine(
        llm_connector=llm_connector,
        sql_executor=sql_executor
    )
    
    while True:
        try:
            question = input("\n> 请输入你的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("感谢使用ILEX-SQL系统！")
                break
            
            if not question:
                continue
            
            print(f"\n处理问题: {question}")
            
            # 评估问题复杂度
            mode_decision = mode_selector.get_mode_decision(question)
            print(f"复杂度分数: {mode_decision['complexity_score']:.3f}")
            print(f"选择模式: {mode_decision['mode']}")
            
            # 生成SQL
            if mode_decision['use_exploration_mode']:
                print("使用探索模式...")
                final_sql, success, details = exploration_engine.solve_complex_question(
                    question,
                    "database.db",
                    sql_executor.get_schema()
                )
            else:
                print("使用经验模式...")
                schema = sql_executor.get_schema()
                prompt = f"""
                基于以下数据库schema，为问题生成SQL查询：
                
                问题: {question}
                
                数据库Schema:
                {schema}
                
                请只返回SQL语句，不要包含其他解释。
                """
                
                final_sql = llm_connector(prompt)
                success = True
            
            # 执行SQL
            if success and final_sql:
                print(f"生成的SQL: {final_sql}")
                result, error = sql_executor(final_sql)
                
                if error:
                    print(f"执行错误: {error}")
                else:
                    print("执行结果:")
                    if result:
                        for i, row in enumerate(result[:10]):
                            print(f"  {i+1}. {row}")
                    else:
                        print("  查询结果为空")
            else:
                print("生成SQL失败")
                
        except KeyboardInterrupt:
            print("\n\n感谢使用ILEX-SQL系统！")
            break
        except Exception as e:
            print(f"处理问题时发生错误: {e}")

def evaluate_bird_dataset():
    """评估BIRD数据集"""
    print("=== BIRD数据集评估 ===")
    
    try:
        from bird_evaluator_working import BIRDEvaluator
        
        # 创建评估器
        evaluator = BIRDEvaluator(
            data_dir="LPE-SQL/data",
            db_root="/Users/chuzhibo/Desktop/workspace_sql/data/dev_databases",
            use_local_model=True,
            timeout=30
        )
        
        print("✓ 评估器初始化成功")
        print("开始评估BIRD数据集...")
        print("注意: 这需要vLLM服务正在运行")
        
        # 执行评估
        stats = evaluator.evaluate_dataset(
            split="dev",
            max_questions=None,  # 评估所有问题
            output_file="bird_evaluation_results.json"
        )
        
        print(f"\n✓ BIRD数据集评估完成！")
        print(f"结果已保存到: bird_evaluation_results.json")
        
    except Exception as e:
        print(f"❌ 评估过程中发生错误: {e}")
        print("如果vLLM服务未启动，可以尝试使用模拟版本:")
        print("  python bird_evaluator_mock.py --max-questions 5")
        import traceback
        traceback.print_exc()


def evaluate_bird_dataset_mock():
    """模拟评估BIRD数据集（无需vLLM服务）"""
    print("=== BIRD数据集模拟评估 ===")
    
    try:
        from bird_evaluator_mock import BIRDEvaluator
        
        # 创建评估器
        evaluator = BIRDEvaluator(
            data_dir="LPE-SQL/data",
            db_root="/Users/chuzhibo/Desktop/workspace_sql/data/dev_databases",
            timeout=30
        )
        
        print("✓ 模拟评估器初始化成功")
        print("开始模拟评估BIRD数据集...")
        print("注意: 这是模拟版本，使用模拟的LLM响应")
        
        # 执行评估
        stats = evaluator.evaluate_dataset(
            split="dev",
            max_questions=10,  # 限制为10个问题用于快速测试
            output_file="bird_evaluation_results_mock.json"
        )
        
        print(f"\n✓ BIRD数据集模拟评估完成！")
        print(f"结果已保存到: bird_evaluation_results_mock.json")
        print("您可以查看结果文件了解评估格式")
        
    except Exception as e:
        print(f"❌ 模拟评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_individual_components()
        elif sys.argv[1] == "--interactive":
            interactive_mode()
        elif sys.argv[1] == "--bird":
            evaluate_bird_dataset()
        elif sys.argv[1] == "--bird-mock":
            evaluate_bird_dataset_mock()
        else:
            print("使用方法:")
            print("  python run_example.py              # 运行示例")
            print("  python run_example.py --test       # 测试组件")
            print("  python run_example.py --interactive # 交互模式")
            print("  python run_example.py --bird       # 评估BIRD数据集")
            print("  python run_example.py --bird-mock  # 模拟评估BIRD数据集")
    else:
        main()