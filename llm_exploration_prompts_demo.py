#!/usr/bin/env python3
"""
演示LLM-based探索模式的核心提示词
展示每个组件如何使用LLM进行智能决策
"""

def show_problem_decomposition_prompt():
    """展示问题分解提示"""
    print("=" * 80)
    print("LLM问题分解提示 (Problem Decomposition Prompt)")
    print("=" * 80)
    
    prompt = """
You are an intelligent SQL query decomposition expert. Given a complex natural language question and database schema information, decompose it into smaller, manageable subproblems that can be solved step by step.

### Database Schema:
{schema}

### Original Question:
{question}

### Execution Context (Previous Results):
{context}

### Task:
1. Analyze the question type (sequential, conditional, comparative, or complex multi-step)
2. Identify the key components and dependencies
3. Decompose into logical subproblems
4. Determine the optimal execution order
5. Estimate complexity for each subproblem

### Requirements:
- Each subproblem should be self-contained and solvable
- Clearly specify dependencies between subproblems
- Provide SQL templates where applicable
- Include reasoning for each decomposition

### Output Format (JSON):
{
    "question_type": "sequential|conditional|comparative|complex",
    "overall_complexity": 0.0-1.0,
    "decomposition_reasoning": "Brief explanation of decomposition approach",
    "subproblems": [
        {
            "id": 1,
            "description": "Clear description of what this subproblem solves",
            "dependencies": [0],
            "priority": 1-5,
            "estimated_complexity": 0.0-1.0,
            "sql_template": "Optional SQL template or approach",
            "reasoning": "Why this subproblem is needed",
            "expected_output": "What result this should produce"
        }
    ],
    "execution_order": [1, 2, 3],
    "synthesis_approach": "How to combine results for final answer"
}

### Response:
"""
    
    print(prompt)
    print("\n" + "=" * 80)
    print("示例输入:")
    print("=" * 80)
    
    example_schema = """
Table: employees
Columns: 
  - id (INTEGER, PRIMARY KEY)
  - name (TEXT, NOT NULL)
  - department (TEXT)
  - salary (INTEGER)
  - hire_date (TEXT)
"""
    
    example_question = "Find departments with more than 2 employees and show their average salary"
    example_context = "No previous results"
    
    print(f"Schema: {example_schema}")
    print(f"Question: {example_question}")
    print(f"Context: {example_context}")
    
    print("\n" + "=" * 80)
    print("示例LLM响应:")
    print("=" * 80)
    
    example_response = """
{
    "question_type": "conditional",
    "overall_complexity": 0.7,
    "decomposition_reasoning": "This question has two parts: 1) Find departments with more than 2 employees, 2) Calculate average salary for those departments. The second part depends on the first.",
    "subproblems": [
        {
            "id": 1,
            "description": "Count employees in each department",
            "dependencies": [],
            "priority": 5,
            "estimated_complexity": 0.4,
            "sql_template": "SELECT department, COUNT(*) as emp_count FROM employees GROUP BY department",
            "reasoning": "Need to count employees per department first to identify which have more than 2",
            "expected_output": "Department names with employee counts"
        },
        {
            "id": 2,
            "description": "Filter departments with more than 2 employees",
            "dependencies": [1],
            "priority": 4,
            "estimated_complexity": 0.3,
            "sql_template": "SELECT department FROM (SELECT department, COUNT(*) as cnt FROM employees GROUP BY department) WHERE cnt > 2",
            "reasoning": "Filter the results to only include departments with more than 2 employees",
            "expected_output": "Department names that have more than 2 employees"
        },
        {
            "id": 3,
            "description": "Calculate average salary for filtered departments",
            "dependencies": [2],
            "priority": 3,
            "estimated_complexity": 0.5,
            "sql_template": "SELECT department, AVG(salary) as avg_salary FROM employees WHERE department IN (SELECT department FROM ...) GROUP BY department",
            "reasoning": "Calculate average salary only for departments identified in step 2",
            "expected_output": "Department names with their average salaries"
        }
    ],
    "execution_order": [1, 2, 3],
    "synthesis_approach": "Use the departments filtered in step 2 to calculate average salaries in step 3"
}
    """
    
    print(example_response)


def show_subproblem_selection_prompt():
    """展示子问题选择提示"""
    print("\n" + "=" * 80)
    print("LLM子问题选择提示 (Subproblem Selection Prompt)")
    print("=" * 80)
    
    prompt = """
You are managing the execution of a complex SQL query exploration. Based on the current state, determine the next best subproblem to solve.

### Original Question:
{original_question}

### All Subproblems:
{all_subproblems}

### Solved Subproblems:
{solved_subproblems}

### Execution Context (Previous Results):
{context}

### Task:
1. Identify which subproblems are ready to be solved (dependencies satisfied)
2. Consider the execution order and priorities
3. Select the most appropriate next subproblem
4. Provide reasoning for the selection

### Requirements:
- Only select subproblems whose dependencies are all solved
- Consider priority and logical flow
- If multiple subproblems are ready, choose the one that provides most value
- If no subproblems can be solved, indicate completion

### Output Format (JSON):
{
    "next_subproblem_id": 1,
    "reasoning": "Why this subproblem should be solved next",
    "ready_subproblems": [1, 2, 3],
    "completion_status": "in_progress|completed|blocked"
}

### Response:
"""
    
    print(prompt)
    print("\n" + "=" * 80)
    print("示例输入:")
    print("=" * 80)
    
    example_original = "Find departments with more than 2 employees and show their average salary"
    example_subproblems = """
[
    {
        "id": 1,
        "description": "Count employees in each department",
        "dependencies": [],
        "priority": 5,
        "estimated_complexity": 0.4
    },
    {
        "id": 2,
        "description": "Filter departments with more than 2 employees",
        "dependencies": [1],
        "priority": 4,
        "estimated_complexity": 0.3
    },
    {
        "id": 3,
        "description": "Calculate average salary for filtered departments",
        "dependencies": [2],
        "priority": 3,
        "estimated_complexity": 0.5
    }
]
    """
    
    example_solved = "[]"
    example_context = "No previous results"
    
    print(f"Original Question: {example_original}")
    print(f"All Subproblems: {example_subproblems}")
    print(f"Solved Subproblems: {example_solved}")
    print(f"Context: {example_context}")


def show_sql_generation_prompt():
    """展示SQL生成提示"""
    print("\n" + "=" * 80)
    print("LLM SQL生成提示 (SQL Generation Prompt)")
    print("=" * 80)
    
    prompt = """
You are an expert SQL query generator working within an exploration system. Generate a SQL query to solve the given subproblem.

### Database Schema:
{schema}

### Subproblem:
{subproblem_description}

### Execution Context (Previous Results):
{context}

### Reasoning:
{reasoning}

### Expected Output:
{expected_output}

### Task:
Generate a syntactically correct SQL query that solves this subproblem.

### Requirements:
- Use proper SQL syntax for the database type
- Reference the correct table and column names from the schema
- Handle any special characters in column names appropriately
- Make the query efficient and readable
- Only return the SQL query, no explanations

### SQL Query:
"""
    
    print(prompt)
    print("\n" + "=" * 80)
    print("示例输入:")
    print("=" * 80)
    
    example_schema = """
Table: employees
Columns: 
  - id (INTEGER, PRIMARY KEY)
  - name (TEXT, NOT NULL)
  - department (TEXT)
  - salary (INTEGER)
  - hire_date (TEXT)
"""
    
    example_subproblem = "Count employees in each department"
    example_context = "No previous results needed for this step"
    example_reasoning = "Need to count employees per department first to identify which have more than 2"
    example_expected = "Department names with employee counts"
    
    print(f"Schema: {example_schema}")
    print(f"Subproblem: {example_subproblem}")
    print(f"Context: {example_context}")
    print(f"Reasoning: {example_reasoning}")
    print(f"Expected Output: {example_expected}")


def show_completion_detection_prompt():
    """展示完成检测提示"""
    print("\n" + "=" * 80)
    print("LLM完成检测提示 (Completion Detection Prompt)")
    print("=" * 80)
    
    prompt = """
You are evaluating whether a complex SQL query exploration has successfully answered the original question.

### Original Question:
{original_question}

### Execution History:
{execution_history}

### Task:
1. Analyze if the original question has been answered
2. Consider the quality and completeness of results
3. Evaluate if further exploration is needed
4. Provide confidence assessment

### Requirements:
- Consider both successful and failed attempts
- Evaluate if the results address the core question
- Assess if additional information is needed
- Be conservative in declaring completion

### Output Format (JSON):
{
    "is_complete": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Explanation of completion assessment",
    "remaining_gaps": ["Any remaining information gaps"],
    "suggestions": ["Suggestions for improvement if needed"]
}

### Response:
"""
    
    print(prompt)
    print("\n" + "=" * 80)
    print("示例输入:")
    print("=" * 80)
    
    example_original = "Find departments with more than 2 employees and show their average salary"
    example_history = """
Step 1: Count employees in each department
Result: Found counts - IT: 3, HR: 2, Sales: 2
SQL: SELECT department, COUNT(*) FROM employees GROUP BY department
Status: Success

Step 2: Filter departments with more than 2 employees  
Result: Identified IT department (3 employees)
SQL: SELECT department FROM (SELECT department, COUNT(*) as cnt FROM employees GROUP BY department) WHERE cnt > 2
Status: Success

Step 3: Calculate average salary for IT department
Result: Average salary: 81,667
SQL: SELECT department, AVG(salary) FROM employees WHERE department = 'IT' GROUP BY department
Status: Success
"""
    
    print(f"Original Question: {example_original}")
    print(f"Execution History: {example_history}")


def show_synthesis_prompt():
    """展示最终合成提示"""
    print("\n" + "=" * 80)
    print("LLM最终合成提示 (Final Synthesis Prompt)")
    print("=" * 80)
    
    prompt = """
You are synthesizing the final SQL query from multiple exploration steps. Combine the results from solved subproblems to answer the original question.

### Original Question:
{original_question}

### Database Schema:
{schema}

### Execution History (All Subproblem Results):
{execution_history}

### Task:
1. Analyze what has been discovered through exploration
2. Identify the key information needed to answer the original question
3. Synthesize a final SQL query that combines all relevant results
4. Ensure the query directly addresses the original question

### Requirements:
- Use information from all successful subproblems
- Create a coherent final query
- Handle any necessary joins, aggregations, or filtering
- Ensure the query is syntactically correct

### Final SQL Query:
"""
    
    print(prompt)
    print("\n" + "=" * 80)
    print("示例输入:")
    print("=" * 80)
    
    example_original = "Find departments with more than 2 employees and show their average salary"
    example_schema = """
Table: employees
Columns: 
  - id (INTEGER, PRIMARY KEY)
  - name (TEXT, NOT NULL)
  - department (TEXT)
  - salary (INTEGER)
  - hire_date (TEXT)
"""
    
    example_history = """
Subproblem 1 - Count employees per department:
Result: IT: 3, HR: 2, Sales: 2
SQL: SELECT department, COUNT(*) FROM employees GROUP BY department
Status: Success

Subproblem 2 - Identify departments with > 2 employees:
Result: IT department identified
SQL: SELECT department FROM (SELECT department, COUNT(*) as cnt FROM employees GROUP BY department) WHERE cnt > 2
Status: Success

Subproblem 3 - Calculate average salary for qualifying departments:
Result: IT department average salary: 81,667
SQL: SELECT department, AVG(salary) FROM employees WHERE department = 'IT' GROUP BY department
Status: Success
"""
    
    print(f"Original Question: {example_original}")
    print(f"Schema: {example_schema}")
    print(f"Execution History: {example_history}")


def show_key_advantages():
    """展示LLM-based方法的关键优势"""
    print("\n" + "=" * 80)
    print("LLM-based探索模式的关键优势")
    print("=" * 80)
    
    advantages = [
        {
            "title": "智能问题理解",
            "description": "LLM能够理解复杂的自然语言问题，而不是依赖固定的关键词匹配",
            "example": "可以理解 '找出工资高于平均水平且入职超过两年的员工' 这样的复杂表述"
        },
        {
            "title": "动态问题分解",
            "description": "根据问题的具体内容和复杂度，动态生成最适合的分解策略",
            "example": "对于比较性问题，会自动生成分步的比较逻辑；对于多条件问题，会合理安排条件评估顺序"
        },
        {
            "title": "上下文感知",
            "description": "在每一步决策中都考虑完整的上下文信息，包括数据库schema和历史执行结果",
            "example": "会根据表结构理解哪些列可以用于连接，哪些索引可能有用"
        },
        {
            "title": "灵活的模式识别",
            "description": "能够识别各种各样的问题模式，而不受限于预定义的关键词列表",
            "example": "可以处理 '找出连续三个月销售额增长超过10%的产品' 这样的时间序列分析"
        },
        {
            "title": "智能完成判断",
            "description": "真正理解原始问题是否已经被回答，而不是简单地计算成功率",
            "example": "会判断结果是否完整覆盖了问题的所有要求，是否有遗漏的条件"
        },
        {
            "title": "自适应学习",
            "description": "能够从执行历史中学习，不断改进分解和决策策略",
            "example": "如果发现某种分解方式经常导致错误，会在后续类似问题中调整策略"
        }
    ]
    
    for i, advantage in enumerate(advantages, 1):
        print(f"\n{i}. {advantage['title']}")
        print(f"   {advantage['description']}")
        print(f"   示例: {advantage['example']}")


def main():
    """主函数"""
    print("LLM-based探索模式提示词演示")
    print("展示如何用LLM替代规则-based组件")
    
    # 展示各个提示词
    show_problem_decomposition_prompt()
    show_subproblem_selection_prompt()
    show_sql_generation_prompt()
    show_completion_detection_prompt()
    show_synthesis_prompt()
    show_key_advantages()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("✓ LLM-based探索模式使用智能提示词替代了固定的规则逻辑")
    print("✓ 每个组件都能够理解上下文、推理和做出智能决策")
    print("✓ 系统能够处理更复杂、更多样化的问题类型")
    print("✓ 通过LLM的自然语言理解能力，大大提高了灵活性")
    print("✓ 保留了规则-based方法的结构化优势，同时增加了智能性")


if __name__ == "__main__":
    main()