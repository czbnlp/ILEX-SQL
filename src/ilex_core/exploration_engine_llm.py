"""
LLM-based Exploration Engine
Completely refactored to use LLM-based processing instead of rule-based components.
This version incorporates Chain-of-Thought, Few-Shot Learning, and a Self-Correction Loop.
"""

import time
import json
import yaml
import logging
import re  # 添加re模块导入
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sys
import os
from pathlib import Path

# Add project path
# sys.path.append(str(Path(__file__).parent.parent.parent))
# from enhanced_sql_generator import EnhancedSQLGeneratorLPE


@dataclass
class LLMSubProblem:
    """LLM生成的子問題數據類（簡化版）"""
    id: int
    description: str
    dependencies: List[int]
    sql_template: Optional[str] = None
    reasoning: str = ""  # LLM生成的推理過程
    expected_output: str = ""  # LLM預期的輸出描述


class LLMProblemDecomposer:
    """LLM-based問題分解器"""
    
    def __init__(self, llm_connector=None):
        """
        初始化LLM問題分解器
        
        Args:
            llm_connector: LLM連接器
        """
        self.llm_connector = llm_connector
        self.subproblem_counter = 0
        
        # 分解提示模板 (已加入 CoT 和 Few-Shot Learning)
        self.decomposition_prompt_template = """
You are an intelligent SQL query decomposition expert. Given a complex natural language question and database schema information, decompose it into smaller, manageable subproblems that can be solved step by step.

### Task:
1. First, write down your step-by-step thinking process in the "Thought Process" section.
2. Based on your thinking, generate a JSON object that decomposes the original question.

---
### Example 1 (Sequential Question):
#### Database Schema:
{{"employees": ["id", "name"], "sales": ["id", "employee_id", "amount"]}}
#### Original Question:
"找出銷售額最高的員工姓名，並列出他所有的銷售紀錄。"

#### Thought Process:
The question has two parts. First, I need to find the employee with the highest total sales. This involves grouping sales by `employee_id` and finding the one with the maximum sum. Second, once I have the ID of that employee, I need to retrieve their name from the `employees` table and all their individual sales from the `sales` table. So, I need two steps. Step 1 finds the top employee ID. Step 2 uses this ID to get the final details. This is a sequential dependency.

#### Response:
```json
{{
    "question_type": "sequential",
    "decomposition_reasoning": "First find the top employee ID, then use that ID to fetch their name and sales details.",
    "subproblems": [
        {{
            "id": 1,
            "description": "Calculate the total sales for each employee and find the ID of the employee with the highest total sales.",
            "dependencies": [],
            "sql_template": "SELECT employee_id FROM sales GROUP BY employee_id ORDER BY SUM(amount) DESC LIMIT 1;",
            "reasoning": "This step is necessary to identify the target employee.",
            "expected_output": "The ID of the top-selling employee."
        }},
        {{
            "id": 2,
            "description": "Using the top employee's ID, get their name and list all their sales records.",
            "dependencies": [1],
            "sql_template": "SELECT e.name, s.* FROM employees e JOIN sales s ON e.id = s.employee_id WHERE e.id = [Result of Subproblem 1];",
            "reasoning": "This step uses the result from the previous step to retrieve the final required information.",
            "expected_output": "The employee's name and a list of their sales."
        }}
    ],
    "synthesis_approach": "The result of subproblem 2 is the final answer. If a single query is needed, subproblem 1 can be a subquery within the query for subproblem 2."
}}
```

#### Example 2 (Comparative Question):
##### Database Schema:
{{"departments": ["id", "name"], "products": ["id", "department_id"], "sales": ["id", "product_id", "amount"]}}
##### Original Question:
"比較 '電子產品' 和 '圖書' 這兩個部門的總銷售額。"

##### Thought Process:
To compare the sales of two departments, I need to calculate the total sales for each one separately. These two calculations are independent of each other and can be performed in parallel. So, I will create two subproblems: one for 'Electronics' sales and one for 'Books' sales. After both are solved, a final step or synthesis can be used to present the comparison.

##### Response:
```json
{{
    "question_type": "comparative",
    "decomposition_reasoning": "Calculate the total sales for each department independently, then compare the results.",
    "subproblems": [
        {{
            "id": 1,
            "description": "Calculate the total sales for the 'Electronics' department.",
            "dependencies": [],
            "sql_template": "SELECT SUM(s.amount) FROM sales s JOIN products p ON s.product_id = p.id JOIN departments d ON p.department_id = d.id WHERE d.name = 'Electronics';",
            "reasoning": "This is the first part of the comparison.",
            "expected_output": "A single number representing the total sales for Electronics."
        }},
        {{
            "id": 2,
            "description": "Calculate the total sales for the 'Books' department.",
            "dependencies": [],
            "sql_template": "SELECT SUM(s.amount) FROM sales s JOIN products p ON s.product_id = p.id JOIN departments d ON p.department_id = d.id WHERE d.name = 'Books';",
            "reasoning": "This is the second part of the comparison, which can be run in parallel with the first.",
            "expected_output": "A single number representing the total sales for Books."
        }}
    ],
    "synthesis_approach": "Combine the results of subproblem 1 and 2 to present a final comparison table or statement."
}}
```

#### Your Task:
Database Schema:
{schema}
Original Question:
{question}
Execution Context (Previous Results):
{context}
Thought Process:
Response:
"""

        self.subproblem_selection_prompt = """
You are managing the execution of a complex SQL query exploration. Based on the current state, determine the next best subproblem to solve.
### Task:
1. First, analyze the situation in the "Thought Process" section. Consider which subproblems are ready (dependencies met) and which is the most logical next step.
2. Based on your thinking, provide your choice in the specified JSON format.

### Original Question:
{original_question}

### All Subproblems:
{all_subproblems}

### Solved Subproblems:
{solved_subproblems}

### Execution Context (Previous Results):
{context}

### Thought Process:

### Response:
```json
{{
    "next_subproblem_id": 1,
    "reasoning": "Why this subproblem should be solved next", 
    "ready_subproblems": [],
    "completion_status": "in_progress|completed|blocked"
}}
"""
    def decompose_problem(self, question: str, context: str, db_schema: str) -> List[LLMSubProblem]:
        prompt = self.decomposition_prompt_template.format(
            schema=db_schema,
            question=question,
            context=context
        )
        
        try:
            response = self.llm_connector(prompt)
            decomposition_result = self._parse_json_from_response(response)
            
            subproblems = []
            for subproblem_data in decomposition_result.get('subproblems', []):
                subproblem = LLMSubProblem(
                    id=subproblem_data['id'],
                    description=subproblem_data['description'],
                    dependencies=subproblem_data.get('dependencies', []),
                    sql_template=subproblem_data.get('sql_template'),
                    reasoning=subproblem_data.get('reasoning', ''),
                    expected_output=subproblem_data.get('expected_output', '')
                )
                subproblems.append(subproblem)
            return subproblems
            
        except Exception as e:
            logging.error(f"LLM問題分解失敗: {e}")
            return self._fallback_decomposition(question, context, db_schema)

    def get_next_subproblem(self, original_question: str, context: str, solved_subproblems: List[int], all_subproblems: List[LLMSubProblem]) -> Optional[LLMSubProblem]:
        all_subproblems_json = [sp.__dict__ for sp in all_subproblems]
        
        prompt = self.subproblem_selection_prompt.format(
            original_question=original_question,
            all_subproblems=json.dumps(all_subproblems_json, ensure_ascii=False, indent=2),
            solved_subproblems=json.dumps(solved_subproblems),
            context=context
        )
        
        try:
            response = self.llm_connector(prompt)
            selection_result = self._parse_json_from_response(response)
            
            if selection_result.get('completion_status') == 'completed':
                return None
                
            next_id = selection_result.get('next_subproblem_id')
            if next_id is None:
                return self._fallback_subproblem_selection(all_subproblems, solved_subproblems)

            for subproblem in all_subproblems:
                if subproblem.id == next_id:
                    return subproblem
            return None
            
        except Exception as e:
            logging.error(f"LLM子問題選擇失敗: {e}")
            return self._fallback_subproblem_selection(all_subproblems, solved_subproblems)

    def _parse_json_from_response(self, response: str) -> Dict[str, Any]:
        """增强版JSON解析，处理各种LLM响应格式"""
        try:
            # 尝试多种方式提取JSON
            json_str = response
            
            # 1. 尝试提取markdown代码块中的JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            
            # 2. 处理可能的转义字符
            json_str = json_str.replace('\\"', '"').replace("\\'", "'")
            
            # 3. 修复常见JSON格式错误
            json_str = json_str.replace("'", '"')  # 单引号转双引号
            json_str = re.sub(r',\s*}', '}', json_str)  # 修复多余的逗号
            json_str = re.sub(r',\s*]', ']', json_str)  # 修复多余的逗号
            
            # 4. 尝试解析
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # 尝试修复不完整的JSON
                if not json_str.strip().startswith("{"):
                    json_str = "{" + json_str
                if not json_str.strip().endswith("}"):
                    json_str = json_str + "}"
                return json.loads(json_str)
                
        except Exception as e:
            logging.error(f"解析LLM响应失败，尝试最后修复: {e}")
            try:
                # 最后尝试提取第一个{...}之间的内容
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    return json.loads(response[json_start:json_end])
            except:
                logging.error("所有JSON解析尝试均失败")
                return {}

    def _fallback_decomposition(self, question: str, context: str, db_schema: str) -> List[LLMSubProblem]:
        """回退分解方法"""
        return [
            LLMSubProblem(id=1, description=f"Understand requirements for: {question}", dependencies=[]),
            LLMSubProblem(id=2, description=f"Generate SQL for: {question}", dependencies=[1])
        ]

    def _fallback_subproblem_selection(self, all_subproblems: List[LLMSubProblem], solved_subproblems: List[int]) -> Optional[LLMSubProblem]:
        """回退子問題選擇"""
        solved_set = set(solved_subproblems)
        for subproblem in all_subproblems:
            if subproblem.id not in solved_set:
                dependencies_satisfied = all(dep in solved_set for dep in subproblem.dependencies)
                if dependencies_satisfied:
                    return subproblem
        return None

class LLMExecutionMemory:
    """LLM-based執行記憶管理"""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.execution_records = []
        
        # LLM上下文生成提示 (加入CoT)
        self.context_prompt = """
    You are managing the execution memory for a complex SQL query exploration system.
    Task:
    First, think about what has been accomplished, what is still needed, and any important learnings in the "Thought Process" section.
    Then, format the execution history into a clear, concise summary that can be used as context for the next exploration step.
    Execution History:
    {execution_history}
    Thought Process:
    Summary:
    """

        # LLM完成判斷提示 (加入CoT)
        self.completion_prompt = """
    You are evaluating whether a complex SQL query exploration has successfully answered the original question.
    Task:
    First, analyze the history and compare it to the original question in the "Thought Process" section.
    Based on your analysis, provide a structured JSON response to indicate if the task is complete.
    Original Question:
    {original_question}
    Execution History:
    {execution_history}
    Thought Process:
    Response:
    {{
        "is_complete": true,
        "confidence": 0.9,
        "reasoning": "Explanation of completion assessment",
        "remaining_gaps": [],
        "suggestions": []
    }}
    """
    # The rest of LLMExecutionMemory methods like get_context_for_prompt, is_exploration_complete, add, clear_memory etc.
    # would remain the same, as the prompt changes are handled by the LLM and the parsing logic.
    # We will reuse the _parse_json_from_response logic from LLMProblemDecomposer for robustness.

    def _parse_json_from_response(self, response: str) -> Dict[str, Any]:
        """從包含CoT的LLM響應中穩健地提取JSON"""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                if json_str.strip().startswith("```"):
                    json_str = json_str.strip().lstrip("```json").lstrip("```").rstrip("```")
                return json.loads(json_str)
            return {}
        except json.JSONDecodeError:
            return {}

    def get_context_for_prompt(self, llm_connector) -> str:
        if not self.execution_records:
            return "No previous execution history available."
        
        history_text = []
        for i, record in enumerate(self.execution_records):
            status = "✓" if record.success else "✗"
            history_text.append(f"{status} Step {i+1}: {record.subquestion}")
            history_text.append(f"  Result: {record.result}")
            if record.sql: history_text.append(f"  SQL: {record.sql}")
            if record.error: history_text.append(f"  Error: {record.error}")
        
        prompt = self.context_prompt.format(execution_history="\n".join(history_text))
        try:
            response = llm_connector(prompt)
            # The thought process is for the LLM, we return the summary part.
            summary_start = response.find("### Summary:")
            return response[summary_start + len("### Summary:"):].strip() if summary_start != -1 else response
        except Exception as e:
            logging.error(f"LLM上下文生成失敗: {e}")
            return "\n".join(history_text)

    def is_exploration_complete(self, original_question: str, llm_connector) -> Tuple[bool, float]:
        if not self.execution_records:
            return False, 0.0
        
        context = self.get_context_for_prompt(llm_connector)
        prompt = self.completion_prompt.format(original_question=original_question, execution_history=context)
        
        try:
            response = llm_connector(prompt)
            result = self._parse_json_from_response(response)
            return result.get('is_complete', False), result.get('confidence', 0.0)
        except Exception as e:
            logging.error(f"LLM完成判斷失敗: {e}")
            return self._fallback_completion_check()

    def add(self, subquestion: str, result: str, sql: str = None, error: str = None, execution_time: float = 0):
        record = ExecutionRecord(
            subquestion=subquestion,
            result=result,
            sql=sql,
            error=error,
            execution_time=execution_time,
            success=error is None and result is not None
        )
        self.execution_records.append(record)
        if len(self.execution_records) > self.max_size:
            self.execution_records.pop(0)

    def clear_memory(self):
        self.execution_records.clear()

    def _fallback_completion_check(self) -> Tuple[bool, float]:
        if not self.execution_records: return False, 0.0
        success_count = sum(1 for record in self.execution_records[-3:] if record.success)
        if len(self.execution_records) >= 3 and success_count == 3: return True, 0.8
        elif success_count >= 2: return False, 0.6
        else: return False, 0.3

@dataclass
class ExecutionRecord:
    subquestion: str
    result: Any
    sql: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    success: bool = True

class LLMExplorationEngine:
    """LLM-based探索引擎"""
    def __init__(self, 
                config_path: str = "config/ilex_config.yaml",
                llm_connector=None,
                sql_executor=None):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.config = {'max_exploration_steps': 5, 'timeout_per_step': 30, 
                    'max_retries_per_subquestion': 2, 'enable_intermediate_validation': True,
                    'execution_memory_size': 10}
        
        self.execution_memory = LLMExecutionMemory(max_size=self.config.get('execution_memory_size', 10))
        self.problem_decomposer = LLMProblemDecomposer(llm_connector)
        self.llm_connector = llm_connector
        self.sql_executor = sql_executor
        
        self.max_exploration_steps = self.config.get('max_exploration_steps', 5)
        self.max_retries_per_subquestion = self.config.get('max_retries_per_subquestion', 2)
        
        self.exploration_stats = {'total_explorations': 0, 'successful_explorations': 0, 'failed_explorations': 0}
        
        self.sql_generation_prompt = """
    You are an expert SQL query generator. Generate a SQL query to solve the given subproblem.
    Task:
    First, think about how to translate the subproblem into a valid SQL query based on the schema and context in the "Thought Process" section.
    Then, provide ONLY the generated SQL query in the "SQL Query" section.
    Database Schema:
    {schema}
    Subproblem:
    {subproblem_description}
    Execution Context (Previous Results):
    {context}
    Reasoning:
    {reasoning}
    Expected Output:
    {expected_output}
    Thought Process:
    SQL Query:
    """
    # 新增: SQL 自我修正提示
        self.sql_correction_prompt = """
    You are an expert SQL debugger. The previous SQL query you generated failed with an error. Your task is to fix it.
    Task:
    First, analyze the error message and the original SQL in the "Thought Process" section to understand what went wrong.
    Then, provide ONLY the corrected SQL query in the "Fixed SQL Query" section.
    Database Schema:
    {schema}
    Original Subproblem:
    {subproblem_description}
    Faulty SQL Query:```sql
    {faulty_sql}
    ```
    ### Execution Error Message:
    {error_message}

    ### Thought Process:

    ### Fixed SQL Query:
    """

        self.synthesis_prompt = """
    You are synthesizing the final answer from multiple exploration steps.

    ### Task:
    1. First, analyze the execution history and the original question to determine how to combine the results in the "Thought Process" section.
    2. Then, provide a final SQL query or a conclusive text answer based on your analysis.

    ### Original Question:
    {original_question}

    ### Database Schema:
    {schema}

    ### Execution History (All Subproblem Results):
    {execution_history}

    ### Thought Process:

    ### Final Answer (SQL or Text):
    """

    def solve_complex_question(self, original_question: str, db_path: str, db_schema: str = "") -> Tuple[str, bool, Dict[str, Any]]:
        start_time = time.time()
        self.exploration_stats['total_explorations'] += 1
        self.logger.info(f"開始LLM探索模式: {original_question[:100]}...")
        self.execution_memory.clear_memory()
        
        try:
            self.logger.info("步驟 1: 使用LLM分解問題...")
            all_subproblems = self.problem_decomposer.decompose_problem(original_question, "", db_schema)
            self.logger.info(f"問題分解完成，共{len(all_subproblems)}個子問題")
            
            step = 0
            solved_subproblems_ids = []
            
            while step < self.max_exploration_steps:
                self.logger.info(f"--- 探索步驟 {step + 1}/{self.max_exploration_steps} ---")
                
                context = self.execution_memory.get_context_for_prompt(self.llm_connector)
                next_subproblem = self.problem_decomposer.get_next_subproblem(
                    original_question, context, solved_subproblems_ids, all_subproblems
                )
                
                if next_subproblem is None:
                    self.logger.info("所有可解決的子問題均已完成。")
                    break
                
                # *** 主要修改: 解決子問題的邏輯現在包含自我修正循環 ***
                subproblem_result = self._solve_subproblem_with_correction(next_subproblem, db_path, db_schema)
                
                self.execution_memory.add(
                    subquestion=next_subproblem.description,
                    result=subproblem_result.get('result'),
                    sql=subproblem_result.get('sql'),
                    error=subproblem_result.get('error'),
                    execution_time=subproblem_result.get('execution_time', 0)
                )
                
                if subproblem_result['success']:
                    solved_subproblems_ids.append(next_subproblem.id)
                    self.logger.info(f"子問題 {next_subproblem.id} 解決成功。")
                else:
                    self.logger.warning(f"子問題 {next_subproblem.id} 解決失敗，錯誤: {subproblem_result.get('error')}")
                
                is_complete, confidence = self.execution_memory.is_exploration_complete(original_question, self.llm_connector)
                if is_complete and confidence > 0.75:
                    self.logger.info(f"LLM判斷探索完成 (置信度: {confidence:.2f})")
                    break
                step += 1
            
            self.logger.info("步驟 3: 合成最終答案...")
            final_answer, synthesis_success = self._synthesize_final_answer(original_question, db_schema)
            
            execution_time = time.time() - start_time
            if synthesis_success:
                self.exploration_stats['successful_explorations'] += 1
                self.logger.info("LLM探索成功生成最終答案。")
                return final_answer, True, {'execution_time': execution_time, 'final_answer': final_answer}
            else:
                self.exploration_stats['failed_explorations'] += 1
                self.logger.error("LLM探索合成最終答案失敗。")
                return "", False, {'error': 'Failed to synthesize final answer', 'execution_time': execution_time}
                
        except Exception as e:
            self.exploration_stats['failed_explorations'] += 1
            self.logger.error(f"LLM探索模式發生錯誤: {e}", exc_info=True)
            return "", False, {'error': str(e), 'execution_time': time.time() - start_time}
    def _solve_subproblem_with_correction(self, subproblem: LLMSubProblem, db_path: str, db_schema: str) -> Dict[str, Any]:
        """使用LLM解決子問題，並帶有自我修正循環"""
        start_time = time.time()
        context = self.execution_memory.get_context_for_prompt(self.llm_connector)
        
        # 初始SQL生成
        prompt = self.sql_generation_prompt.format(
            schema=db_schema,
            subproblem_description=subproblem.description,
            context=context,
            reasoning=subproblem.reasoning,
            expected_output=subproblem.expected_output
        )
        sql_response = self.llm_connector(prompt)
        sql = self._extract_sql_from_response(sql_response, "SQL Query:")
        
        if not sql:
            return {'success': False, 'error': 'Failed to generate initial SQL', 'execution_time': time.time() - start_time}
        
        # 執行與重試循環
        for attempt in range(self.max_retries_per_subquestion + 1):
            if attempt > 0:
                self.logger.info(f"SQL執行失敗，開始第 {attempt} 次修正...")
            
            if not self.sql_executor:
                 return {'success': True, 'result': f"Generated SQL: {sql}", 'sql': sql, 'execution_time': time.time() - start_time}

            # 修复：使用正确的调用方式，SQLExecutor使用__call__方法而不是execute_query
            result, error = self.sql_executor(sql, db_path)

            if error is None:
                return {
                    'success': True,
                    'result': str(result),
                    'sql': sql,
                    'execution_time': time.time() - start_time
                }
            else:
                error_message = error
                self.logger.warning(f"嘗試 {attempt + 1} 失敗: {error_message}")
                if attempt < self.max_retries_per_subquestion:
                    # 準備修正SQL
                    correction_prompt = self.sql_correction_prompt.format(
                        schema=db_schema,
                        subproblem_description=subproblem.description,
                        faulty_sql=sql,
                        error_message=error_message
                    )
                    corrected_sql_response = self.llm_connector(correction_prompt)
                    corrected_sql = self._extract_sql_from_response(corrected_sql_response, "Fixed SQL Query:")
                    
                    if corrected_sql and corrected_sql != sql:
                        sql = corrected_sql
                    else:
                        self.logger.error("LLM未能生成有效的修正SQL，終止重試。")
                        break # 如果LLM無法提供修正，則跳出循環
        
        # 所有嘗試均失敗
        return {
            'success': False,
            'error': error_message if 'error_message' in locals() else 'Max retries reached without success.',
            'sql': sql,
            'execution_time': time.time() - start_time
        }

    def _synthesize_final_answer(self, original_question: str, db_schema: str) -> Tuple[str, bool]:
        """使用LLM合成最終答案（可以是SQL或文字）"""
        try:
            execution_history = self.execution_memory.get_context_for_prompt(self.llm_connector)
            prompt = self.synthesis_prompt.format(
                original_question=original_question,
                schema=db_schema,
                execution_history=execution_history
            )
            final_response = self.llm_connector(prompt)
            # 提取 "Final Answer" 後的內容
            final_answer_marker = "### Final Answer (SQL or Text):"
            answer_pos = final_response.find(final_answer_marker)
            if answer_pos != -1:
                final_answer = final_response[answer_pos + len(final_answer_marker):].strip()
                return final_answer, True if final_answer else False
            
            return final_response, True # 如果找不到標記，返回整個響應

        except Exception as e:
            logging.error(f"LLM最終答案合成失敗: {e}")
            return "", False

    def _extract_sql_from_response(self, response: str, marker: str) -> str:
        """增强版SQL提取，包含基本语法验证"""
        import re
        
        # 首先尝试从标记后提取
        content_after_marker = response.split(marker)[-1]
        
        # 尝试多种SQL提取模式
        sql_patterns = [
            r"```sql\s*(.*?)\s*```",  # Markdown格式
            r"(SELECT .*?;)",          # 以SELECT开头并以分号结尾
            r"(SELECT .*?(?=```))",    # 以SELECT开头并以```结尾
            r"(SELECT .*?$)",          # 以SELECT开头并以行尾结尾
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, content_after_marker, re.IGNORECASE | re.DOTALL)
            if matches:
                sql = matches[0].strip()
                # 基本SQL语法验证
                sql = self._validate_sql(sql)
                return sql
        
        # 如果没有模式匹配，返回标记后的所有内容作为回退
        sql = content_after_marker.strip().lstrip("```sql").lstrip("```").rstrip("```").strip()
        return self._validate_sql(sql)
        
    def _validate_sql(self, sql: str) -> str:
        """增强版SQL语法验证和修复"""
        # 1. 修复常见的语法错误
        sql = sql.replace("`", "")  # 移除MySQL风格的引号
        sql = re.sub(r"\bAS\s+(\w+)(\s*,\s*)", r"AS \1 \2", sql)  # 修复AS后缺少逗号
        sql = re.sub(r"(\w+)\s*=\s*(\w+)", r"\1 = \2", sql)  # 添加等号周围的空格
        
        # 2. 修复表名和列名中的非法字符
        sql = re.sub(r"([\w]+)\s*\.\s*([\w]+)", r"\1.\2", sql)  # 移除点号周围的空格
        sql = re.sub(r'("[^"]+"|\'[^\']+\')', lambda m: m.group(0).replace(" ", ""), sql)  # 移除引号内空格
        
        # 3. 确保SELECT语句格式正确
        if "SELECT" in sql.upper():
            # 确保FROM子句存在
            if "FROM" not in sql.upper():
                sql = re.sub(r"(SELECT.*?)(;|$)", r"\1 FROM unknown_table \2", sql, flags=re.IGNORECASE)
            
            # 确保WHERE子句格式正确
            sql = re.sub(r"WHERE\s+(\w+)\s*([=<>]+)\s*(\w+)", r"WHERE \1 \2 \3", sql, flags=re.IGNORECASE)
        
        # 4. 确保SQL以分号结尾
        if not sql.endswith(";"):
            sql += ";"
            
        # 5. 记录验证后的SQL
        logging.debug(f"验证后的SQL: {sql}")
        return sql

    def get_statistics(self) -> Dict[str, Any]:
        return self.exploration_stats.copy()