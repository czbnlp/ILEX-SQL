"""
问题分解器模块
将复杂问题分解为可管理的子问题
"""

import re
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SubProblem:
    """子问题数据类"""
    id: int
    description: str
    dependencies: List[int]  # 依赖的子问题ID
    priority: int  # 优先级 (1-5)
    estimated_complexity: float  # 预估复杂度 (0-1)
    sql_template: Optional[str] = None  # SQL模板（可选）

class ProblemDecomposer:
    """问题分解器类"""
    
    def __init__(self, config_path: str = "config/ilex_config.yaml"):
        """
        初始化问题分解器
        
        Args:
            config_path: 配置文件路径
        """
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.config = config.get('ilex', {}).get('exploration', {})
        except Exception as e:
            self.config = {}
            self.logger.warning(f"加载配置文件失败: {e}，使用默认配置")
        
        # 分解策略关键词
        self.decomposition_keywords = {
            'sequential': [
                'first', 'then', 'next', 'after', 'before', 'finally',
                'step', 'stage', 'phase', 'followed by', 'subsequently'
            ],
            'conditional': [
                'if', 'when', 'where', 'provided that', 'in case',
                'depending on', 'based on', 'according to'
            ],
            'comparative': [
                'more than', 'less than', 'higher than', 'lower than',
                'greater than', 'better than', 'worse than', 'compared to'
            ],
            'aggregation': [
                'average', 'sum', 'count', 'total', 'maximum', 'minimum',
                'most', 'least', 'highest', 'lowest', 'percentage'
            ]
        }
        
        # 子问题计数器
        self.subproblem_counter = 0
        
    def _analyze_problem_type(self, question: str) -> str:
        """
        分析问题类型
        
        Args:
            question: 问题文本
            
        Returns:
            问题类型 ('sequential', 'conditional', 'comparative', 'general')
        """
        question_lower = question.lower()
        
        # 检查顺序问题
        sequential_keywords = ['first', 'then', 'next', 'after', 'before', 'finally',
                              'step', 'stage', 'phase', 'followed by', 'subsequently']
        if any(keyword in question_lower for keyword in sequential_keywords):
            return 'sequential'
        
        # 检查条件问题
        conditional_keywords = ['if', 'when', 'where', 'provided that', 'in case',
                               'depending on', 'based on', 'according to']
        if any(keyword in question_lower for keyword in conditional_keywords):
            return 'conditional'
        
        # 检查比较问题
        comparative_keywords = ['more than', 'less than', 'higher than', 'lower than',
                              'greater than', 'better than', 'worse than', 'compared to',
                              'compare', 'difference', 'ranking', 'top', 'bottom']
        if any(keyword in question_lower for keyword in comparative_keywords):
            return 'comparative'
        
        # 默认返回通用类型
        return 'general'
    
    def _decompose_sequential(self, question: str, context: str, db_schema: str) -> List[SubProblem]:
        """
        分解顺序性问题
        
        Args:
            question: 原始问题
            context: 执行记忆上下文
            db_schema: 数据库schema
            
        Returns:
            子问题列表
        """
        # 识别顺序步骤
        steps = self._extract_sequential_steps(question)
        
        subproblems = []
        for i, step_description in enumerate(steps):
            subproblem = SubProblem(
                id=self.subproblem_counter,
                description=step_description,
                dependencies=list(range(i)) if i > 0 else [],  # 依赖前面的步骤
                priority=5 - i,  # 越早的步骤优先级越高
                estimated_complexity=self._estimate_complexity(step_description)
            )
            subproblems.append(subproblem)
            self.subproblem_counter += 1
        
        return subproblems
    
    def _decompose_conditional(self, question: str, context: str, db_schema: str) -> List[SubProblem]:
        """
        分解条件性问题
        
        Args:
            question: 原始问题
            context: 执行记忆上下文
            db_schema: 数据库schema
            
        Returns:
            子问题列表
        """
        # 识别条件和主要查询
        condition_part, main_part = self._extract_conditional_parts(question)
        
        subproblems = []
        
        # 第一步：解决条件部分
        if condition_part:
            condition_subproblem = SubProblem(
                id=self.subproblem_counter,
                description=condition_part,
                dependencies=[],
                priority=5,
                estimated_complexity=self._estimate_complexity(condition_part)
            )
            subproblems.append(condition_subproblem)
            self.subproblem_counter += 1
        
        # 第二步：基于条件结果解决主要部分
        if main_part:
            main_subproblem = SubProblem(
                id=self.subproblem_counter,
                description=main_part,
                dependencies=[0] if condition_part else [],
                priority=3,
                estimated_complexity=self._estimate_complexity(main_part)
            )
            subproblems.append(main_subproblem)
            self.subproblem_counter += 1
        
        return subproblems
    
    def decompose_problem(self, 
                         original_question: str, 
                         execution_memory_context: str = "",
                         db_schema: str = "") -> List[SubProblem]:
        """
        分解复杂问题
        
        Args:
            original_question: 原始问题
            execution_memory_context: 执行记忆上下文
            db_schema: 数据库schema信息
            
        Returns:
            子问题列表
        """
        self.logger.info(f"开始分解问题: {original_question[:100]}...")
        
        # 分析问题类型
        problem_type = self._analyze_problem_type(original_question)
        
        # 根据问题类型选择分解策略
        if problem_type == 'sequential':
            subproblems = self._decompose_sequential(original_question, execution_memory_context, db_schema)
        elif problem_type == 'conditional':
            subproblems = self._decompose_conditional(original_question, execution_memory_context, db_schema)
        elif problem_type == 'comparative':
            subproblems = self._decompose_comparative(original_question, execution_memory_context, db_schema)
        else:
            subproblems = self._decompose_general(original_question, execution_memory_context, db_schema)
        
        self.logger.info(f"分解得到 {len(subproblems)} 个子问题")
        for i, subproblem in enumerate(subproblems):
            self.logger.info(f"  子问题 {i+1}: {subproblem.description}")
        
        return subproblems
    
    def _decompose_comparative(self, question: str, context: str, db_schema: str) -> List[SubProblem]:
        """
        分解比较性问题
        
        Args:
            question: 原始问题
            context: 执行记忆上下文
            db_schema: 数据库schema
            
        Returns:
            子问题列表
        """
        # 识别比较的两个部分
        part1, part2, comparison_type = self._extract_comparative_parts(question)
        
        subproblems = []
        
        # 第一步：获取第一个比较对象
        if part1:
            subproblem1 = SubProblem(
                id=self.subproblem_counter,
                description=f"获取: {part1}",
                dependencies=[],
                priority=4,
                estimated_complexity=self._estimate_complexity(part1)
            )
            subproblems.append(subproblem1)
            self.subproblem_counter += 1
        
        # 第二步：获取第二个比较对象
        if part2:
            subproblem2 = SubProblem(
                id=self.subproblem_counter,
                description=f"获取: {part2}",
                dependencies=[],
                priority=4,
                estimated_complexity=self._estimate_complexity(part2)
            )
            subproblems.append(subproblem2)
            self.subproblem_counter += 1
        
        # 第三步：进行比较
        comparison_subproblem = SubProblem(
            id=self.subproblem_counter,
            description=f"比较两个结果: {comparison_type}",
            dependencies=[0, 1] if part1 and part2 else [],
            priority=3,
            estimated_complexity=0.3  # 比较通常相对简单
        )
        subproblems.append(comparison_subproblem)
        self.subproblem_counter += 1
        
        return subproblems
    
    def _decompose_general(self, question: str, context: str, db_schema: str) -> List[SubProblem]:
        """
        通用问题分解
        
        Args:
            question: 原始问题
            context: 执行记忆上下文
            db_schema: 数据库schema
            
        Returns:
            子问题列表
        """
        # 基于问题的一般特征进行分解
        subproblems = []
        
        # 检查是否需要聚合操作
        if self._needs_aggregation(question):
            agg_subproblem = SubProblem(
                id=self.subproblem_counter,
                description=f"计算聚合值: {question}",
                dependencies=[],
                priority=4,
                estimated_complexity=0.6
            )
            subproblems.append(agg_subproblem)
            self.subproblem_counter += 1
        
        # 检查是否需要连接操作
        if self._needs_joins(question):
            join_subproblem = SubProblem(
                id=self.subproblem_counter,
                description=f"执行表连接: {question}",
                dependencies=[],
                priority=3,
                estimated_complexity=0.7
            )
            subproblems.append(join_subproblem)
            self.subproblem_counter += 1
        
        # 如果没有特定模式，创建一个通用子问题
        if not subproblems:
            general_subproblem = SubProblem(
                id=self.subproblem_counter,
                description=f"解决: {question}",
                dependencies=[],
                priority=5,
                estimated_complexity=self._estimate_complexity(question)
            )
            subproblems.append(general_subproblem)
            self.subproblem_counter += 1
        
        return subproblems
    
    def _extract_sequential_steps(self, question: str) -> List[str]:
        """
        提取顺序步骤
        
        Args:
            question: 问题文本
            
        Returns:
            步骤列表
        """
        # 使用正则表达式识别顺序步骤
        step_patterns = [
            r'(?:first|then|next|after|before|finally)\s*,?\s*([^,]+)',
            r'(?:step\s+\d+|stage\s+\d+|phase\s+\d+)\s*:\s*([^,]+)',
            r'(\d+\.?\s*)([^,]+)'
        ]
        
        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, question.lower())
            for match in matches:
                if isinstance(match, tuple):
                    step_text = match[-1].strip()
                else:
                    step_text = match.strip()
                
                if step_text and len(step_text) > 10:  # 过滤太短的文本
                    steps.append(step_text)
        
        # 如果没有找到明确的步骤，尝试基于逗号和连接词分割
        if not steps:
            # 基于连接词分割
            connectors = ['first', 'then', 'next', 'after', 'before', 'finally']
            for connector in connectors:
                if connector in question.lower():
                    parts = question.lower().split(connector)
                    for part in parts[1:]:  # 跳过第一部分（通常不是步骤）
                        part = part.strip()
                        if part and len(part) > 10:
                            steps.append(part)
        
        return steps if steps else [question]  # 如果无法分解，返回原问题
    
    def _extract_conditional_parts(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        """
        提取条件问题的两部分
        
        Args:
            question: 问题文本
            
        Returns:
            (条件部分, 主要部分)
        """
        conditional_patterns = [
            r'if\s+(.+?),\s*(.+)',
            r'when\s+(.+?),\s*(.+)',
            r'where\s+(.+?),\s*(.+)',
            r'provided\s+that\s+(.+?),\s*(.+)',
            r'(.+?)\s+if\s+(.+)',
            r'(.+?)\s+when\s+(.+)',
            r'(.+?)\s+where\s+(.+)'
        ]
        
        for pattern in conditional_patterns:
            match = re.search(pattern, question.lower())
            if match:
                if len(match.groups()) == 2:
                    return match.group(1).strip(), match.group(2).strip()
        
        return None, question
    
    def _extract_comparative_parts(self, question: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        提取比较问题的两部分
        
        Args:
            question: 问题文本
            
        Returns:
            (部分1, 部分2, 比较类型)
        """
        comparative_patterns = [
            (r'(.+?)\s+(?:more than|greater than|higher than)\s+(.+)', 'greater'),
            (r'(.+?)\s+(?:less than|fewer than|lower than)\s+(.+)', 'less'),
            (r'(.+?)\s+(?:better than|worse than)\s+(.+)', 'quality'),
            (r'compare\s+(.+?)\s+and\s+(.+)', 'compare'),
            (r'difference\s+between\s+(.+?)\s+and\s+(.+)', 'difference')
        ]
        
        for pattern, comp_type in comparative_patterns:
            match = re.search(pattern, question.lower())
            if match:
                return match.group(1).strip(), match.group(2).strip(), comp_type
        
        return None, None, 'unknown'
    
    def _needs_aggregation(self, question: str) -> bool:
        """
        判断是否需要聚合操作
        
        Args:
            question: 问题文本
            
        Returns:
            是否需要聚合
        """
        agg_keywords = ['average', 'sum', 'count', 'total', 'maximum', 'minimum',
                       'most', 'least', 'highest', 'lowest', 'percentage']
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in agg_keywords)
    
    def _needs_joins(self, question: str) -> bool:
        """
        判断是否需要连接操作
        
        Args:
            question: 问题文本
            
        Returns:
            是否需要连接
        """
        join_keywords = ['join', 'combine', 'together', 'related', 'associated',
                        'corresponding', 'matching', 'linked', 'relationship']
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in join_keywords)
    
    def _estimate_complexity(self, subproblem: str) -> float:
        """
        估计子问题复杂度
        
        Args:
            subproblem: 子问题描述
            
        Returns:
            复杂度分数 (0-1)
        """
        # 基于简单特征的复杂度估计
        complexity = 0.1  # 基础复杂度
        
        # 长度因子
        words = subproblem.split()
        complexity += min(len(words) / 20.0, 0.3)
        
        # 关键词因子
        high_complexity_keywords = ['join', 'aggregate', 'compare', 'calculate', 'analyze']
        medium_complexity_keywords = ['find', 'get', 'select', 'filter']
        
        subproblem_lower = subproblem.lower()
        
        for keyword in high_complexity_keywords:
            if keyword in subproblem_lower:
                complexity += 0.2
        
        for keyword in medium_complexity_keywords:
            if keyword in subproblem_lower:
                complexity += 0.1
        
        # 限制在0-1之间
        return min(complexity, 1.0)
    
    def get_next_subproblem(self, 
                           original_question: str, 
                           execution_memory_context: str,
                           solved_subproblems: List[int] = None) -> Optional[SubProblem]:
        """
        获取下一个应该解决的子问题
        
        Args:
            original_question: 原始问题
            execution_memory_context: 执行记忆上下文
            solved_subproblems: 已解决的子问题ID列表
            
        Returns:
            下一个子问题，如果没有可解决的则返回None
        """
        if solved_subproblems is None:
            solved_subproblems = []
        
        # 分解问题
        all_subproblems = self.decompose_problem(original_question, execution_memory_context)
        
        # 找出可以解决的子问题（依赖已满足）
        available_subproblems = []
        for subproblem in all_subproblems:
            # 检查所有依赖是否已解决
            dependencies_met = all(dep_id in solved_subproblems for dep_id in subproblem.dependencies)
            
            # 检查是否还未解决
            not_solved = subproblem.id not in solved_subproblems
            
            if dependencies_met and not_solved:
                available_subproblems.append(subproblem)
        
        if not available_subproblems:
            return None
        
        # 按优先级排序，返回优先级最高的
        available_subproblems.sort(key=lambda x: x.priority, reverse=True)
        return available_subproblems[0]
    
    def reset_counter(self):
        """重置子问题计数器"""
        self.subproblem_counter = 0
        self.logger.info("子问题计数器已重置")