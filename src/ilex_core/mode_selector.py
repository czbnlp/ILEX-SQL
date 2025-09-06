"""
模式选择器模块
智能判断问题复杂度并选择合适的处理模式
"""

import re
import yaml
import json
import logging
from typing import Dict, List, Tuple, Any

class ModeSelector:
    """模式选择器类"""
    
    def __init__(self, config_path: str = "../config/ilex_config.yaml"):
        """
        初始化模式选择器
        
        Args:
            config_path: 配置文件路径
        """
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.config = self._load_config(config_path)
        self.complexity_threshold = self.config.get('ilex', {}).get('mode_selection', {}).get('complexity_threshold', 0.7)
        self.enable_exploration = self.config.get('ilex', {}).get('mode_selection', {}).get('enable_exploration', True)
        self.enable_complexity_analysis = self.config.get('ilex', {}).get('mode_selection', {}).get('enable_complexity_analysis', True)
        
        # 复杂度分析关键词
        self.complexity_keywords = {
            'multi_step': [
                'first', 'then', 'next', 'after', 'before', 'finally',
                'step', 'stage', 'phase', 'followed by'
            ],
            'aggregation': [
                'average', 'sum', 'count', 'total', 'maximum', 'minimum',
                'most', 'least', 'highest', 'lowest', 'percentage'
            ],
            'joins': [
                'join', 'combine', 'together', 'related', 'associated',
                'corresponding', 'matching', 'linked'
            ],
            'subqueries': [
                'where', 'having', 'that', 'which', 'who', 'whose',
                'such that', 'for which', 'in which'
            ]
        }
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"加载配置文件失败: {e}，使用默认配置")
            return {}
    
    def assess_complexity(self, question: str, db_schema: str = "") -> float:
        """
        评估问题复杂度
        
        Args:
            question: 自然语言问题
            db_schema: 数据库schema信息
            
        Returns:
            复杂度分数 (0-1)
        """
        if not self.enable_complexity_analysis:
            return 0.5  # 默认中等复杂度
            
        # 分析问题特征
        indicators = self.get_complexity_indicators(question)
        
        # 计算复杂度分数
        complexity_score = self._calculate_complexity_score(indicators)
        
        self.logger.info(f"问题复杂度评估: {question[:50]}... -> {complexity_score:.3f}")
        self.logger.info(f"复杂度指标: {indicators}")
        
        return complexity_score
    
    def get_complexity_indicators(self, question: str) -> Dict[str, Any]:
        """
        获取复杂度指标
        
        Args:
            question: 自然语言问题
            
        Returns:
            复杂度指标字典
        """
        question_lower = question.lower()
        
        indicators = {
            'requires_multi_step': False,
            'has_aggregation': False,
            'needs_joins': False,
            'has_subqueries': False,
            'question_length': len(question.split()),
            'has_comparative': False,
            'has_temporal': False,
            'has_conditional': False
        }
        
        # 检查多步操作关键词
        for keyword in self.complexity_keywords['multi_step']:
            if keyword in question_lower:
                indicators['requires_multi_step'] = True
                break
                
        # 检查聚合函数关键词
        for keyword in self.complexity_keywords['aggregation']:
            if keyword in question_lower:
                indicators['has_aggregation'] = True
                break
                
        # 检查连接操作关键词
        for keyword in self.complexity_keywords['joins']:
            if keyword in question_lower:
                indicators['needs_joins'] = True
                break
                
        # 检查子查询关键词
        for keyword in self.complexity_keywords['subqueries']:
            if keyword in question_lower:
                indicators['has_subqueries'] = True
                break
        
        # 检查比较级
        comparative_patterns = [
            r'more\s+than', r'less\s+than', r'higher\s+than', r'lower\s+than',
            r'greater\s+than', r'fewer\s+than', r'better\s+than', r'worse\s+than'
        ]
        for pattern in comparative_patterns:
            if re.search(pattern, question_lower):
                indicators['has_comparative'] = True
                break
        
        # 检查时间相关
        temporal_patterns = [
            r'before\s+\d+', r'after\s+\d+', r'during\s+\d+',
            r'in\s+\d{4}', r'until\s+\d+', r'since\s+\d+'
        ]
        for pattern in temporal_patterns:
            if re.search(pattern, question_lower):
                indicators['has_temporal'] = True
                break
        
        # 检查条件句
        conditional_patterns = [
            r'if\s+\w+', r'when\s+\w+', r'where\s+\w+',
            r'only\s+if', r'provided\s+that', r'such\s+that'
        ]
        for pattern in conditional_patterns:
            if re.search(pattern, question_lower):
                indicators['has_conditional'] = True
                break
        
        return indicators
    
    def _calculate_complexity_score(self, indicators: Dict[str, Any]) -> float:
        """
        基于指标计算复杂度分数
        
        Args:
            indicators: 复杂度指标
            
        Returns:
            复杂度分数 (0-1)
        """
        score = 0.0
        
        # 基础分数
        base_score = 0.1
        
        # 各指标权重
        weights = {
            'requires_multi_step': 0.3,
            'has_aggregation': 0.15,
            'needs_joins': 0.2,
            'has_subqueries': 0.2,
            'question_length': 0.05,
            'has_comparative': 0.05,
            'has_temporal': 0.025,
            'has_conditional': 0.025
        }
        
        # 计算加权分数
        if indicators['requires_multi_step']:
            score += weights['requires_multi_step']
        if indicators['has_aggregation']:
            score += weights['has_aggregation']
        if indicators['needs_joins']:
            score += weights['needs_joins']
        if indicators['has_subqueries']:
            score += weights['has_subqueries']
        if indicators['has_comparative']:
            score += weights['has_comparative']
        if indicators['has_temporal']:
            score += weights['has_temporal']
        if indicators['has_conditional']:
            score += weights['has_conditional']
        
        # 问题长度因子
        length_factor = min(indicators['question_length'] / 50.0, 1.0) * weights['question_length']
        score += length_factor
        
        # 加上基础分数并限制在0-1之间
        total_score = min(base_score + score, 1.0)
        
        return total_score
    
    def should_use_exploration(self, complexity_score: float) -> bool:
        """
        判断是否应该使用探索模式
        
        Args:
            complexity_score: 复杂度分数
            
        Returns:
            是否使用探索模式
        """
        if not self.enable_exploration:
            return False
            
        return complexity_score >= self.complexity_threshold
    
    def get_mode_decision(self, question: str, db_schema: str = "") -> Dict[str, Any]:
        """
        获取模式决策结果
        
        Args:
            question: 自然语言问题
            db_schema: 数据库schema信息
            
        Returns:
            模式决策结果
        """
        complexity_score = self.assess_complexity(question, db_schema)
        indicators = self.get_complexity_indicators(question)
        use_exploration = self.should_use_exploration(complexity_score)
        
        decision = {
            'complexity_score': complexity_score,
            'indicators': indicators,
            'use_exploration_mode': use_exploration,
            'mode': 'exploration' if use_exploration else 'experience',
            'confidence': abs(complexity_score - self.complexity_threshold),
            'reasoning': self._generate_reasoning(complexity_score, indicators, use_exploration)
        }
        
        self.logger.info(f"模式决策: {decision}")
        
        return decision
    
    def _generate_reasoning(self, complexity_score: float, indicators: Dict[str, Any], use_exploration: bool) -> str:
        """
        生成决策推理说明
        
        Args:
            complexity_score: 复杂度分数
            indicators: 复杂度指标
            use_exploration: 是否使用探索模式
            
        Returns:
            推理说明
        """
        reasons = []
        
        if indicators['requires_multi_step']:
            reasons.append("问题需要多步操作")
        if indicators['has_aggregation']:
            reasons.append("问题包含聚合操作")
        if indicators['needs_joins']:
            reasons.append("问题需要多表连接")
        if indicators['has_subqueries']:
            reasons.append("问题可能需要子查询")
        
        if use_exploration:
            threshold_reason = f"复杂度分数 {complexity_score:.3f} 超过阈值 {self.complexity_threshold}"
            reasons.append(threshold_reason)
            mode_reason = "选择探索模式以进行分步推理"
        else:
            threshold_reason = f"复杂度分数 {complexity_score:.3f} 低于阈值 {self.complexity_threshold}"
            reasons.append(threshold_reason)
            mode_reason = "选择经验模式以快速生成答案"
        
        reasons.append(mode_reason)
        
        return "；".join(reasons) + "。"
    
    def update_threshold(self, new_threshold: float):
        """
        更新复杂度阈值
        
        Args:
            new_threshold: 新的复杂度阈值
        """
        if 0 <= new_threshold <= 1:
            self.complexity_threshold = new_threshold
            self.logger.info(f"复杂度阈值已更新为: {new_threshold}")
        else:
            self.logger.warning(f"无效的复杂度阈值: {new_threshold}，必须在0-1之间")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取模式选择器统计信息
        
        Returns:
            统计信息
        """
        return {
            'complexity_threshold': self.complexity_threshold,
            'enable_exploration': self.enable_exploration,
            'enable_complexity_analysis': self.enable_complexity_analysis,
            'complexity_keywords_count': {
                key: len(value) for key, value in self.complexity_keywords.items()
            }
        }


# 测试函数
def test_mode_selector():
    """测试模式选择器功能"""
    selector = ModeSelector()
    
    # 测试问题
    test_questions = [
        "What is the average salary of employees?",
        "Find the name of the employee who has the highest salary in the sales department.",
        "First, find the manager with the highest salary, then find all employees in the same department.",
        "List all customers who have placed more than 10 orders and whose total order amount is greater than $1000.",
        "Show me the departments where the average salary is higher than the company average."
    ]
    
    print("=== 模式选择器测试 ===")
    for question in test_questions:
        decision = selector.get_mode_decision(question)
        print(f"问题: {question}")
        print(f"复杂度: {decision['complexity_score']:.3f}")
        print(f"模式: {decision['mode']}")
        print(f"推理: {decision['reasoning']}")
        print("-" * 50)


if __name__ == "__main__":
    test_mode_selector()