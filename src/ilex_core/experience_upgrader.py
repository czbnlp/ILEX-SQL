"""
经验升级器模块
将成功的探索路径转换为可复用经验
"""

import json
import yaml
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from .execution_memory import ExecutionRecord

@dataclass
class ExperienceEntry:
    """经验条目数据类"""
    original_question: str
    problem_type: str
    exploration_path: List[Dict[str, Any]]
    final_sql: str
    success_rate: float
    execution_time: float
    complexity_score: float
    timestamp: float
    confidence: float
    tags: List[str]
    
class ExperienceUpgrader:
    """经验升级器类"""
    
    def __init__(self, config_path: str = "../config/ilex_config.yaml"):
        """
        初始化经验升级器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.config = config.get('ilex', {}).get('experience_upgrade', {})
        except Exception as e:
            self.config = {}
            logging.warning(f"加载配置文件失败: {e}")
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 配置参数
        self.auto_extract_success_patterns = self.config.get('auto_extract_success_patterns', True)
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.8)
        self.max_exploration_history_size = self.config.get('max_exploration_history_size', 1000)
        self.enable_experience_learning = self.config.get('enable_experience_learning', True)
        
        # 经验数据库
        self.experience_database: List[ExperienceEntry] = []
        self.experience_patterns: Dict[str, List[Dict[str, Any]]] = {}
        
        # 统计信息
        self.extraction_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'average_confidence': 0.0,
            'pattern_discovered': 0
        }
    
    def extract_experience_from_exploration(self, 
                                          exploration_history: Dict[str, Any],
                                          original_question: str,
                                          final_sql: str,
                                          success: bool) -> Optional[ExperienceEntry]:
        """
        从探索历史中提取经验
        
        Args:
            exploration_history: 探索历史
            original_question: 原始问题
            final_sql: 最终SQL
            success: 是否成功
            
        Returns:
            经验条目，如果提取失败则返回None
        """
        if not self.enable_experience_learning:
            return None
        
        self.extraction_stats['total_extractions'] += 1
        self.logger.info(f"开始从探索历史提取经验: {original_question[:50]}...")
        
        try:
            # 检查探索是否成功
            if not self._is_successful_exploration(exploration_history, success):
                self.logger.info("探索未达到成功标准，跳过经验提取")
                self.extraction_stats['failed_extractions'] += 1
                return None
            
            # 分析探索路径
            path_analysis = self._analyze_exploration_path(exploration_history)
            
            # 计算置信度
            confidence = self._calculate_confidence(path_analysis, exploration_history)
            
            # 检查置信度是否达到阈值
            if confidence < self.min_confidence_threshold:
                self.logger.info(f"置信度 {confidence:.3f} 低于阈值 {self.min_confidence_threshold}，跳过经验提取")
                self.extraction_stats['failed_extractions'] += 1
                return None
            
            # 生成经验条目
            experience_entry = ExperienceEntry(
                original_question=original_question,
                problem_type=self._classify_problem_type(original_question),
                exploration_path=exploration_history.get('exploration_path', []),
                final_sql=final_sql,
                success_rate=path_analysis['success_rate'],
                execution_time=exploration_history.get('execution_time', 0),
                complexity_score=self._calculate_complexity_score(original_question, exploration_history),
                timestamp=time.time(),
                confidence=confidence,
                tags=self._generate_tags(path_analysis, original_question)
            )
            
            # 添加到经验数据库
            self._add_to_experience_database(experience_entry)
            
            # 提取模式
            self._extract_patterns(experience_entry)
            
            # 更新统计信息
            self.extraction_stats['successful_extractions'] += 1
            self.extraction_stats['average_confidence'] = (
                (self.extraction_stats['average_confidence'] * (self.extraction_stats['successful_extractions'] - 1) + confidence)
                / self.extraction_stats['successful_extractions']
            )
            
            self.logger.info(f"成功提取经验，置信度: {confidence:.3f}")
            return experience_entry
            
        except Exception as e:
            self.logger.error(f"经验提取失败: {e}")
            self.extraction_stats['failed_extractions'] += 1
            return None
    
    def _is_successful_exploration(self, exploration_history: Dict[str, Any], success: bool) -> bool:
        """
        判断探索是否成功
        
        Args:
            exploration_history: 探索历史
            success: 最终成功标志
            
        Returns:
            是否成功
        """
        if not success:
            return False
        
        # 检查探索路径的质量
        exploration_path = exploration_history.get('exploration_path', [])
        if not exploration_path:
            return False
        
        # 计算路径成功率
        successful_steps = sum(1 for step in exploration_path if step.get('success', False))
        total_steps = len(exploration_path)
        
        if total_steps == 0:
            return False
        
        path_success_rate = successful_steps / total_steps
        
        # 要求路径成功率至少达到70%
        return path_success_rate >= 0.7
    
    def _analyze_exploration_path(self, exploration_history: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析探索路径
        
        Args:
            exploration_history: 探索历史
            
        Returns:
            路径分析结果
        """
        exploration_path = exploration_history.get('exploration_path', [])
        
        if not exploration_path:
            return {
                'success_rate': 0.0,
                'average_execution_time': 0.0,
                'path_length': 0,
                'complexity_level': 'unknown'
            }
        
        # 计算成功率
        successful_steps = sum(1 for step in exploration_path if step.get('success', False))
        success_rate = successful_steps / len(exploration_path)
        
        # 计算平均执行时间
        total_time = sum(step.get('execution_time', 0) for step in exploration_path)
        average_execution_time = total_time / len(exploration_path)
        
        # 分析复杂度
        complexity_level = self._determine_complexity_level(exploration_path)
        
        return {
            'success_rate': success_rate,
            'average_execution_time': average_execution_time,
            'path_length': len(exploration_path),
            'complexity_level': complexity_level,
            'has_failures': successful_steps < len(exploration_path),
            'total_execution_time': total_time
        }
    
    def _determine_complexity_level(self, exploration_path: List[Dict[str, Any]]) -> str:
        """
        确定路径复杂度级别
        
        Args:
            exploration_path: 探索路径
            
        Returns:
            复杂度级别
        """
        path_length = len(exploration_path)
        
        if path_length <= 2:
            return 'simple'
        elif path_length <= 4:
            return 'moderate'
        else:
            return 'complex'
    
    def _calculate_confidence(self, path_analysis: Dict[str, Any], exploration_history: Dict[str, Any]) -> float:
        """
        计算经验置信度
        
        Args:
            path_analysis: 路径分析
            exploration_history: 探索历史
            
        Returns:
            置信度分数 (0-1)
        """
        confidence = 0.0
        
        # 基于路径成功率的置信度
        success_rate = path_analysis['success_rate']
        confidence += success_rate * 0.4
        
        # 基于路径长度的置信度（适中的长度通常更可靠）
        path_length = path_analysis['path_length']
        if 2 <= path_length <= 4:
            confidence += 0.3
        elif path_length <= 6:
            confidence += 0.2
        else:
            confidence += 0.1
        
        # 基于执行时间的置信度（过快或过慢都可能不可靠）
        avg_time = path_analysis['average_execution_time']
        if 0.1 <= avg_time <= 10.0:  # 合理的时间范围
            confidence += 0.2
        else:
            confidence += 0.1
        
        # 基于是否有失败的置信度
        if not path_analysis['has_failures']:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _classify_problem_type(self, question: str) -> str:
        """
        分类问题类型
        
        Args:
            question: 问题文本
            
        Returns:
            问题类型
        """
        question_lower = question.lower()
        
        # 顺序性问题
        sequential_keywords = ['first', 'then', 'next', 'after', 'before', 'finally']
        if any(keyword in question_lower for keyword in sequential_keywords):
            return 'sequential'
        
        # 条件性问题
        conditional_keywords = ['if', 'when', 'where', 'provided that']
        if any(keyword in question_lower for keyword in conditional_keywords):
            return 'conditional'
        
        # 比较性问题
        comparative_keywords = ['more than', 'less than', 'higher than', 'compare']
        if any(keyword in question_lower for keyword in comparative_keywords):
            return 'comparative'
        
        # 聚合性问题
        aggregation_keywords = ['average', 'sum', 'count', 'total', 'maximum', 'minimum']
        if any(keyword in question_lower for keyword in aggregation_keywords):
            return 'aggregation'
        
        return 'general'
    
    def _calculate_complexity_score(self, question: str, exploration_history: Dict[str, Any]) -> float:
        """
        计算问题复杂度分数
        
        Args:
            question: 问题文本
            exploration_history: 探索历史
            
        Returns:
            复杂度分数 (0-1)
        """
        complexity = 0.1  # 基础复杂度
        
        # 基于问题长度
        words = question.split()
        complexity += min(len(words) / 30.0, 0.3)
        
        # 基于探索路径长度
        path_length = len(exploration_history.get('exploration_path', []))
        complexity += min(path_length / 10.0, 0.4)
        
        # 基于执行时间
        execution_time = exploration_history.get('execution_time', 0)
        complexity += min(execution_time / 60.0, 0.2)
        
        return min(complexity, 1.0)
    
    def _generate_tags(self, path_analysis: Dict[str, Any], question: str) -> List[str]:
        """
        生成经验标签
        
        Args:
            path_analysis: 路径分析
            question: 问题文本
            
        Returns:
            标签列表
        """
        tags = []
        
        # 基于复杂度级别的标签
        complexity_level = path_analysis['complexity_level']
        tags.append(f'complexity:{complexity_level}')
        
        # 基于成功率的标签
        success_rate = path_analysis['success_rate']
        if success_rate >= 0.9:
            tags.append('high_success')
        elif success_rate >= 0.7:
            tags.append('medium_success')
        else:
            tags.append('low_success')
        
        # 基于问题类型的标签
        problem_type = self._classify_problem_type(question)
        tags.append(f'type:{problem_type}')
        
        # 基于路径长度的标签
        path_length = path_analysis['path_length']
        if path_length <= 2:
            tags.append('short_path')
        elif path_length <= 4:
            tags.append('medium_path')
        else:
            tags.append('long_path')
        
        return tags
    
    def _add_to_experience_database(self, experience_entry: ExperienceEntry):
        """
        添加经验到数据库
        
        Args:
            experience_entry: 经验条目
        """
        self.experience_database.append(experience_entry)
        
        # 维护数据库大小
        if len(self.experience_database) > self.max_exploration_history_size:
            # 保留置信度最高的经验
            self.experience_database.sort(key=lambda x: x.confidence, reverse=True)
            self.experience_database = self.experience_database[:self.max_exploration_history_size]
        
        self.logger.info(f"经验已添加到数据库，当前数据库大小: {len(self.experience_database)}")
    
    def _extract_patterns(self, experience_entry: ExperienceEntry):
        """
        从经验条目中提取模式
        
        Args:
            experience_entry: 经验条目
        """
        # 基于问题类型的模式
        problem_type = experience_entry.problem_type
        if problem_type not in self.experience_patterns:
            self.experience_patterns[problem_type] = []
        
        # 提取路径模式
        path_pattern = {
            'path_length': len(experience_entry.exploration_path),
            'success_rate': experience_entry.success_rate,
            'complexity_level': experience_entry.complexity_score,
            'tags': experience_entry.tags,
            'example_question': experience_entry.original_question[:100] + '...',
            'confidence': experience_entry.confidence
        }
        
        self.experience_patterns[problem_type].append(path_pattern)
        
        # 更新模式发现统计
        self.extraction_stats['pattern_discovered'] += 1
        
        self.logger.info(f"提取到新模式: {problem_type}")
    
    def find_similar_experiences(self, question: str, max_results: int = 5) -> List[ExperienceEntry]:
        """
        查找相似经验
        
        Args:
            question: 问题文本
            max_results: 最大返回结果数
            
        Returns:
            相似经验列表
        """
        if not self.experience_database:
            return []
        
        # 简单的相似度计算（基于问题类型和关键词）
        question_type = self._classify_problem_type(question)
        question_words = set(question.lower().split())
        
        similar_experiences = []
        
        for experience in self.experience_database:
            # 检查问题类型是否匹配
            if experience.problem_type != question_type:
                continue
            
            # 计算词汇重叠度
            experience_words = set(experience.original_question.lower().split())
            intersection = question_words.intersection(experience_words)
            union = question_words.union(experience_words)
            
            if len(union) == 0:
                similarity = 0.0
            else:
                similarity = len(intersection) / len(union)
            
            # 如果相似度足够高，添加到结果
            if similarity > 0.2:  # 相似度阈值
                similar_experiences.append((experience, similarity))
        
        # 按相似度排序并返回前N个结果
        similar_experiences.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, sim in similar_experiences[:max_results]]
    
    def get_experience_statistics(self) -> Dict[str, Any]:
        """
        获取经验统计信息
        
        Returns:
            统计信息字典
        """
        if not self.experience_database:
            return {'total_experiences': 0}
        
        # 按类型分组统计
        type_counts = {}
        for experience in self.experience_database:
            exp_type = experience.problem_type
            type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
        
        # 计算平均置信度
        avg_confidence = sum(exp.confidence for exp in self.experience_database) / len(self.experience_database)
        
        # 计算平均复杂度
        avg_complexity = sum(exp.complexity_score for exp in self.experience_database) / len(self.experience_database)
        
        return {
            'total_experiences': len(self.experience_database),
            'type_distribution': type_counts,
            'average_confidence': avg_confidence,
            'average_complexity': avg_complexity,
            'extraction_stats': self.extraction_stats,
            'pattern_types': list(self.experience_patterns.keys())
        }
    
    def save_experience_database(self, filepath: str):
        """
        保存经验数据库
        
        Args:
            filepath: 文件路径
        """
        data = {
            'experience_database': [asdict(exp) for exp in self.experience_database],
            'experience_patterns': self.experience_patterns,
            'extraction_stats': self.extraction_stats,
            'timestamp': time.time()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"经验数据库已保存到: {filepath}")
        except Exception as e:
            self.logger.error(f"保存经验数据库失败: {e}")
    
    def load_experience_database(self, filepath: str):
        """
        加载经验数据库
        
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载经验数据库
            self.experience_database = [
                ExperienceEntry(**exp) for exp in data.get('experience_database', [])
            ]
            
            # 加载模式
            self.experience_patterns = data.get('experience_patterns', {})
            
            # 加载统计信息
            self.extraction_stats = data.get('extraction_stats', self.extraction_stats)
            
            self.logger.info(f"经验数据库已从文件加载: {filepath}")
        except Exception as e:
            self.logger.error(f"加载经验数据库失败: {e}")
    
    def clear_experience_database(self):
        """清空经验数据库"""
        self.experience_database.clear()
        self.experience_patterns.clear()
        self.extraction_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'average_confidence': 0.0,
            'pattern_discovered': 0
        }
        self.logger.info("经验数据库已清空")


# 测试函数
def test_experience_upgrader():
    """测试经验升级器功能"""
    print("=== 经验升级器测试 ===")
    
    # 创建经验升级器
    upgrader = ExperienceUpgrader()
    
    # 模拟探索历史
    exploration_history = {
        'exploration_path': [
            {'step': 1, 'subquestion': '找出薪水最高的经理', 'success': True, 'execution_time': 0.5},
            {'step': 2, 'subquestion': '找出经理所在的部门', 'success': True, 'execution_time': 0.3},
            {'step': 3, 'subquestion': '找出该部门的员工', 'success': True, 'execution_time': 0.4}
        ],
        'execution_time': 1.2
    }
    
    # 测试经验提取
    original_question = "First, find the manager with the highest salary, then find all employees in the same department."
    final_sql = "SELECT e.name FROM employees e WHERE e.department_id = (SELECT m.department_id FROM managers m WHERE m.salary = (SELECT MAX(salary) FROM managers))"
    
    experience = upgrader.extract_experience_from_exploration(
        exploration_history, original_question, final_sql, success=True
    )
    
    if experience:
        print(f"成功提取经验:")
        print(f"  问题类型: {experience.problem_type}")
        print(f"  置信度: {experience.confidence:.3f}")
        print(f"  标签: {experience.tags}")
        print(f"  复杂度: {experience.complexity_score:.3f}")
    
    # 测试相似经验查找
    similar_question = "First, find the employee with the highest salary, then find all colleagues in the same department."
    similar_experiences = upgrader.find_similar_experiences(similar_question)
    
    print(f"\n相似经验查找结果 ({len(similar_experiences)} 个):")
    for i, exp in enumerate(similar_experiences):
        print(f"  {i+1}. {exp.original_question[:50]}... (置信度: {exp.confidence:.3f})")
    
    # 测试统计信息
    stats = upgrader.get_experience_statistics()
    print(f"\n统计信息: {stats}")


if __name__ == "__main__":
    print(f"\n{'*'*80}")
    print(f"开始执行经验升级器")
    print(f"{'*'*80}\n")
    test_experience_upgrader()