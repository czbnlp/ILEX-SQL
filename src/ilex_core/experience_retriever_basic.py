"""
基础版经验检索器 - 无需任何额外依赖
使用简单的关键词匹配和Jaccard相似度
"""

import json
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

class BasicExperienceRetriever:
    """基础经验检索器，使用简单的文本相似度方法"""
    
    def __init__(self, 
                 top_k: int = 4,
                 correct_rate: float = 0.5,
                 knowledge_base_dir: str = "src/knowledge_base",
                 use_init_knowledge_base: bool = True):
        """
        初始化基础经验检索器
        
        Args:
            top_k: 检索示例数量
            correct_rate: 正确示例比例
            knowledge_base_dir: 知识库目录
            use_init_knowledge_base: 是否使用初始知识库
        """
        self.top_k = top_k
        self.correct_rate = correct_rate
        self.knowledge_base_dir = knowledge_base_dir
        self.use_init_knowledge_base = use_init_knowledge_base
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 初始化知识库路径
        self._setup_knowledge_base_paths()
        
        # 加载知识库
        self._load_knowledge_base()
        
        # 统计信息
        self.stats = {
            'retrieval_count': 0,
            'correct_examples_used': 0,
            'mistake_examples_used': 0
        }
    
    def _setup_knowledge_base_paths(self):
        """设置知识库文件路径"""
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        
        # 当前知识库文件
        self.correct_set_path = os.path.join(self.knowledge_base_dir, "correct_set.json")
        self.mistake_set_path = os.path.join(self.knowledge_base_dir, "mistake_set.json")
        
        # 初始知识库文件
        self.init_correct_set_path = os.path.join(self.knowledge_base_dir, "init_correct_set.json")
        self.init_mistake_set_path = os.path.join(self.knowledge_base_dir, "init_mistake_set.json")
    
    def _load_knowledge_base(self):
        """加载知识库"""
        try:
            if self.use_init_knowledge_base:
                # 使用初始知识库
                self.correct_set = self._load_json(self.init_correct_set_path) if os.path.exists(self.init_correct_set_path) else []
                self.mistake_set = self._load_json(self.init_mistake_set_path) if os.path.exists(self.init_mistake_set_path) else []
            else:
                # 使用积累的知识库
                self.correct_set = self._load_json(self.correct_set_path) if os.path.exists(self.correct_set_path) else []
                self.mistake_set = self._load_json(self.mistake_set_path) if os.path.exists(self.mistake_set_path) else []
            
            self.logger.info(f"✓ 加载知识库: {len(self.correct_set)} 个正确示例, {len(self.mistake_set)} 个错误示例")
            
        except Exception as e:
            self.logger.error(f"加载知识库失败: {e}")
            self.correct_set = []
            self.mistake_set = []
    
    def _load_json(self, filepath: str) -> List[Dict]:
        """加载JSON数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"加载JSON失败 {filepath}: {e}")
            return []
    
    def _save_json(self, data: List[Dict], filepath: str):
        """保存JSON数据"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存JSON失败 {filepath}: {e}")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """预处理文本：分词、小写化、去停用词"""
        if not text:
            return []
        
        # 简单的预处理
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # 移除标点符号
        words = text.split()
        
        # 基础停用词列表
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        # 过滤停用词和短词
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return words
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """计算Jaccard相似度"""
        words1 = set(self._preprocess_text(text1))
        words2 = set(self._preprocess_text(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_keyword_overlap(self, query: str, example_text: str) -> float:
        """计算关键词重叠度"""
        query_words = set(self._preprocess_text(query))
        example_words = set(self._preprocess_text(example_text))
        
        if not query_words or not example_words:
            return 0.0
        
        overlap = len(query_words.intersection(example_words))
        return overlap / len(query_words)
    
    def _extract_text_from_example(self, example: Dict, is_correct: bool) -> str:
        """从示例中提取文本用于相似度计算"""
        text_parts = []
        
        # 问题文本（核心）
        if 'question' in example:
            text_parts.append(example['question'])
        
        # 提示信息
        if 'hint' in example and example['hint']:
            text_parts.append(f"hint: {example['hint']}")
        
        # 思考过程
        if 'thought process' in example and example['thought process']:
            text_parts.append(f"thought: {example['thought process']}")
        
        # SQL查询（用于提供语法信息）
        if is_correct:
            if 'sql' in example:
                text_parts.append(f"sql: {example['sql']}")
        else:
            if 'error_sql' in example:
                text_parts.append(f"error_sql: {example['error_sql']}")
            if 'ground_truth_sql' in example:
                text_parts.append(f"correct_sql: {example['ground_truth_sql']}")
        
        # 反思内容（错误示例）
        if not is_correct and 'reflective_cot' in example and example['reflective_cot']:
            text_parts.append(f"reflection: {example['reflective_cot']}")
        
        return " ".join(text_parts)
    
    def retrieve_similar_examples(self, query: str, correct_rate: Optional[float] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        检索相似示例
        
        Args:
            query: 输入问题
            correct_rate: 覆盖正确比例
            
        Returns:
            正确示例列表和错误示例列表
        """
        try:
            if correct_rate is not None:
                current_correct_rate = correct_rate
            else:
                current_correct_rate = self.correct_rate
            
            # 计算要检索的示例数量
            num_correct_to_retrieve = int(current_correct_rate * self.top_k)
            num_mistakes_to_retrieve = self.top_k - num_correct_to_retrieve
            
            # 调整数量如果示例不足
            if len(self.mistake_set) < num_mistakes_to_retrieve:
                num_correct_to_retrieve = self.top_k - len(self.mistake_set)
                num_mistakes_to_retrieve = len(self.mistake_set)
            
            if len(self.correct_set) < num_correct_to_retrieve:
                num_mistakes_to_retrieve = self.top_k - len(self.correct_set)
                num_correct_to_retrieve = len(self.correct_set)
            
            # 计算所有示例的相似度
            correct_scores = []
            for i, example in enumerate(self.correct_set):
                example_text = self._extract_text_from_example(example, is_correct=True)
                similarity = self._calculate_keyword_overlap(query, example_text)
                correct_scores.append((similarity, i, example))
            
            mistake_scores = []
            for i, example in enumerate(self.mistake_set):
                example_text = self._extract_text_from_example(example, is_correct=False)
                similarity = self._calculate_keyword_overlap(query, example_text)
                mistake_scores.append((similarity, i, example))
            
            # 按相似度排序
            correct_scores.sort(reverse=True)
            mistake_scores.sort(reverse=True)
            
            # 选择最相似的示例
            correct_examples = [item[2] for item in correct_scores[:num_correct_to_retrieve]]
            mistake_examples = [item[2] for item in mistake_scores[:num_mistakes_to_retrieve]]
            
            self.logger.info(f"检索到 {len(correct_examples)} 个正确示例, {len(mistake_examples)} 个错误示例")
            
            # 更新统计
            self.stats['retrieval_count'] += 1
            self.stats['correct_examples_used'] += len(correct_examples)
            self.stats['mistake_examples_used'] += len(mistake_examples)
            
            return correct_examples, mistake_examples
            
        except Exception as e:
            self.logger.error(f"检索相似示例失败: {e}")
            return [], []
    
    def add_to_sets(self, question: str, sql: str, correct: bool = True, **kwargs):
        """
        添加新示例到知识库
        
        Args:
            question: 问题文本
            sql: SQL查询
            correct: 是否正确
            **kwargs: 额外元数据
        """
        try:
            if correct:
                # 添加到正确示例集
                new_entry = {
                    'question': question,
                    'sql': sql,
                    'hint': kwargs.get('hint', ''),
                    'thought process': kwargs.get('thought_process', ''),
                    'difficulty': kwargs.get('difficulty', ''),
                    'knowledge': kwargs.get('knowledge', '')
                }
                
                # 避免重复
                if not any(ex['question'] == question for ex in self.correct_set):
                    self.correct_set.append(new_entry)
                    self._save_json(self.correct_set, self.correct_set_path)
                    self.logger.info("✓ 已添加正确示例到知识库")
            else:
                # 添加到错误示例集
                new_entry = {
                    'question': question,
                    'error_sql': kwargs.get('error_sql', sql),
                    'compiler_hint': kwargs.get('compiler_hint', ''),
                    'reflective_cot': kwargs.get('reflective_cot', ''),
                    'ground_truth_sql': kwargs.get('ground_truth_sql', ''),
                    'difficulty': kwargs.get('difficulty', ''),
                    'hint': kwargs.get('hint', ''),
                    'knowledge': kwargs.get('knowledge', '')
                }
                
                # 避免重复
                if not any(ex['question'] == question for ex in self.mistake_set):
                    self.mistake_set.append(new_entry)
                    self._save_json(self.mistake_set, self.mistake_set_path)
                    self.logger.info("✓ 已添加错误示例到知识库")
        
        except Exception as e:
            self.logger.error(f"添加示例到知识库失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'correct_set_size': len(self.correct_set),
            'mistake_set_size': len(self.mistake_set),
            'total_examples': len(self.correct_set) + len(self.mistake_set)
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print("\n=== 基础经验检索器统计 ===")
        print(f"正确示例数: {stats['correct_set_size']}")
        print(f"错误示例数: {stats['mistake_set_size']}")
        print(f"总示例数: {stats['total_examples']}")
        print(f"检索次数: {stats['retrieval_count']}")
        print(f"使用的正确示例: {stats['correct_examples_used']}")
        print(f"使用的错误示例: {stats['mistake_examples_used']}")
        print("=" * 40)


# 测试函数
def test_basic_experience_retriever():
    """测试基础经验检索器"""
    print("=== 测试基础经验检索器 ===")
    
    try:
        # 初始化检索器
        retriever = BasicExperienceRetriever(
            top_k=2,
            correct_rate=0.5,
            use_init_knowledge_base=True
        )
        
        # 测试检索
        query = "Find the average salary of employees"
        correct_examples, mistake_examples = retriever.retrieve_similar_examples(query)
        
        print(f"检索到 {len(correct_examples)} 个正确示例")
        print(f"检索到 {len(mistake_examples)} 个错误示例")
        
        if correct_examples:
            print(f"第一个正确示例问题: {correct_examples[0]['question'][:50]}...")
        
        if mistake_examples:
            print(f"第一个错误示例问题: {mistake_examples[0]['question'][:50]}...")
        
        # 测试添加示例
        retriever.add_to_sets(
            question="What is the highest salary in the company?",
            sql="SELECT MAX(salary) FROM employees",
            correct=True,
            hint="Use MAX function to find highest salary",
            thought_process="Identify the salary column and use MAX aggregation"
        )
        
        retriever.print_statistics()
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_basic_experience_retriever()