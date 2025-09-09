"""
简化版经验检索器 - 无需sentence-transformers和faiss
使用简单的关键词匹配和余弦相似度计算
"""

import json
import numpy as np
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleExperienceRetriever:
    """简化版经验检索器，使用TF-IDF和余弦相似度"""
    
    def __init__(self, 
                 top_k: int = 4,
                 correct_rate: float = 0.5,
                 knowledge_base_dir: str = "src/knowledge_base",
                 use_init_knowledge_base: bool = True):
        """
        初始化简化经验检索器
        
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
        
        # 初始化TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
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
            
            # 预计算向量（如果数据存在）
            self._precompute_vectors()
            
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
    
    def _precompute_vectors(self):
        """预计算知识库中所有示例的向量表示"""
        try:
            # 准备文本数据
            all_texts = []
            all_examples = []
            example_types = []  # 'correct' or 'mistake'
            
            # 添加正确示例
            for example in self.correct_set:
                text = self._extract_text_from_example(example, is_correct=True)
                all_texts.append(text)
                all_examples.append(example)
                example_types.append('correct')
            
            # 添加错误示例
            for example in self.mistake_set:
                text = self._extract_text_from_example(example, is_correct=False)
                all_texts.append(text)
                all_examples.append(example)
                example_types.append('mistake')
            
            if all_texts:
                # 拟合向量化器并转换所有文本
                self.all_vectors = self.vectorizer.fit_transform(all_texts)
                self.all_examples = all_examples
                self.example_types = example_types
                self.logger.info(f"✓ 预计算了 {len(all_texts)} 个示例的向量表示")
            else:
                self.all_vectors = None
                self.all_examples = []
                self.example_types = []
                
        except Exception as e:
            self.logger.error(f"预计算向量失败: {e}")
            self.all_vectors = None
            self.all_examples = []
            self.example_types = []
    
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
            correct_count = sum(1 for t in self.example_types if t == 'correct')
            mistake_count = sum(1 for t in self.example_types if t == 'mistake')
            
            if mistake_count < num_mistakes_to_retrieve:
                num_correct_to_retrieve = self.top_k - mistake_count
                num_mistakes_to_retrieve = mistake_count
            
            if correct_count < num_correct_to_retrieve:
                num_mistakes_to_retrieve = self.top_k - correct_count
                num_correct_to_retrieve = correct_count
            
            if self.all_vectors is None or len(self.all_examples) == 0:
                return [], []
            
            # 转换查询为向量
            query_vector = self.vectorizer.transform([query])
            
            # 计算相似度
            similarities = cosine_similarity(query_vector, self.all_vectors)[0]
            
            # 获取排序索引
            sorted_indices = np.argsort(similarities)[::-1]
            
            # 分别收集正确和错误示例
            correct_examples = []
            mistake_examples = []
            
            for idx in sorted_indices:
                example_type = self.example_types[idx]
                example = self.all_examples[idx]
                
                if example_type == 'correct' and len(correct_examples) < num_correct_to_retrieve:
                    correct_examples.append(example)
                elif example_type == 'mistake' and len(mistake_examples) < num_mistakes_to_retrieve:
                    mistake_examples.append(example)
                
                # 如果都收集够了，就停止
                if len(correct_examples) >= num_correct_to_retrieve and len(mistake_examples) >= num_mistakes_to_retrieve:
                    break
            
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
                    
                    # 重新计算向量
                    self._precompute_vectors()
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
                    
                    # 重新计算向量
                    self._precompute_vectors()
        
        except Exception as e:
            self.logger.error(f"添加示例到知识库失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'correct_set_size': len(self.correct_set),
            'mistake_set_size': len(self.mistake_set),
            'total_examples': len(self.correct_set) + len(self.mistake_set),
            'vectorizer_fitted': hasattr(self, 'all_vectors') and self.all_vectors is not None
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print("\n=== 简化经验检索器统计 ===")
        print(f"正确示例数: {stats['correct_set_size']}")
        print(f"错误示例数: {stats['mistake_set_size']}")
        print(f"总示例数: {stats['total_examples']}")
        print(f"检索次数: {stats['retrieval_count']}")
        print(f"使用的正确示例: {stats['correct_examples_used']}")
        print(f"使用的错误示例: {stats['mistake_examples_used']}")
        print(f"向量器已训练: {stats['vectorizer_fitted']}")
        print("=" * 40)


# 测试函数
def test_simple_experience_retriever():
    """测试简化经验检索器"""
    print("=== 测试简化经验检索器 ===")
    
    try:
        # 初始化检索器
        retriever = SimpleExperienceRetriever(
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
    test_simple_experience_retriever()