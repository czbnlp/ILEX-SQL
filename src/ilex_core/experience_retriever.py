"""
Experience-based SQL generation retriever
Based on LPE-SQL implementation for retrieving relevant examples from knowledge base
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import os
import logging

class ExperienceRetriever:
    """Experience retriever for Text-to-SQL using embedding-based similarity search"""
    
    def __init__(self, 
                 top_k: int = 4,
                 correct_rate: float = 0.5,
                 embedding_model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu",
                 knowledge_base_dir: str = "src/knowledge_base",
                 use_init_knowledge_base: bool = True):
        """
        Initialize the experience retriever
        
        Args:
            top_k: Number of examples to retrieve
            correct_rate: Ratio of correct examples vs mistake examples (0.0-1.0)
            embedding_model_path: Path to sentence transformer model
            device: Device to run embeddings on
            knowledge_base_dir: Directory for knowledge base files
            use_init_knowledge_base: Whether to use initial knowledge base
        """
        self.top_k = top_k
        self.correct_rate = correct_rate
        self.device = device
        self.knowledge_base_dir = knowledge_base_dir
        self.use_init_knowledge_base = use_init_knowledge_base
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model_path, device=device)
            logging.info(f"✓ Loaded embedding model: {embedding_model_path}")
        except Exception as e:
            logging.warning(f"Failed to load embedding model: {e}, using fallback")
            self.embedding_model = None
        
        # Initialize knowledge base paths
        self._setup_knowledge_base_paths()
        
        # Load knowledge base
        self._load_knowledge_base()
        
        # Statistics
        self.stats = {
            'retrieval_count': 0,
            'correct_examples_used': 0,
            'mistake_examples_used': 0
        }
    
    def _setup_knowledge_base_paths(self):
        """Setup paths for knowledge base files"""
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        
        # Current knowledge base files
        self.correct_set_path = os.path.join(self.knowledge_base_dir, "correct_set.json")
        self.mistake_set_path = os.path.join(self.knowledge_base_dir, "mistake_set.json")
        self.correct_vectors_path = os.path.join(self.knowledge_base_dir, "correct_vectors.npy")
        self.mistake_vectors_path = os.path.join(self.knowledge_base_dir, "mistake_vectors.npy")
        
        # Initial knowledge base files
        self.init_correct_set_path = os.path.join(self.knowledge_base_dir, "init_correct_set.json")
        self.init_mistake_set_path = os.path.join(self.knowledge_base_dir, "init_mistake_set.json")
    
    def _load_knowledge_base(self):
        """Load knowledge base from files"""
        try:
            if self.use_init_knowledge_base:
                # Use initial knowledge base
                self.correct_set = self._load_json(self.init_correct_set_path) if os.path.exists(self.init_correct_set_path) else []
                self.mistake_set = self._load_json(self.init_mistake_set_path) if os.path.exists(self.init_mistake_set_path) else []
            else:
                # Use accumulated knowledge base
                self.correct_set = self._load_json(self.correct_set_path) if os.path.exists(self.correct_set_path) else []
                self.mistake_set = self._load_json(self.mistake_set_path) if os.path.exists(self.mistake_set_path) else []
            
            logging.info(f"✓ Loaded knowledge base: {len(self.correct_set)} correct examples, {len(self.mistake_set)} mistake examples")
            
            # Load or generate embeddings
            self.correct_vectors = self._load_or_generate_vectors(self.correct_set, self.correct_vectors_path) if self.correct_set else None
            self.mistake_vectors = self._load_or_generate_vectors(self.mistake_set, self.mistake_vectors_path) if self.mistake_set else None
            
        except Exception as e:
            logging.error(f"Error loading knowledge base: {e}")
            self.correct_set = []
            self.mistake_set = []
            self.correct_vectors = None
            self.mistake_vectors = None
    
    def _load_json(self, filepath: str) -> List[Dict]:
        """Load JSON data from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load JSON from {filepath}: {e}")
            return []
    
    def _save_json(self, data: List[Dict], filepath: str):
        """Save JSON data to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save JSON to {filepath}: {e}")
    
    def _load_or_generate_vectors(self, dataset: List[Dict], vectors_path: str) -> Optional[np.ndarray]:
        """Load existing vectors or generate new ones"""
        try:
            if os.path.exists(vectors_path):
                # Load existing vectors
                vectors = np.load(vectors_path)
                logging.info(f"✓ Loaded existing vectors from {vectors_path}")
                return vectors
            elif dataset and self.embedding_model:
                # Generate new vectors
                texts = [entry['question'] for entry in dataset]
                vectors = self.embedding_model.encode(texts)
                np.save(vectors_path, vectors)
                logging.info(f"✓ Generated and saved vectors to {vectors_path}")
                return vectors
            else:
                return None
        except Exception as e:
            logging.error(f"Error loading/generating vectors: {e}")
            return None
    
    def retrieve_similar_examples(self, query: str, correct_rate: Optional[float] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Retrieve similar examples from knowledge base
        
        Args:
            query: Input question
            correct_rate: Override correct rate for this retrieval
            
        Returns:
            Tuple of (correct_examples, mistake_examples)
        """
        if not self.embedding_model:
            return [], []
        
        if correct_rate is not None:
            current_correct_rate = correct_rate
        else:
            current_correct_rate = self.correct_rate
        
        # Calculate number of examples to retrieve
        num_correct_to_retrieve = int(current_correct_rate * self.top_k)
        num_mistakes_to_retrieve = self.top_k - num_correct_to_retrieve
        
        # Adjust if not enough examples available
        if self.mistake_set and len(self.mistake_set) < num_mistakes_to_retrieve:
            num_correct_to_retrieve = self.top_k - len(self.mistake_set)
            num_mistakes_to_retrieve = len(self.mistake_set)
        
        # Encode query
        query_vector = self.embedding_model.encode([query])
        
        # Retrieve examples
        correct_examples = self._retrieve_from_vectors(
            query_vector, self.correct_set, self.correct_vectors, num_correct_to_retrieve
        ) if self.correct_vectors is not None and num_correct_to_retrieve > 0 else []
        
        mistake_examples = self._retrieve_from_vectors(
            query_vector, self.mistake_set, self.mistake_vectors, num_mistakes_to_retrieve
        ) if self.mistake_vectors is not None and num_mistakes_to_retrieve > 0 else []
        
        logging.info(f"Retrieved {len(correct_examples)} correct examples, {len(mistake_examples)} mistake examples")
        
        # Update statistics
        self.stats['retrieval_count'] += 1
        self.stats['correct_examples_used'] += len(correct_examples)
        self.stats['mistake_examples_used'] += len(mistake_examples)
        
        return correct_examples, mistake_examples
    
    def _retrieve_from_vectors(self, query_vector: np.ndarray, dataset: List[Dict], vectors: np.ndarray, num_k: int) -> List[Dict]:
        """Retrieve most similar examples using FAISS"""
        try:
            if vectors is None or len(dataset) == 0 or num_k <= 0:
                return []
            
            # Create FAISS index
            vector_dimension = vectors.shape[1]
            index = faiss.IndexFlatL2(vector_dimension)
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors)
            index.add(vectors)
            
            # Normalize query vector
            faiss.normalize_L2(query_vector)
            
            # Search for similar examples
            distances, indices = index.search(query_vector, k=min(num_k, len(dataset)))
            
            # Return similar examples
            similar_examples = [dataset[i] for i in indices[0]]
            
            return similar_examples
            
        except Exception as e:
            logging.error(f"Error in vector retrieval: {e}")
            return []
    
    def add_to_sets(self, question: str, sql: str, correct: bool = True, **kwargs):
        """
        Add new example to knowledge base
        
        Args:
            question: Question text
            sql: SQL query
            correct: Whether this is a correct example
            **kwargs: Additional metadata (hint, thought_process, difficulty, etc.)
        """
        try:
            new_entry = {
                'question': question,
                'sql': sql,
                'hint': kwargs.get('hint', ''),
                'thought process': kwargs.get('thought_process', ''),
                'difficulty': kwargs.get('difficulty', ''),
                'knowledge': kwargs.get('knowledge', '')
            }
            
            if correct:
                # Add to correct set
                if new_entry not in self.correct_set:
                    self.correct_set.append(new_entry)
                    self._save_json(self.correct_set, self.correct_set_path)
                    
                    # Update vectors if embedding model is available
                    if self.embedding_model:
                        question_vector = self.embedding_model.encode([question])
                        if self.correct_vectors is not None:
                            self.correct_vectors = np.vstack([self.correct_vectors, question_vector[0]])
                        else:
                            self.correct_vectors = question_vector
                        np.save(self.correct_vectors_path, self.correct_vectors)
                    
                    logging.info(f"✓ Added correct example to knowledge base")
            else:
                # Add to mistake set with additional error information
                new_entry.update({
                    'error_sql': kwargs.get('error_sql', ''),
                    'compiler_hint': kwargs.get('compiler_hint', ''),
                    'reflective_cot': kwargs.get('reflective_cot', ''),
                    'ground_truth_sql': kwargs.get('ground_truth_sql', '')
                })
                
                if new_entry not in self.mistake_set:
                    self.mistake_set.append(new_entry)
                    self._save_json(self.mistake_set, self.mistake_set_path)
                    
                    # Update vectors if embedding model is available
                    if self.embedding_model:
                        question_vector = self.embedding_model.encode([question])
                        if self.mistake_vectors is not None:
                            self.mistake_vectors = np.vstack([self.mistake_vectors, question_vector[0]])
                        else:
                            self.mistake_vectors = question_vector
                        np.save(self.mistake_vectors_path, self.mistake_vectors)
                    
                    logging.info(f"✓ Added mistake example to knowledge base")
            
        except Exception as e:
            logging.error(f"Error adding to knowledge base: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            **self.stats,
            'correct_set_size': len(self.correct_set),
            'mistake_set_size': len(self.mistake_set),
            'embedding_model_loaded': self.embedding_model is not None
        }
    
    def print_statistics(self):
        """Print retrieval statistics"""
        stats = self.get_statistics()
        print("\n=== Experience Retriever Statistics ===")
        print(f"Correct set size: {stats['correct_set_size']}")
        print(f"Mistake set size: {stats['mistake_set_size']}")
        print(f"Total retrievals: {stats['retrieval_count']}")
        print(f"Correct examples used: {stats['correct_examples_used']}")
        print(f"Mistake examples used: {stats['mistake_examples_used']}")
        print(f"Embedding model loaded: {stats['embedding_model_loaded']}")
        print("=" * 40)


# Test function
def test_experience_retriever():
    """Test the experience retriever"""
    print("=== Testing Experience Retriever ===")
    
    try:
        # Initialize retriever
        retriever = ExperienceRetriever(
            top_k=2,
            correct_rate=0.5,
            use_init_knowledge_base=True
        )
        
        # Test retrieval
        query = "Find the average salary of employees in the IT department"
        correct_examples, mistake_examples = retriever.retrieve_similar_examples(query)
        
        print(f"Retrieved {len(correct_examples)} correct examples")
        print(f"Retrieved {len(mistake_examples)} mistake examples")
        
        # Test adding examples
        retriever.add_to_sets(
            question="What is the highest salary in the company?",
            sql="SELECT MAX(salary) FROM employees",
            correct=True,
            hint="Use MAX function to find highest salary",
            thought_process="Identify the salary column and use MAX aggregation"
        )
        
        retriever.print_statistics()
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print(f"\n{'*'*80}")
    print(f"开始执行经验检索器")
    print(f"{'*'*80}\n")
    test_experience_retriever()