#!/usr/bin/env python3
"""
Enhanced SQL Generator - LPE-SQL Style Implementation
Completely refactored to adopt LPE-SQL's experience storage and retrieval system
"""

import re
import json
import sqlite3
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
from llm_connector_local import LocalLLMConnector
from master_sql_postprocessor import MasterSQLPostProcessor


class TextToSQLRetriever:
    """Experience retriever adopting exact LPE-SQL logic"""
    
    def __init__(self, top_k=4, correct_rate=0.5, engine='qwen2-72b',
                 embedding_model_path="sentence-transformers/all-MiniLM-L6-v2", 
                 device="cpu", accumulate_knowledge_base=True, use_init_knowledge_base=True):
        """
        Initialize retriever with exact LPE-SQL parameters and logic
        """
        self.top_k = top_k
        self.correct_rate = correct_rate
        self.embedding_model = SentenceTransformer(embedding_model_path, device=device)
        
        # Setup knowledge base paths following LPE-SQL structure
        root_path = './src/knowledge_base'
        engine_dir = os.path.join(root_path, f"{engine}_{top_k}_{accumulate_knowledge_base}_{use_init_knowledge_base}_rate_{correct_rate}")
        os.makedirs(engine_dir, exist_ok=True)
        
        self.correct_set_path = os.path.join(engine_dir, "correct_set.json")
        self.mistake_set_path = os.path.join(engine_dir, "mistake_set.json")
        self.correct_vectors_path = os.path.join(engine_dir, "correct_vectors.npy")
        self.mistake_vectors_path = os.path.join(engine_dir, "mistake_vectors.npy")
        
        # Initial knowledge base paths
        self.init_correct_set_path = os.path.join(root_path, "init_correct_set.json")
        self.init_mistake_set_path = os.path.join(root_path, "init_mistake_set.json")
        
        # Load knowledge base
        if use_init_knowledge_base:
            self.correct_set = self._load_json(self.init_correct_set_path) if os.path.exists(self.init_correct_set_path) else []
            self.mistake_set = self._load_json(self.init_mistake_set_path) if os.path.exists(self.init_mistake_set_path) else []
        else:
            self.correct_set = self._load_json(self.correct_set_path) if os.path.exists(self.correct_set_path) else []
            self.mistake_set = self._load_json(self.mistake_set_path) if os.path.exists(self.mistake_set_path) else []
        
        print(f"Loaded knowledge base: {len(self.correct_set)} correct examples, {len(self.mistake_set)} mistake examples")
        
        # Load or generate vectors
        self.correct_vectors = self._load_or_encode_dataset(self.correct_set, self.correct_vectors_path) if self.correct_set else None
        self.mistake_vectors = self._load_or_encode_dataset(self.mistake_set, self.mistake_vectors_path) if self.mistake_set else None
    
    def _load_json(self, filepath):
        """Load JSON data from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON from {filepath}: {e}")
            return []
    
    def _save_json(self, data, filepath):
        """Save JSON data to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save JSON to {filepath}: {e}")
    
    def _encode_dataset(self, dataset):
        """Encode dataset questions to vectors"""
        texts = [entry['question'] for entry in dataset]
        vectors = self.embedding_model.encode(texts)
        return vectors
    
    def _save_vectors_to_disk(self, vectors, filepath):
        np.save(filepath, vectors)
    
    def _load_vectors_from_disk(self, filepath):
        return np.load(filepath)
    
    def _load_or_encode_dataset(self, dataset, filepath):
        """Load existing vectors or encode new ones"""
        if os.path.exists(filepath):
            return self._load_vectors_from_disk(filepath)
        elif dataset:
            vectors = self._encode_dataset(dataset)
            self._save_vectors_to_disk(vectors, filepath)
            return vectors
        else:
            return None
    
    def retrieve_similar_examples(self, query, correct_rate=None):
        """Retrieve similar examples using exact LPE-SQL logic"""
        query_vector = self.embedding_model.encode([query])
        
        # Use provided correct_rate or default
        current_correct_rate = correct_rate if correct_rate is not None else self.correct_rate
        
        num_correct_to_retrieve = int(current_correct_rate * self.top_k)
        num_mistakes_to_retrieve = self.top_k - num_correct_to_retrieve
        
        # Adjust if not enough examples available
        if len(self.mistake_set) < num_mistakes_to_retrieve:
            num_correct_to_retrieve = self.top_k - len(self.mistake_set)
            num_mistakes_to_retrieve = len(self.mistake_set)
        
        correct_examples = self._retrieve_from_vectors(
            query_vector, self.correct_set, self.correct_vectors, num_correct_to_retrieve
        ) if self.correct_vectors is not None and num_correct_to_retrieve > 0 else []
        
        mistake_examples = self._retrieve_from_vectors(
            query_vector, self.mistake_set, self.mistake_vectors, num_mistakes_to_retrieve
        ) if self.mistake_vectors is not None and num_mistakes_to_retrieve > 0 else []
        
        print(f"correct: {num_correct_to_retrieve}; mistake: {num_mistakes_to_retrieve}")
        return correct_examples, mistake_examples
    
    def _retrieve_from_vectors(self, query_vector, dataset, vectors, num_k):
        """Retrieve from vectors using FAISS - exact LPE-SQL implementation"""
        if vectors is None or len(dataset) == 0 or num_k <= 0:
            return []
        
        vector_dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(vector_dimension)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        
        faiss.normalize_L2(query_vector)
        distances, ann = index.search(query_vector, k=min(num_k, len(dataset)))
        print(ann[0])  # Print indices like LPE-SQL does
        similar_examples = [dataset[i] for i in ann[0]]
        return similar_examples
    
    def add_to_sets(self, question, sql, correct=True, **kwargs):
        """Add new example to knowledge base - exact LPE-SQL logic"""
        if correct:
            self.correct_set.append({
                'question': question,
                'hint': kwargs.get('knowledge', ''),
                'sql': sql,
                'thought process': kwargs.get('cot', ''),
                'difficulty': kwargs.get('difficulty', '')
            })
            new_vector = self.embedding_model.encode([question])
            if self.correct_vectors is None:
                self.correct_vectors = new_vector
            else:
                self.correct_vectors = np.vstack([self.correct_vectors, new_vector])
            self._save_vectors_to_disk(self.correct_vectors, self.correct_vectors_path)
            self._save_json(self.correct_set, self.correct_set_path)
        else:
            mistake_entry = {
                'question': question,
                'hint': kwargs.get('knowledge', ''),
                'error_sql': kwargs.get('error_sql', ''),
                'compiler_hint': kwargs.get('compiler_hint', ''),
                'reflective_cot': kwargs.get('reflective_cot', ''),
                'ground_truth_sql': sql,
                'difficulty': kwargs.get('difficulty', ''),
            }
            self.mistake_set.append(mistake_entry)
            new_vector = self.embedding_model.encode([question])
            if self.mistake_vectors is None:
                self.mistake_vectors = new_vector
            else:
                self.mistake_vectors = np.vstack([self.mistake_vectors, new_vector])
            self._save_vectors_to_disk(self.mistake_vectors, self.mistake_vectors_path)
            self._save_json(self.mistake_set, self.mistake_set_path)
    
    def get_in_context_examples(self, query, correct_rate=None):
        """Get in-context examples for few-shot learning"""
        if correct_rate is not None:
            self.correct_rate = correct_rate
        correct_examples, mistake_examples = self.retrieve_similar_examples(query)
        return correct_examples, mistake_examples


# Exact LPE-SQL prompt generation functions
def generate_examples(question, retrieval, correct_rate=None):
    """Generate examples for few-shot learning - exact LPE-SQL implementation"""
    if retrieval.top_k == 0:
        return ""
    if correct_rate is not None:
        retrieval.correct_rate = correct_rate
    
    correct_examples, mistake_examples = retrieval.get_in_context_examples(question, correct_rate)
    
    correct_prompt = '\n\n'.join([
        f"example{index+1}: {{\n" + 
        '\n'.join([f"## {key}: {value}\n" for key, value in example.items() if key != 'difficulty']) + 
        "\n}"
        for index, example in enumerate(correct_examples)
    ])
    
    mistake_prompt = '\n\n'.join([
        f"example{index+1}: {{\n" + 
        '\n'.join([f"## {key}: {value}\n" for key, value in example.items() if key != 'difficulty']) + 
        "\n}"
        for index, example in enumerate(mistake_examples)
    ])
    
    if correct_prompt != "":
        correct_prompt = f"\n###For your reference, here are some examples of Questions, sql queries, and thought processes related to the Question you're working with\n\n{correct_prompt}"
    
    if mistake_prompt != "":
        mistake_prompt = f"### Below are examples of mistakes you've made before that are similar to the question you're about to tackle, so please refer to not making the same mistake!\n\n{mistake_prompt}"
    
    return correct_prompt + '\n\n' + mistake_prompt


def generate_schema_prompt(sql_dialect, db_path):
    """Generate schema prompt - exact implementation from LPE-SQL"""
    # This would be implemented based on the table_schema.py from LPE-SQL
    # For now, implementing a basic version that extracts schema info
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_parts = []
        schema_parts.append(f"### Schema of the database with sample rows:")
        schema_parts.append("=" * 60)
        
        for table in tables:
            table_name = table[0]
            
            # Get table info
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # Build CREATE TABLE statement
            columns_def = []
            for col in columns:
                col_name, col_type = col[1], col[2]
                col_def = f"{col_name} {col_type}"
                if col[5]:  # primary key
                    col_def += " PRIMARY KEY"
                if col[3]:  # not null
                    col_def += " NOT NULL"
                columns_def.append(col_def)
            
            create_table = f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(columns_def) + "\n);"
            schema_parts.append(create_table)
            
            # Add sample data
            try:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                sample_data = cursor.fetchall()
                if sample_data:
                    column_names = [col[1] for col in columns]
                    schema_parts.append(f"\n/*")
                    schema_parts.append(f" {len(sample_data)} example rows:")
                    schema_parts.append(f" SELECT * FROM {table_name} LIMIT {len(sample_data)};")
                    
                    # Format the data nicely
                    col_widths = [max(len(str(name)), max(len(str(row[i])) for row in sample_data)) for i, name in enumerate(column_names)]
                    
                    header = " | ".join(name.ljust(width) for name, width in zip(column_names, col_widths))
                    schema_parts.append(f" {header}")
                    
                    separator = " | ".join("-" * width for width in col_widths)
                    schema_parts.append(f" {separator}")
                    
                    for row in sample_data:
                        data_row = " | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths))
                        schema_parts.append(f" {data_row}")
                    
                    schema_parts.append(f" */")
            except:
                pass
            
            schema_parts.append("")
        
        conn.close()
        return "\n".join(schema_parts)
        
    except Exception as e:
        return f"# Error loading schema: {e}"


def generate_comment_prompt(question, sql_dialect, knowledge=None):
    """Generate comment prompt - exact LPE-SQL implementation"""
    base_prompt = f"-- Using valid {sql_dialect}"
    knowledge_text = " and understanding Hint" if knowledge else ""
    knowledge_prompt = f"-- Hint: {knowledge}" if knowledge else ""
    
    combined_prompt = (
        f"{base_prompt}{knowledge_text}, answer the following questions for the tables provided above.\n"
        f"-- {question}\n"
        f"{knowledge_prompt}"
    )
    return combined_prompt


def generate_cot_prompt():
    """Generate chain-of-thought prompt - exact LPE-SQL implementation"""
    return f"\nGenerate the SQLite for the above question after thinking step by step: "


def generate_instruction_prompt():
    """Generate instruction prompt - exact LPE-SQL implementation"""
    return f"""
        \nIn your response, you do not need to mention your intermediate steps. 
        Do not include any comments in your response.
        Do not need to start with the symbol ```
        Your SQL code should be concise and efficient.
        You only need to return the result SQLite SQL code
        start from SELECT
        """


def generate_common_prompts_sql(db_path, question, sql_dialect, retrieval, knowledge=None, accumulate_knowledge_base=True, correct_rate=None):
    """Generate complete prompts - exact LPE-SQL implementation"""
    if accumulate_knowledge_base:
        examples = generate_examples(question, retrieval, correct_rate)
    else:
        examples = ""
    
    schema_prompt = generate_schema_prompt(sql_dialect, db_path)
    comment_prompt = generate_comment_prompt(question, sql_dialect, knowledge)
    cot_prompt = generate_cot_prompt()
    instruction_prompt = generate_instruction_prompt()
    
    combined_prompts = "\n\n".join([
        examples, schema_prompt, comment_prompt, cot_prompt, instruction_prompt
    ])
    return combined_prompts


class EnhancedSQLGeneratorLPE:
    """Enhanced SQL Generator using exact LPE-SQL experience mode logic"""
    
    def __init__(self, llm_connector=None, top_k=4, correct_rate=0.5, engine='qwen2-72b',
                 accumulate_knowledge_base=True, use_init_knowledge_base=True):
        """
        Initialize with exact LPE-SQL parameters
        """
        self.llm_connector = llm_connector or LocalLLMConnector()
        self.retrieval = TextToSQLRetriever(
            top_k=top_k,
            correct_rate=correct_rate,
            engine=engine,
            accumulate_knowledge_base=accumulate_knowledge_base,
            use_init_knowledge_base=use_init_knowledge_base
        )
        self.postprocessor = MasterSQLPostProcessor()
        self.accumulate_knowledge_base = accumulate_knowledge_base
    
    def generate_sql(self, question: str, db_path: str, knowledge: str = None, correct_rate: float = None) -> str:
        """
        Generate SQL using exact LPE-SQL experience mode logic
        """
        try:
            # Generate prompt using LPE-SQL logic
            prompt = generate_common_prompts_sql(
                db_path=db_path,
                question=question,
                sql_dialect='SQLite',
                retrieval=self.retrieval,
                knowledge=knowledge,
                accumulate_knowledge_base=self.accumulate_knowledge_base,
                correct_rate=correct_rate
            )
            
            # Get SQL from LLM
            response = self.llm_connector(prompt)
            sql = self._extract_sql_from_response(response)
            
            # Post-process the SQL
            corrected_sql = self.postprocessor.fix_sql_syntax(sql, {})
            
            return corrected_sql
            
        except Exception as e:
            print(f"Error in SQL generation: {e}")
            return ""
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL from LLM response - exact logic from original"""
        sql_patterns = [
            r'SELECT.*?(?:;|$)',
            r'select.*?(?:;|$)',
            r'```sql\s*(SELECT.*?)\s*```',
            r'```(SELECT.*?)```'
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            if matches:
                sql = matches[0].strip()
                if not sql.endswith(';'):
                    sql += ';'
                return sql
        
        return response.strip()
    
    def add_experience(self, question: str, sql: str, correct: bool = True, **kwargs):
        """Add experience to knowledge base"""
        self.retrieval.add_to_sets(question, sql, correct, **kwargs)
    
    def get_retrieval_stats(self):
        """Get retrieval statistics"""
        return {
            'correct_set_size': len(self.retrieval.correct_set),
            'mistake_set_size': len(self.retrieval.mistake_set),
            'top_k': self.retrieval.top_k,
            'correct_rate': self.retrieval.correct_rate
        }


# Test function
def test_enhanced_sql_generator_lpe():
    """Test the refactored LPE-SQL style generator"""
    print("=== Testing Enhanced SQL Generator (LPE-SQL Style) ===")
    
    try:
        # Initialize generator
        generator = EnhancedSQLGeneratorLPE(
            top_k=2,
            correct_rate=0.5,
            accumulate_knowledge_base=True,
            use_init_knowledge_base=True
        )
        
        print(f"✓ Generator initialized")
        print(f"✓ Knowledge base: {generator.get_retrieval_stats()}")
        
        # Test SQL generation
        question = "What is the highest salary in the company?"
        db_path = "database.db"
        
        sql = generator.generate_sql(question, db_path)
        print(f"✓ Generated SQL: {sql}")
        
        # Test adding experience
        generator.add_experience(
            question="Find employees with salary above 50000",
            sql="SELECT * FROM employees WHERE salary > 50000",
            correct=True,
            knowledge="Use WHERE clause for filtering by salary",
            cot="Identify salary column and apply filtering condition"
        )
        
        print(f"✓ Updated knowledge base: {generator.get_retrieval_stats()}")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_enhanced_sql_generator_lpe()