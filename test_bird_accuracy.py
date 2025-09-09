#!/usr/bin/env python3
"""
Test current algorithm accuracy with 10 BIRD dataset samples
"""

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time

# Add project path
sys.path.append(str(Path(__file__).parent))

from bird_evaluator_unified import UnifiedBIRDEvaluator


def load_bird_samples(data_path: str = "data", sample_count: int = 10) -> List[Dict[str, Any]]:
    """Load BIRD dataset samples"""
    dev_file = os.path.join(data_path, "dev.json")
    
    with open(dev_file, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    
    # Select diverse samples: different databases and difficulty levels
    selected_samples = []
    db_ids_used = set()
    difficulty_counts = {'simple': 0, 'moderate': 0, 'challenging': 0}
    
    for sample in all_samples:
        if len(selected_samples) >= sample_count:
            break
            
        db_id = sample['db_id']
        difficulty = sample.get('difficulty', 'simple')
        
        # Ensure diversity: max 2 per database, balanced difficulties
        if db_id in db_ids_used and list(db_ids_used).count(db_id) >= 2:
            continue
            
        if difficulty_counts[difficulty] >= (sample_count // 3 + 1):
            continue
        
        selected_samples.append(sample)
        db_ids_used.add(db_id)
        difficulty_counts[difficulty] += 1
    
    print(f"✓ Selected {len(selected_samples)} diverse samples")
    print(f"  - Databases: {len(set(s['db_id'] for s in selected_samples))}")
    print(f"  - Difficulties: {difficulty_counts}")
    
    return selected_samples


def test_single_sample(sample: Dict[str, Any], evaluator: UnifiedBIRDEvaluator) -> Dict[str, Any]:
    """Test a single BIRD sample"""
    db_id = sample['db_id']
    question = sample['question']
    ground_truth_sql = sample['SQL']
    evidence = sample.get('evidence', '')
    difficulty = sample.get('difficulty', 'unknown')
    
    print(f"\n--- Testing Sample ---")
    print(f"DB: {db_id}")
    print(f"Difficulty: {difficulty}")
    print(f"Question: {question[:100]}...")
    
    try:
        # Get database path
        db_path = os.path.join("data", "dev_databases", db_id, f"{db_id}.sqlite")
        
        if not os.path.exists(db_path):
            print(f"✗ Database not found: {db_path}")
            return {
                'sample_id': sample['question_id'],
                'success': False,
                'error': f'Database not found: {db_path}',
                'difficulty': difficulty
            }
        
        # Test with hybrid SQL generator (experience + exploration modes)
        start_time = time.time()
        
        # Prepare question data
        question_data = {
            'question': question,
            'db_id': db_id,
            'difficulty': difficulty,
            'question_id': sample['question_id']
        }
        
        # Evaluate using the unified evaluator
        result = evaluator.evaluate_single_question(
            question_data=question_data,
            gold_sql=ground_truth_sql,
            idx=sample['question_id']
        )
        
        execution_time = time.time() - start_time
        success = result['correct']
        generated_sql = result['predicted_sql']
        details = result.get('details', {})
        
        print(f"✓ Generation success: {success}")
        print(f"✓ Execution time: {execution_time:.2f}s")
        
        if success:
            print(f"✓ Generated SQL: {generated_sql[:100]}...")
            print(f"✓ Ground truth: {ground_truth_sql[:100]}...")
            
            # Check if SQL is equivalent (basic check)
            sql_match = generated_sql.strip().lower() == ground_truth_sql.strip().lower()
            
            return {
                'sample_id': sample['question_id'],
                'success': True,
                'sql_match': sql_match,
                'generated_sql': generated_sql,
                'ground_truth_sql': ground_truth_sql,
                'execution_time': execution_time,
                'mode_used': details.get('mode_used', 'unknown'),
                'difficulty': difficulty,
                'details': details
            }
        else:
            error_msg = details.get('error', 'Unknown error')
            print(f"✗ Generation failed: {error_msg}")
            
            return {
                'sample_id': sample['question_id'],
                'success': False,
                'error': error_msg,
                'execution_time': execution_time,
                'difficulty': difficulty,
                'details': details
            }
            
    except Exception as e:
        print(f"✗ Exception during testing: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'sample_id': sample['question_id'],
            'success': False,
            'error': str(e),
            'difficulty': difficulty
        }


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze test results"""
    total_samples = len(results)
    successful_samples = sum(1 for r in results if r['success'])
    
    # Calculate accuracy by difficulty
    difficulty_stats = {}
    for difficulty in ['simple', 'moderate', 'challenging']:
        difficulty_results = [r for r in results if r['difficulty'] == difficulty]
        if difficulty_results:
            successful_difficulty = sum(1 for r in difficulty_results if r['success'])
            sql_matches = sum(1 for r in difficulty_results if r.get('sql_match', False))
            
            difficulty_stats[difficulty] = {
                'total': len(difficulty_results),
                'successful': successful_difficulty,
                'success_rate': successful_difficulty / len(difficulty_results),
                'sql_matches': sql_matches,
                'accuracy': sql_matches / len(difficulty_results) if difficulty_results else 0
            }
    
    # Mode usage statistics
    mode_stats = {}
    for result in results:
        if result['success']:
            mode = result.get('mode_used', 'unknown')
            mode_stats[mode] = mode_stats.get(mode, 0) + 1
    
    # Execution time statistics
    execution_times = [r.get('execution_time', 0) for r in results if r['success']]
    avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
    
    return {
        'total_samples': total_samples,
        'successful_samples': successful_samples,
        'overall_success_rate': successful_samples / total_samples,
        'difficulty_stats': difficulty_stats,
        'mode_stats': mode_stats,
        'avg_execution_time': avg_execution_time,
        'results': results
    }


def main():
    """Main test function"""
    print("=" * 80)
    print("ILEX-SQL Algorithm Accuracy Test with BIRD Dataset")
    print("Testing 10 diverse samples from BIRD dev set")
    print("=" * 80)
    
    try:
        # Load BIRD samples
        print("Loading BIRD dataset samples...")
        samples = load_bird_samples(sample_count=10)
        
        if not samples:
            print("✗ No samples loaded")
            return 1
        
        # Initialize evaluator
        print("\nInitializing BIRD evaluator...")
        evaluator = UnifiedBIRDEvaluator(
            data_dir="data",
            db_root="data/dev_databases",
            max_concurrency=1,
            use_local_model=True,
            use_mock=True,  # Use mock mode to avoid LLM dependencies for basic testing
            timeout=60
        )
        print("✓ BIRD evaluator initialized")
        
        # Test samples
        print(f"\nTesting {len(samples)} samples...")
        results = []
        
        for i, sample in enumerate(samples):
            print(f"\n--- Sample {i+1}/{len(samples)} ---")
            result = test_single_sample(sample, evaluator)
            results.append(result)
        
        # Analyze results
        print("\n" + "=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
        
        analysis = analyze_results(results)
        
        print(f"✓ Total samples tested: {analysis['total_samples']}")
        print(f"✓ Successful generations: {analysis['successful_samples']}")
        print(f"✓ Overall success rate: {analysis['overall_success_rate']:.2%}")
        print(f"✓ Average execution time: {analysis['avg_execution_time']:.2f}s")
        
        # Difficulty breakdown
        print(f"\n--- Results by Difficulty ---")
        for difficulty, stats in analysis['difficulty_stats'].items():
            print(f"\n{difficulty.upper()}:")
            print(f"  Total: {stats['total']}")
            print(f"  Successful: {stats['successful']}")
            print(f"  Success rate: {stats['success_rate']:.2%}")
            print(f"  SQL accuracy: {stats['accuracy']:.2%}")
        
        # Mode usage
        print(f"\n--- Mode Usage ---")
        for mode, count in analysis['mode_stats'].items():
            print(f"  {mode}: {count} samples")
        
        # Detailed results
        print(f"\n--- Detailed Results ---")
        for i, result in enumerate(results):
            sample_id = result['sample_id']
            success = result['success']
            difficulty = result['difficulty']
            
            if success:
                sql_match = result.get('sql_match', False)
                mode = result.get('mode_used', 'unknown')
                exec_time = result.get('execution_time', 0)
                print(f"Sample {sample_id} ({difficulty}): ✓ Success | Match: {sql_match} | Mode: {mode} | Time: {exec_time:.2f}s")
            else:
                error = result.get('error', 'Unknown error')
                print(f"Sample {sample_id} ({difficulty}): ✗ Failed | Error: {error}")
        
        # Save results
        results_file = "bird_test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {results_file}")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())