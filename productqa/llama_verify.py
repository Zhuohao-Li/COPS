import json
import os
from typing import List, Dict, Tuple
import argparse

def short_eval(gold: str, pred: str) -> bool:
    """Simple evaluation method to check if answers match"""
    if not isinstance(pred, str):
        return False
    return gold.lower().strip() == pred.lower().strip()

def search_eval(gold: List[Dict], pred: str) -> bool:
    """Evaluate answers for search type questions"""
    if not isinstance(pred, str):
        return False
    gold_asins = set(x['asin'] for x in gold)
    # Extract ASIN from prediction
    pred = pred.upper()
    if 'ASIN:' in pred:
        pred_asin = pred.split('ASIN:')[1].strip()
        return pred_asin in gold_asins
    return False

def evaluate_answers(qa_path: str, pred_path: str) -> Tuple[List[str], Dict[str, float], Dict[str, List[int]]]:
    """Evaluate predicted answers and return correct answer IDs, metrics and type stats"""
    # Load original QA data
    golds = []
    with open(qa_path, 'r', encoding='utf-8') as f:
        for line in f:
            golds.append(json.loads(line))
            
    # Load predictions
    preds = []
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            preds.append(json.loads(line))
            
    correct_ids = []
    # Comment out search_qa in type_stats since we're not evaluating it
    type_stats = {
        # 'search_qa': [0, 0],  # [correct_count, total_count]
        'fact_qa': [0, 0], 
        'reasoning_qa': [0, 0]
    }
    
    for gold, pred in zip(golds, preds):
        qa_type = gold['type']
        qa_id = gold['id']
        
        # Skip search_qa questions
        if qa_type == 'search_qa':
            continue
        
        # Update total count for this type
        type_stats[qa_type][1] += 1
        
        # Evaluate based on question type (only fact_qa and reasoning_qa)
        is_correct = short_eval(gold['short_answer'], pred['answer'])
            
        if is_correct:
            type_stats[qa_type][0] += 1
            correct_ids.append(qa_id)
    
    # Calculate accuracy for each type
    metrics = {}
    for qa_type, (correct, total) in type_stats.items():
        if total > 0:
            metrics[qa_type] = correct / total
        else:
            metrics[qa_type] = 0.0
            
    return correct_ids, metrics, type_stats

def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='Evaluate QA predictions')
    parser.add_argument('--qa-dir', type=str, required=True, 
                       help='Path to the ground truth qa.jsonl file')
    parser.add_argument('--pred-dir', type=str, required=True, 
                       help='Path to the prediction file (task_short.jsonl or task_short_cops.jsonl)')
    parser.add_argument('--output-dir', type=str, required=True, 
                       help='Directory to save evaluation_results.json')
    args = parser.parse_args()

    # Use file paths directly for input files
    qa_path = args.qa_dir      # Direct path to qa.jsonl
    pred_path = args.pred_dir  # Direct path to prediction file
    
    # Create output directory and set output file path
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'evaluation_results.json')
    
    # Evaluate answers
    correct_ids, metrics, type_stats = evaluate_answers(qa_path, pred_path)
    
    # Calculate total stats
    total_correct = sum(stats[0] for stats in type_stats.values())
    total_questions = sum(stats[1] for stats in type_stats.values())
    
    # Save results
    results = {
        'correct_ids': correct_ids,
        'metrics': metrics,
        'total_accuracy': total_correct / total_questions if total_questions > 0 else 0
    }
    
    # Write results to evaluation_results.json in the output directory
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    # Print evaluation results
    print('\nEvaluation Results:')
    print('-' * 50)
    for qa_type, accuracy in metrics.items():
        print(f'{qa_type}: {accuracy:.3f}')
    print(f'Total Accuracy: {results["total_accuracy"]:.3f}')
    print(f'\nResults saved to: {output_path}')

if __name__ == '__main__':
    main()