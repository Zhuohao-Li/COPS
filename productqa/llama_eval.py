import json
import os
from typing import List, Dict, Tuple, Set
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def normalize_answer(answer: str) -> str:
    """Normalize answer text for better matching"""
    # 转换为小写并去除多余空格
    answer = answer.lower().strip()
    
    # yes/no答案的同义词映射
    yes_synonyms = {'yes', 'correct', 'true', 'right', 'yeah', 'yep', 'positive', 'affirmative'}
    no_synonyms = {'no', 'incorrect', 'false', 'wrong', 'nope', 'negative'}
    
    if answer in yes_synonyms:
        return 'yes'
    if answer in no_synonyms:
        return 'no'
    
    return answer

def short_eval(gold: str, pred: str, model: SentenceTransformer = None, threshold: float = 0.7) -> bool:
    """Enhanced evaluation method to check if answers match or are semantically similar"""
    if not isinstance(pred, str):
        return False
    
    # 标准化答案
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(pred)
    
    # 直接字符串匹配
    if gold_norm == pred_norm:
        return True
    
    # 数字匹配 - 允许不同的格式
    if gold_norm.replace('.0', '') == pred_norm.replace('.0', ''):
        return True
    
    # 如果提供了模型，计算语义相似度
    if model is not None:
        try:
            # 编码答案
            gold_embedding = model.encode(gold_norm)
            pred_embedding = model.encode(pred_norm)
            
            # 计算余弦相似度
            similarity = cosine_similarity([gold_embedding], [pred_embedding])[0][0]
            
            return similarity >= threshold
        except:
            return False
            
    return False

def search_eval(gold: List[Dict], pred: str) -> bool:
    """Evaluate answers for search type questions with more flexible matching"""
    if not isinstance(pred, str):
        return False
    
    gold_asins = set(x['asin'] for x in gold)
    pred = pred.upper()
    
    # 标准ASIN格式匹配
    if 'ASIN:' in pred:
        pred_asin = pred.split('ASIN:')[1].strip()
        return pred_asin in gold_asins
    
    # 尝试从文本中提取ASIN格式的字符串
    import re
    asin_pattern = r'B[0-9A-Z]{9}'
    found_asins = re.findall(asin_pattern, pred)
    
    for asin in found_asins:
        if asin in gold_asins:
            return True
    
    return False

def evaluate_answer(pred_answer: str, gold_answer: str) -> bool:
    """
    评估预测答案是否正确
    1. 忽略大小写
    2. 对于短答案，检查完全匹配
    3. 对于较长答案，检查gold_answer是否包含在pred_answer中
    """
    pred_answer = pred_answer.lower().strip()
    gold_answer = gold_answer.lower().strip()
    
    # 对于短答案（如"yes"/"no"），进行完全匹配
    if len(gold_answer.split()) <= 2:
        return pred_answer == gold_answer
    
    # 对于较长答案，检查gold_answer是否完全包含在pred_answer中
    return gold_answer in pred_answer

def process_evaluation(qa_data: List[Dict], pred_data: List[Dict]) -> Tuple[Set[str], Dict]:
    """处理评估结果"""
    correct_ids = set()
    results = {}
    
    # 创建预测答案的查找字典
    pred_dict = {item['id']: item.get('answer', '') for item in pred_data}
    
    for qa in qa_data:
        qa_id = qa['id']
        if qa_id not in pred_dict:
            continue
            
        pred_answer = pred_dict[qa_id]
        gold_answer = qa.get('short_answer', '')  # 使用short_answer作为标准答案
        
        is_correct = evaluate_answer(pred_answer, gold_answer)
        
        if is_correct:
            correct_ids.add(qa_id)
            
        results[qa_id] = {
            'question': qa['question'],
            'predicted': pred_answer,
            'gold': gold_answer,
            'is_correct': is_correct
        }
    
    return correct_ids, results

def evaluate_answers(qa_path: str, pred_path: str, use_similarity: bool = True, threshold: float = 0.7) -> Tuple[List[str], Dict[str, float], Dict[str, List[int]]]:
    """Evaluate predicted answers and return correct answer IDs, metrics and type stats"""
    # 如果使用相似度评估，加载模型
    model = None
    if use_similarity:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
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
    
    # 加载之前的评估结果（如果存在）
    eval_dir = os.path.dirname(pred_path)
    previous_results_path = os.path.join(eval_dir, 'evaluation_results.json')
    previous_correct_ids = set()
    if os.path.exists(previous_results_path):
        try:
            with open(previous_results_path, 'r', encoding='utf-8') as f:
                previous_results = json.load(f)
                previous_correct_ids = set(previous_results.get('correct_ids', []))
        except:
            print("Warning: Could not load previous evaluation results")
            
    correct_ids = []
    type_stats = {
        'fact_qa': [0, 0], 
        'reasoning_qa': [0, 0]
    }
    
    for gold, pred in zip(golds, preds):
        qa_type = gold['type']
        qa_id = gold['id']
        
        # 跳过 search_qa 类型的问题
        if qa_type == 'search_qa':
            continue
            
        # Update total count for this type
        type_stats[qa_type][1] += 1
        
        # 进行新的评估
        is_correct = short_eval(gold['short_answer'], pred['answer'], 
                              model if use_similarity else None, 
                              threshold)
        
        # 如果当前预测正确，或者之前的预测是正确的，则标记为正确
        if is_correct or qa_id in previous_correct_ids:
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
    parser = argparse.ArgumentParser(description='Evaluate QA predictions')
    parser.add_argument('--qa-dir', type=str, required=True, 
                       help='Path to the ground truth qa.jsonl file')
    parser.add_argument('--pred-dir', type=str, required=True, 
                       help='Path to the prediction file')
    parser.add_argument('--output-dir', type=str, required=True, 
                       help='Directory to save evaluation_results.json')
    parser.add_argument('--use-similarity', action='store_true',
                       help='Use semantic similarity for evaluation')
    parser.add_argument('--similarity-threshold', type=float, default=0.7,
                       help='Threshold for semantic similarity (default: 0.7)')
    args = parser.parse_args()

    # Evaluate answers
    correct_ids, metrics, type_stats = evaluate_answers(
        args.qa_dir, 
        args.pred_dir, 
        args.use_similarity,
        args.similarity_threshold
    )
    
    # Calculate total stats (excluding search_qa)
    total_correct = sum(stats[0] for stats in type_stats.values())
    total_questions = sum(stats[1] for stats in type_stats.values())
    
    # Save results
    results = {
        'correct_ids': correct_ids,
        'metrics': metrics,
        'total_accuracy': total_correct / total_questions if total_questions > 0 else 0
    }
    
    # Create output directory and save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    # Print evaluation results
    print('\nEvaluation Results:')
    print('-' * 50)
    for qa_type, accuracy in metrics.items():
        correct, total = type_stats[qa_type]
        print(f'{qa_type}: {accuracy:.3f} ({correct}/{total})')
    print(f'Total Accuracy: {results["total_accuracy"]:.3f} ({total_correct}/{total_questions})')
    print(f'\nResults saved to: {output_path}')

if __name__ == '__main__':
    main()