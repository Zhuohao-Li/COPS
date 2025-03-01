import json
import os
import subprocess
from typing import List, Dict, Set
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import argparse

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_json(file_path: str) -> Dict:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_incorrect_ids(qa_data: List[Dict], correct_ids: Set[str]) -> List[str]:
    """获取需要重新处理的QA对ID"""
    return [qa['id'] for qa in qa_data if qa['id'] not in correct_ids]

def get_qa_by_ids(qa_data: List[Dict], ids: List[str]) -> List[Dict]:
    """根据ID列表获取对应的QA对"""
    id_to_qa = {qa['id']: qa for qa in qa_data}
    return [id_to_qa[id_] for id_ in ids if id_ in id_to_qa]

def find_similar_correct_qa(
    target_question: str,
    correct_qa_pairs: List[Dict],
    model: SentenceTransformer,
    qa_pair: Dict,
    top_k: int = 3
) -> List[Dict]:
    """使用KNN算法找到最相似的k个正确QA对，排除已经使用过的示例"""
    # 获取之前使用过的示例
    used_examples = set()
    if 'similar_examples' in qa_pair:
        used_examples = {ex['question'] for ex in qa_pair['similar_examples']}
    
    # 计算目标问题的embedding
    target_embedding = model.encode(target_question)
    
    # 计算所有正确QA对的embeddings
    correct_questions = [qa['question'] for qa in correct_qa_pairs]
    correct_embeddings = model.encode(correct_questions)
    
    # 使用KNN找到更多的邻居（比需要的多一些，以便排除已使用的示例）
    k = min(top_k + len(used_examples) + 5, len(correct_qa_pairs))
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(correct_embeddings)
    
    # 获取最近的k个邻居的距离和索引
    distances, indices = knn.kneighbors([target_embedding])
    
    # 过滤掉已使用的示例，只保留新的示例
    new_similar_pairs = []
    for idx in indices[0]:
        qa = correct_qa_pairs[idx]
        if qa['question'] not in used_examples:
            new_similar_pairs.append(qa)
            if len(new_similar_pairs) >= top_k:
                break
    
    # 如果有之前的示例，将新示例添加到已有示例后面
    if 'similar_examples' in qa_pair:
        previous_examples = [
            correct_qa for correct_qa in correct_qa_pairs 
            if any(ex['question'] == correct_qa['question'] for ex in qa_pair['similar_examples'])
        ]
        return previous_examples + new_similar_pairs
    
    return new_similar_pairs

def prepare_enhanced_qa(qa_pair: Dict, similar_qa_pairs: List[Dict]) -> Dict:
    """Prepare enhanced QA pair with similar examples,保留并扩展之前的示例"""
    # 构建增强的问题，包含所有示例
    enhanced_question = qa_pair['question'] + "\n\nSimilar Examples:\n"
    
    # 添加所有示例（包括之前的和新的）
    for i, similar_qa in enumerate(similar_qa_pairs, 1):
        answer = similar_qa.get('answer', similar_qa.get('short_answer', 'No answer available'))
        enhanced_question += f"\nExample {i}:\nQ: {similar_qa['question']}\nA: {answer}"
    
    return {
        "id": qa_pair['id'],
        "question": enhanced_question,
        "type": qa_pair['type'],
        "asin": qa_pair['asin'],
        "similar_examples": [
            {
                "question": similar_qa['question'],
                "answer": similar_qa.get('answer', similar_qa.get('short_answer', 'No answer available'))
            } for similar_qa in similar_qa_pairs
        ]
    }

def main():
    parser = argparse.ArgumentParser(description='Process incorrect QA pairs with similar correct examples')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory containing qa.jsonl')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--max', type=int, default=None, help='Maximum number of QA pairs to process')
    parser.add_argument('--eval-dir', type=str, help='Directory containing evaluation results, defaults to input-dir/evaluation_results')
    parser.add_argument('--70b', action='store_true', help='Use LLaMA-3.1-70B-Instruct model instead of 8B')
    args = parser.parse_args()

    # 加载数据
    qa_data = load_jsonl(os.path.join(args.input_dir, 'qa.jsonl'))
    
    # 如果指定了max参数，只取前max个QA对
    if args.max:
        qa_data = qa_data[:args.max]
    
    # 使用指定的评估结果目录，如果未指定则使用默认路径
    eval_dir = args.eval_dir if args.eval_dir else os.path.join(args.input_dir, 'evaluation_results')
    eval_results = load_json(os.path.join(eval_dir, 'evaluation_results.json'))
    correct_ids = set(eval_results['correct_ids'])

    # 获取需要重新处理的QA对
    incorrect_ids = get_incorrect_ids(qa_data, correct_ids)

    # 获取正确的QA对
    correct_qa_pairs = get_qa_by_ids(qa_data, list(correct_ids))
    
    # 加载embedding模型
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # 准备增强后的QA对
    enhanced_qa_pairs = []
    for qa_pair in qa_data:
        if qa_pair['id'] in incorrect_ids:
            similar_qa_pairs = find_similar_correct_qa(
                qa_pair['question'], 
                correct_qa_pairs, 
                embedding_model,
                qa_pair  # 传入原始QA对以检查之前的示例
            )
            enhanced_qa = prepare_enhanced_qa(qa_pair, similar_qa_pairs)
            enhanced_qa_pairs.append(enhanced_qa)
        else:
            enhanced_qa_pairs.append(qa_pair)

    # 保存增强后的QA对
    os.makedirs(args.output_dir, exist_ok=True)
    enhanced_qa_path = os.path.join(args.output_dir, 'enhanced_qa.jsonl')
    with open(enhanced_qa_path, 'w', encoding='utf-8') as f:
        for qa in enhanced_qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')

    # 调用llama_sgl.py处理增强后的QA对
    llama_cmd = [
        'python3', 
        'llama_sgl.py',
        '--input-dir', args.output_dir,
        '--output-dir', args.output_dir,
        '--max', str(len(enhanced_qa_pairs)),  # 使用实际的enhanced_qa_pairs长度
        '--cops'
    ]
    
    # 如果指定了70b参数，添加到命令中
    if getattr(args, '70b'):
        llama_cmd.append('--70b')
    
    subprocess.run(llama_cmd)

if __name__ == "__main__":
    main()
