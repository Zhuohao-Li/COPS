import json
import os
import subprocess
import argparse
from typing import List, Dict, Set

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
    return [qa['id'] for qa in qa_data 
            if qa['id'] not in correct_ids 
            and qa['type'] != 'search_qa']  # 跳过 search_qa

def get_qa_by_ids(qa_data: List[Dict], ids: List[str]) -> List[Dict]:
    """根据ID列表获取对应的QA对，排除 search_qa"""
    id_to_qa = {qa['id']: qa for qa in qa_data 
                if qa['type'] != 'search_qa'}  # 跳过 search_qa
    return [id_to_qa[id_] for id_ in ids if id_ in id_to_qa]

def prepare_reflection_qa(qa_pair: Dict, pred_answer: str, gold_answer: str) -> Dict:
    """准备带有反思的QA对"""
    reflection_prompt = f"""I need help analyzing why my previous answer was incorrect:

Original Question: {qa_pair['question']}
My Previous Answer: {pred_answer}
Correct Answer: {gold_answer}

Please help me understand:
1. Why was my answer incorrect?
2. What key information did I miss or misinterpret?
3. How should I approach this type of question in the future?

Please provide a concise analysis."""

    enhanced_question = f"{qa_pair['question']}\n\nReflection Context:\n{reflection_prompt}"
    
    return {
        "id": qa_pair['id'],
        "question": enhanced_question,
        "type": qa_pair['type'],
        "asin": qa_pair['asin'],
        "original_answer": pred_answer,
        "correct_answer": gold_answer
    }

def prepare_final_qa(qa_pair: Dict, reflection: str) -> Dict:
    """准备最终的QA对，包含反思内容"""
    enhanced_question = f"""{qa_pair['question']}

Based on previous reflection:
{reflection}

Please provide a new answer, taking into account the above reflection."""

    return {
        "id": qa_pair['id'],
        "question": enhanced_question,
        "type": qa_pair['type'],
        "asin": qa_pair['asin']
    }

def create_reason_prompt(question: str, answer: str) -> str:
    return f"""analyze the following question and answer, and explain why the answer might be incorrect:

question: {question}
answer: {answer}

please explain the reasons in one word: """

def get_incorrect_questions(eval_results_path: str, qa_data: List[Dict]) -> List[Dict]:
    """获取所有错误的问题"""
    with open(eval_results_path, 'r') as f:
        eval_results = json.load(f)
    
    correct_ids = set(eval_results.get('correct_ids', []))
    incorrect_questions = []
    
    for qa in qa_data:
        if qa['id'] not in correct_ids:
            incorrect_questions.append(qa)
    
    return incorrect_questions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max', type=int, default=None)
    parser.add_argument('--reason', action='store_true', help='使用参考文件获取问题和评估结果')
    parser.add_argument('--70b', action='store_true', help='使用70B模型')
    parser.add_argument('--ref', action='store_true', help='处理reflexion_qa.jsonl')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 只在不使用 reason 和 ref 参数时运行第一次 llama_sgl
    if not args.reason and not args.ref:
        first_run_cmd = [
            'python3',
            'llama_sgl.py',
            '--input-dir', args.input_dir,
            '--output-dir', args.output_dir,
        ]
        if args.max:
            first_run_cmd.extend(['--max', str(args.max)])
        if getattr(args, '70b'):
            first_run_cmd.append('--70b')
        subprocess.run(first_run_cmd)
        
        # 重命名第一次结果文件
        first_result_file = '70b_task_short.jsonl' if getattr(args, '70b') else 'task_short.jsonl'
        first_result_path = os.path.join(args.output_dir, first_result_file)
        first_result_new_file = '70b_task_short_ref_first.jsonl' if getattr(args, '70b') else 'task_short_ref_first.jsonl'
        first_result_new_path = os.path.join(args.output_dir, first_result_new_file)
        if os.path.exists(first_result_path):
            os.rename(first_result_path, first_result_new_path)
    
    if args.reason:
        # 从第一次结果文件读取所有问题，包括search_qa
        first_result_file = '70b_task_short_ref_first.jsonl' if getattr(args, '70b') else 'task_short_ref_first.jsonl'
        qa_data = load_jsonl(os.path.join(args.input_dir, first_result_file))
        eval_results_path = os.path.join(args.input_dir, '70b_eval_res_ref/evaluation_results.json')
        
        # 获取正确问题的ID集合
        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)
        correct_ids = set(eval_results.get('correct_ids', []))
        
        # 为错误的非search_qa问题创建原因提示
        incorrect_qa_path = os.path.join(args.output_dir, 'incorrect_qa.jsonl')
        prefix = '70b_' if getattr(args, '70b') else ''
        incorrect_qa_path = os.path.join(args.output_dir, f'{prefix}incorrect_qa.jsonl')
        
        with open(incorrect_qa_path, 'w', encoding='utf-8') as f:
            for qa in qa_data:
                if qa['id'] not in correct_ids and qa['type'] != 'search_qa':
                    prompt = create_reason_prompt(qa['question'], qa['answer'])
                    f.write(json.dumps({'question': prompt}, ensure_ascii=False) + '\n')

        # 使用llama_sgl获取原因
        reason_cmd = [
            'python3',
            'llama_sgl.py',
            '--input-dir', args.output_dir,
            '--output-dir', args.output_dir,
            '--reason'  # 使用reason参数获取原因
        ]
        if getattr(args, '70b'):
            reason_cmd.append('--70b')
        subprocess.run(reason_cmd)

        # 读取原因
        reason_file = '70b_task_short_ref_first.jsonl' if getattr(args, '70b') else 'task_short_ref.jsonl'
        reasons = load_jsonl(os.path.join(args.output_dir, reason_file))
        reason_dict = {i: r['answer'] for i, r in enumerate(reasons)}

        # 创建最终的reflexion数据
        reflexion_data = []
        incorrect_count = 0
        for qa in qa_data:
            reflexion_entry = qa.copy()
            if qa['id'] not in correct_ids and qa['type'] != 'search_qa':
                reason = reason_dict.get(incorrect_count, '')
                reflexion_entry['question'] = f"{qa['question']}\nPrevious incorrect answer: {qa['answer']}\nReason for incorrect answer: {reason}"
                incorrect_count += 1
            reflexion_data.append(reflexion_entry)

        # 保存反思数据到 reflexion_qa.jsonl
        prefix = '70b_' if getattr(args, '70b') else ''
        reflexion_output_path = os.path.join(args.output_dir, f'{prefix}reflexion_qa.jsonl')
        with open(reflexion_output_path, 'w', encoding='utf-8') as f:
            for entry in reflexion_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"Reflexion data saved in: {reflexion_output_path}")

        # 如果指定了--ref参数，使用llama_sgl处理reflexion_qa.jsonl
    if args.ref:
        ref_cmd = [
            'python3',
            'llama_sgl.py',
            '--input-dir', args.output_dir,
            '--output-dir', args.output_dir,
            '--ref'  # 使用ref参数处理reflexion_qa.jsonl
        ]
        if getattr(args, '70b'):
            ref_cmd.append('--70b')
        subprocess.run(ref_cmd)

if __name__ == "__main__":
    main() 