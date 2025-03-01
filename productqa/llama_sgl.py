import json
import os
from typing import List, Dict
from sglang import gen, RuntimeEndpoint
import requests
from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
    print_highlight,
)
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

def save_jsonl(data: List[Dict], file_path: str):
    """Save data in JSONL format"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def format_prompt(question: str, metadata: Dict, schema: Dict, similar_examples: List[Dict] = None) -> str:
    """Format the prompt for the model"""
    examples_text = ""
    if similar_examples:
        examples_text = "\nHere are some similar questions and their correct answers:\n"
        for example in similar_examples:
            examples_text += f"Q: {example['question']}\nA: {example['answer']}\n"

    prompt = f"""As a product QA assistant, please answer the following question based on the provided product information.

Product Information:
{json.dumps(metadata, indent=2, ensure_ascii=False)}

Product Schema:
{json.dumps(schema, indent=2, ensure_ascii=False)}{examples_text}

Question: {question}

Please provide a concise answer. For fact_qa or reasoning_qa type questions, respond with yes/no or specific values. For search_qa type questions, return the matching product ASIN. Answer shortly, for example, "yes" or "no" or "ASIN:B000000000", do not have further explanations.
Answer:"""
    return prompt

def process_qa(qa_data: List[Dict], metadata: Dict, schema: Dict, use_70b: bool = False, max_samples: int = None) -> List[Dict]:
    """Process QA data using the LLM API"""
    if max_samples:
        qa_data = qa_data[:max_samples]
    results = []
    total = len(qa_data)
    
    for idx, item in enumerate(qa_data, 1):
        try:
            # Get similar examples if they exist (for COPS enhanced data)
            similar_examples = item.get('similar_examples', None)
            
            # Prepare prompt with similar examples if available
            prompt = format_prompt(
                item['question'], 
                metadata.get(item.get('asin', ''), {}),  # Use empty dict if asin not found
                schema,
                similar_examples
            )
            
            # Call API
            url = "http://172.17.0.18:40000/v1/chat/completions"
            # 根据参数选择模型
            if use_70b:
                data = {
                    "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
                    "messages": [{"role": "user", "content": prompt}],
                }
            else:
                data = {
                    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "messages": [{"role": "user", "content": prompt}],
                }
            
            response = requests.post(url, json=data)
            response_json = response.json()
            answer = response_json['choices'][0]['message']['content']
            
            # Save result
            result = {
                'id': item['id'],
                'question': item['question'],
                'answer': answer,
                'type': item['type'],
                'asin': item.get('asin', '')  # Preserve asin in results
            }
            results.append(result)
            
            print(f"Processed {idx}/{total}: {item['id']}")
            
            # Save intermediate results every 10 items
            if idx % 10 == 0:
                save_jsonl(results, 'intermediate_results.jsonl')
                print_highlight(f"Saved intermediate results after processing {idx} items")
                
        except Exception as e:
            print(f"Error processing {item['id']}: {str(e)}")
            continue
    
    return results

def init_check_server():
    wait_for_server("http://localhost:40000")
    
def test_connection_example():
    url = "http://172.17.0.18:40000/v1/chat/completions"  # Using the correct endpoint from server logs
    data = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }

    # url = "http://172.17.0.20:40000/generate"
    # data = {"text": "What is the capital of France?"}
    # print_highlight(f"Attempting to connect to {url}")
    response = requests.post(url, json=data)
    # print_highlight(response.json())
    # print_highlight(response.json())
    print(response.json())

def main():
    parser = argparse.ArgumentParser(description='Process QA pairs using LLaMA model')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory containing qa.jsonl, metadata.json, and schema.json')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--max', type=int, default=None, help='Maximum number of QA pairs to process')
    parser.add_argument('--cops', action='store_true', help='Whether the input is from COPS enhancement')
    parser.add_argument('--70b', action='store_true', help='Use LLaMA-3.1-70B-Instruct model instead of 8B')
    parser.add_argument('--ref', action='store_true', help='Process reflexion_qa.jsonl')
    parser.add_argument('--reason', action='store_true', help='Process incorrect_qa.jsonl to get reasons')
    args = parser.parse_args()
    
    try:
        print_highlight(f"Processing input directory: {args.input_dir}")
        
        # Load data based on mode
        if args.reason:
            # 处理incorrect_qa.jsonl获取原因
            qa_file = 'incorrect_qa.jsonl'
            qa_data = load_jsonl(os.path.join(args.input_dir, qa_file))
            metadata = {}
            schema = {}
        elif args.ref:
            # 处理reflexion_qa.jsonl
            qa_file = 'reflexion_qa.jsonl'
            qa_data = load_jsonl(os.path.join(args.input_dir, qa_file))
            metadata = {}
            schema = {}
        else:
            # 正常模式
            qa_file = 'enhanced_qa.jsonl' if args.cops else 'qa.jsonl'
            qa_data = load_jsonl(os.path.join(args.input_dir, qa_file))
            metadata = load_json(os.path.join(args.input_dir, 'metadata.json'))
            schema = load_json(os.path.join(args.input_dir, 'schema.json'))
        
        # Process data
        results = process_qa(qa_data, metadata, schema, use_70b=getattr(args, '70b'), max_samples=args.max)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Add 70b prefix to output filename if using 70b model
        if args.reason:
            base_filename = 'task_short_ref.jsonl'
        elif args.ref:
            base_filename = 'task_short_ref.jsonl'
        else:
            base_filename = 'task_short_cops.jsonl' if args.cops else 'task_short.jsonl'
            
        output_filename = f"70b_{base_filename}" if getattr(args, '70b') else base_filename
        output_path = os.path.join(args.output_dir, output_filename)
        save_jsonl(results, output_path)
        print_highlight(f"Completed processing. Results saved to {output_path}")
            
    except Exception as e:
        print(f"Error processing: {str(e)}")

if __name__ == "__main__":
    main() 