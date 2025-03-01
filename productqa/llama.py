import json
import os
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

def format_prompt(question: str, metadata: Dict, schema: Dict) -> str:
    """Format the prompt for the model"""
    prompt = f"""As a product QA assistant, please answer the following question based on the provided product information.

Product Information:
{json.dumps(metadata, indent=2, ensure_ascii=False)}

Product Schema:
{json.dumps(schema, indent=2, ensure_ascii=False)}

Question: {question}

Please provide a concise answer. For fact_qa or reasoning_qa type questions, respond with yes/no or specific values. For search_qa type questions, return the matching product ASIN.
Answer:"""
    return prompt

def generate_answer(model, tokenizer, prompt: str) -> str:
    """Generate answer using the model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the answer part
    answer = response.split("Answer:")[-1].strip()
    return answer

def process_qa(qa_data: List[Dict], metadata: Dict, schema: Dict, model, tokenizer) -> List[Dict]:
    """Process QA data"""
    results = []
    total = len(qa_data)
    
    for idx, item in enumerate(qa_data, 1):
        try:
            # Prepare prompt
            prompt = format_prompt(item['question'], metadata.get(item['asin'], {}), schema)
            
            # Generate answer
            answer = generate_answer(model, tokenizer, prompt)
            
            # Save result
            result = {
                'id': item['id'],
                'question': item['question'],
                'answer': answer,
                'type': item['type']
            }
            results.append(result)
            
            print(f"Processed {idx}/{total}: {item['id']}")
            
            # Save intermediate results every 10 items
            if idx % 10 == 0:
                save_jsonl(results, 'intermediate_results.jsonl')
                
        except Exception as e:
            print(f"Error processing {item['id']}: {str(e)}")
            continue
    
    return results

def main():
    # Set paths
    base_path = "/productqa/test"
    task_folders = ['all_pans']  # Add other task folders as needed
    
    # Load model
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    
    for folder in task_folders:
        try:
            folder_path = os.path.join(base_path, folder)
            print(f"Processing folder: {folder}")
            
            # Load data
            qa_data = load_jsonl(os.path.join(folder_path, 'qa.jsonl'))
            metadata = load_json(os.path.join(folder_path, 'metadata.json'))
            schema = load_json(os.path.join(folder_path, 'schema.json'))
            
            # Process data
            results = process_qa(qa_data, metadata, schema, model, tokenizer)
            
            # Save results
            output_path = os.path.join(folder_path, 'task.jsonl')
            save_jsonl(results, output_path)
            print(f"Completed processing {folder}. Results saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing folder {folder}: {str(e)}")
            continue

if __name__ == "__main__":
    main()