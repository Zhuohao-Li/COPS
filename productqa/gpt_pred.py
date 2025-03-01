import json
import os
import time
import openai
from tqdm import tqdm

# 提示模板
qa_prompt = """
you are a professional amazon customer service representative. Please provide two answers based on the question: a short direct answer and a detailed explanation.

question: {question}

Please answer in the following format:
Short answer: [your short answer]
Detailed answer: [your detailed explanation]
"""

search_prompt = """
Based on the following product list and user question, please recommend the most suitable product (up to 3). Only return the ASIN number list of the product.

Product list:
{products}

User question: {question}

Please directly return the ASIN list, for example: ["B001XXX", "B002XXX"]
"""

def get_completion(messages, retries=3, delay=5):
    for _ in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message['content']
        except Exception as e:
            print(f"API call error: {e}")
            if _ < retries - 1:
                time.sleep(delay)
            else:
                raise e

def parse_qa_response(response):
    try:
        # 分割回答部分
        parts = response.lower().split('short answer:')
        if len(parts) < 2:
            return None, None
        
        rest = parts[1].split('long answer:')
        if len(rest) < 2:
            return None, None
        
        short_answer = rest[0].strip()
        long_answer = rest[1].strip()
        return short_answer, long_answer
    except:
        return None, None

def parse_search_response(response):
    try:
        # 尝试解析JSON列表
        response = response.strip()
        if response.startswith('[') and response.endswith(']'):
            asins = json.loads(response)
            return asins if isinstance(asins, list) else []
        return []
    except:
        return []

def format_product_description(asin, info):
    """根据产品信息动态生成描述"""
    # 基本信息（所有产品通常都有的）
    desc = [
        f"ASIN: {asin}",
        f"标题: {info.get('title', 'N/A')}",
        f"品牌: {info.get('brand', 'N/A')}",
        f"价格: ${info.get('price', 'N/A')}"
    ]
    
    # 添加其他可用的属性（跳过已经添加的基本属性）
    skip_keys = {'asin', 'title', 'brand', 'price'}
    for key, value in info.items():
        if key.lower() not in skip_keys and value:
            # 将键名格式化为更易读的形式
            formatted_key = key.replace('_', ' ').title()
            desc.append(f"{formatted_key}: {value}")
    
    return "\n".join(desc)

def process_category(category, test_dir, metadata_path):
    results = []
    
    # 读取metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 读取测试数据
    qa_path = os.path.join(test_dir, category, 'qa.jsonl')
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_data = [json.loads(line) for line in f]
    
    print(f"处理类别: {category}")
    for qa in tqdm(qa_data):
        if qa['type'] == 'search_qa':
            # 构建产品列表文本
            products_text = []
            for asin, info in metadata.items():
                product_desc = format_product_description(asin, info)
                products_text.append(product_desc)
            
            products_text = "\n\n".join(products_text)
            
            messages = [{
                "role": "user",
                "content": search_prompt.format(
                    products=products_text,
                    question=qa['question']
                )
            }]
            
            response = get_completion(messages)
            short_answer = parse_search_response(response)
            result = {
                "type": "search_qa",
                "short_answer": short_answer,
                "long_answer": ""  # 搜索类型不需要长答案
            }
        else:
            messages = [{
                "role": "user",
                "content": qa_prompt.format(question=qa['question'])
            }]
            
            response = get_completion(messages)
            short_answer, long_answer = parse_qa_response(response)
            
            result = {
                "type": "general_qa",
                "short_answer": short_answer or "",
                "long_answer": long_answer or ""
            }
        
        results.append(result)
    
    return results

def main():
    # 设置目录
    base_dir = "."  # 根据实际情况修改
    test_dir = os.path.join(base_dir, "test")
    output_dir = os.path.join(base_dir, "gpt35_predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试集类别
    test_categories = [
        'all_pans', 'camera_cases', 'leggings',
        'motherboards', 'rifle_scopes', 'rollerball_pens'
    ]
    
    # 处理每个类别
    for category in test_categories:
        print(f"\n开始处理类别: {category}")
        metadata_path = os.path.join(test_dir, category, 'metadata.json')
        
        results = process_category(category, test_dir, metadata_path)
        
        # 保存预测结果
        output_path = os.path.join(output_dir, f"{category}.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"完成类别 {category} 的预测，结果保存至: {output_path}")

if __name__ == "__main__":
    main()