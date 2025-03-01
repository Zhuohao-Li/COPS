import json
from sentence_transformers import SentenceTransformer
import torch
import os
from tqdm import tqdm
import argparse

class QuestionEmbedder:
    def __init__(self, model_name="Alibaba-NLP/gte-Qwen2-7B-instruct"):
        """Initialize embedder"""
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.max_seq_length = 8192
        # Set tensor parallel size to 8
        os.environ['TP_SIZE'] = '8'
        
    def load_questions(self, jsonl_path, max_pairs=None):
        """
        Load questions and IDs from JSONL file
        Args:
            jsonl_path: Path to JSONL file
            max_pairs: Maximum number of QA pairs to process (None for all)
        """
        questions = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_pairs and i >= max_pairs:
                    break
                try:
                    data = json.loads(line)
                    questions.append({
                        'id': data['id'],
                        'question': data['question'],
                        'asin': data['asin']
                    })
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line")
                except KeyError as e:
                    print(f"Warning: Missing required field {e}")
        return questions

    def embed_questions(self, questions, batch_size=8):
        """Batch embed questions"""
        texts = [q['question'] for q in questions]
        
        # 使用tqdm显示进度
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                prompt_name="query"
            )
            embeddings.extend(batch_embeddings)

        # 创建结果字典，只包含id和embedding
        results = {}
        for q, emb in zip(questions, embeddings):
            results[q['id']] = {
                'embedding': emb.tolist(),
                'asin': q['asin']
            }
            
        return results

    def save_embeddings(self, embeddings, output_path):
        """保存嵌入结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)
        
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Embed questions from JSONL files')
    parser.add_argument('--input-dir', type=str, required=True,
                      help='Directory containing qa.jsonl files')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save embedded outputs')
    parser.add_argument('--max', type=int, default=None,
                      help='Optional: Maximum number of QA pairs to process per file (default: process all)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize embedder
    embedder = QuestionEmbedder()
    
    # Process each qa.jsonl file in input directory
    for filename in os.listdir(args.input_dir):
        if filename.endswith('qa.jsonl'):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, f"embedded_{filename.replace('.jsonl', '.json')}")
            
            print(f"Processing file: {input_path}")
            if args.max:
                print(f"Processing first {args.max} QA pairs")
            
            # Load and embed questions with max_pairs limit
            questions = embedder.load_questions(input_path, max_pairs=args.max)
            embedded_questions = embedder.embed_questions(questions)
            
            # Save results
            embedder.save_embeddings(embedded_questions, output_path)
            
            print(f"Saved embeddings to: {output_path}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main()