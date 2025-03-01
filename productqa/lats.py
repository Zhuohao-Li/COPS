import json
from typing import Dict, List, Tuple
import numpy as np
from openai import OpenAI
import random
import os

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://172.17.0.18:40000/v1",
            api_key="None"
        )
    
    def generate(self, prompt: str) -> str:
        try:
            print(f"生成提示词: {prompt[:100]}...")  # 打印提示词的前100个字符
            response = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9
            )
            print(f"LLM 响应: {response.choices[0].message.content[:100]}...")  # 打印响应的前100个字符
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API调用失败: {str(e)}")
            return ""

class GenerationCandidate:
    def __init__(self, content: str, reflection: str = ""):
        self.content = content
        self.score: float = 0.0
        self.reflection = reflection
        self.visits: int = 0

    def __repr__(self):
        return f"Candidate({self.content[:20]}..., score={self.score})"

class LATS_Agent:
    def __init__(self, environment, config: Dict):
        self.env = environment
        self.config = {
            "num_candidates": 5,
            "memory_size": 20,
            "reflection_depth": 3,
            **config
        }
        self.memory: List[GenerationCandidate] = []
        self.reflection_chain: List[str] = []
        
    def run_episode(self, task_input: str) -> Dict:
        print(f"\n=== 开始新任务轮次 ===")
        print(f"任务输入: {task_input}")
        candidates = self._generate_candidates(task_input)
        print(f"生成的候选方案数量: {len(candidates)}")
        
        evaluated = []
        for c in candidates:
            print(f"评估候选方案: {c.content[:50]}...")  # 打印候选方案的前50个字符
            success, score = self.env.evaluate_output(c.content)
            c.score = score
            evaluated.append((c, success))
            print(f"评估结果: 成功={success}, 得分={score}")
        
        successful = [c for c, s in evaluated if s]
        if successful:
            best_candidate = max(successful, key=lambda x: x.score)
            print(f"找到最佳候选方案 (成功): {best_candidate}")
        else:
            best_candidate = max([c for c, _ in evaluated], key=lambda x: x.score, default=None)
            print(f"找到最佳候选方案 (失败): {best_candidate}")

        return {
            "output": best_candidate.content if best_candidate else "",
            "candidates": [c.content for c in candidates],
            "metadata": {
                "scores": [c.score for c in candidates],
                "reflection": best_candidate.reflection if best_candidate else ""
            }
        }

    def backpropagate(self, output: Dict, final_score: float):
        print(f"\n=== 反向传播 ===")
        print(f"最终得分: {final_score}")
        if output["output"]:
            print(f"更新记忆: {output['output'][:50]}...")  # 打印输出的前50个字符
            self._update_memory(
                content=output["output"],
                score=final_score,
                reflection=output["metadata"]["reflection"]
            )
        
        if final_score < self.env.success_threshold:
            print("生成反思...")
            reflection = self._generate_reflection(output["output"], final_score)
            self._update_reflection_chain(reflection)
            print(f"反思内容: {reflection[:100]}...")  # 打印反思的前100个字符

    def _generate_candidates(self, prompt: str) -> List[GenerationCandidate]:
        print("\n=== 生成候选方案 ===")
        memory_samples = self._sample_memory()
        print(f"从记忆中采样: {len(memory_samples)} 个样本")
        
        context = {
            "task": prompt,
            "examples": [f"{m.content}（得分：{m.score}）" for m in memory_samples],
            "reflections": self.reflection_chain[-self.config["reflection_depth"] :]
        }
        print(f"生成上下文: {context}")
        
        try:
            llm_response = self.env.llm.generate(self._build_prompt(context))
            parsed = self._parse_response(llm_response)
            print(f"解析后的候选方案: {parsed}")
            return [GenerationCandidate(c) for c in parsed[:self.config["num_candidates"]]]
        except Exception as e:
            print(f"生成候选时发生错误：{str(e)}")
            return [GenerationCandidate("默认候选方案")]

    def _generate_reflection(self, output: str, score: float) -> str:
        print("\n=== 生成反思 ===")
        prompt = f"""Please analyze the shortcomings of the following output and provide improvement suggestions (current score: {score}/100):
Task requirements: {self.env.current_task_description}
Current output: {output}
Please provide specific suggestions in the following dimensions:
1. Content completeness
2. Logical coherence
3. Format standardization
4. Creativity"""
        print(f"反思提示词: {prompt[:100]}...")  # 打印提示词的前100个字符
        return self.env.llm.generate(prompt)

    def _update_memory(self, content: str, score: float, reflection: str):
        print("\n=== 更新记忆 ===")
        new_entry = GenerationCandidate(content, reflection)
        new_entry.score = score
        print(f"新记忆条目: {new_entry}")
        
        if len(self.memory) >= self.config["memory_size"]:
            print("记忆已满，进行修剪...")
            self.memory.sort(key=lambda x: -x.score)
            self.memory = self.memory[:self.config["memory_size"]//2] + \
                        self.memory[self.config["memory_size"]//2:][::2]
        
        self.memory.append(new_entry)
        print(f"当前记忆大小: {len(self.memory)}")

    def _update_reflection_chain(self, reflection: str):
        print("\n=== 更新反思链 ===")
        self.reflection_chain.append(reflection.strip())
        if len(self.reflection_chain) > self.config["reflection_depth"]:
            print("反思链超出深度限制，进行修剪...")
            self.reflection_chain = self.reflection_chain[-self.config["reflection_depth"] :]
        print(f"当前反思链长度: {len(self.reflection_chain)}")

    def _sample_memory(self) -> List[GenerationCandidate]:
        print("\n=== 从记忆中采样 ===")
        if not self.memory:
            print("记忆为空，返回空列表")
            return []
            
        contents = [c.content for c in self.memory]
        diversity_scores = self._calculate_diversity(contents)
        print(f"多样性得分: {diversity_scores}")
        
        candidates = []
        for idx, c in enumerate(self.memory):
            combined = c.score * 0.7 + diversity_scores[idx] * 0.3
            candidates.append( (c, combined) )
        
        sampled = [c for c, _ in sorted(candidates, key=lambda x: -x[1])[:3]]
        print(f"采样结果: {sampled}")
        return sampled

    def _calculate_diversity(self, documents: List[str]) -> List[float]:
        print("\n=== 计算多样性 ===")
        diversity_scores = []
        for i, doc in enumerate(documents):
            other_docs = documents[:i] + documents[i+1:]
            if not other_docs:
                diversity_scores.append(1.0)
                continue
            
            # 计算Jaccard相似度
            doc_words = set(doc.split())
            similarities = []
            for other_doc in other_docs:
                other_words = set(other_doc.split())
                intersection = len(doc_words.intersection(other_words))
                union = len(doc_words.union(other_words))
                similarities.append(intersection / union if union != 0 else 0)
            
            # 多样性得分为1减去平均相似度
            avg_similarity = np.mean(similarities) if similarities else 0
            diversity_scores.append(1 - avg_similarity)
        
        print(f"多样性得分: {diversity_scores}")
        return diversity_scores

    def _build_prompt(self, context: Dict) -> str:
        print("\n=== 构建提示词 ===")
        prompt = f"""You are a professional content optimization assistant. Please generate {self.config['num_candidates']} optimized versions based on the following requirements:
        
Task description: {context['task']}
Reference examples: {context['examples'] or 'None'}
Historical improvement suggestions: {context['reflections'] or 'None'}

Generation requirements:
1. Strictly use JSON array format
2. Each element should be a string
3. Reflect different optimization strategies
4. Maintain professionalism and accuracy

Example return format:
["Option 1 content", "Option 2 content", ...]"""
        print(f"提示词内容: {prompt[:100]}...")  # 打印提示词的前100个字符
        return prompt

    def _parse_response(self, response: str) -> List[str]:
        print("\n=== 解析响应 ===")
        cleaned = response.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:-3].strip()
        elif cleaned.startswith('```'):
            cleaned = cleaned[3:-3].strip()
        
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                print(f"解析成功: {data}")
                return [str(item) for item in data]
            return [str(data)]
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {str(e)}")
            return [line.strip() for line in cleaned.split('\n') if line.strip()]
        except Exception as e:
            print(f"响应解析异常: {str(e)}")
            return ["响应解析失败"]

class TaskEnvironment:
    def __init__(self):
        self.success_threshold = 0.7
        self.current_task_description = ""
        self.current_task_id = 0  # 添加当前任务ID
        self.llm = LLMClient()
        
        # 只加载前500个QA对
        self.qa_pairs = []
        count = 0
        with open("./sample/all_pans/qa.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if count >= 500:
                    break
                self.qa_pairs.append(json.loads(line))
                count += 1

    def generate_input(self, task_id: int) -> str:
        if task_id >= len(self.qa_pairs):
            raise ValueError(f"任务ID {task_id} 超出QA对范围")
            
        self.current_task_id = task_id  # 更新当前任务ID
        qa_pair = self.qa_pairs[task_id]
        self.current_qa_pair = qa_pair
        self.current_task_description = f"问题：{qa_pair['question']}\n参考答案：{qa_pair['short_answer']}"
        return qa_pair['question']
    
    def evaluate_output(self, output: str) -> Tuple[bool, float]:
        # 比较生成的答案与标准答案
        ground_truth = self.qa_pairs[self.current_task_id]["short_answer"]
        if isinstance(ground_truth, list):  # 处理搜索类问题
            ground_truth = [item["asin"] for item in ground_truth]
            # 检查生成的答案中是否包含正确的asin
            success = any(asin in output for asin in ground_truth)
        else:  # 处理其他类型问题
            # 直接比较答案
            success = ground_truth.lower() in output.lower()
        
        score = 100.0 if success else 0.0
        return success, score

    def save_accuracy(self, round_num, accuracy):
        os.makedirs("./sample/all_pans/eval_res_lats", exist_ok=True)
        with open("./sample/all_pans/eval_res_lats/accuracy.txt", "a") as f:
            f.write(f"Round {round_num}: {accuracy:.3f}\n")

class TaskCoordinator:
    def __init__(self, env: TaskEnvironment, agent: LATS_Agent, total_tasks: int = 100, max_rounds: int = 10):
        self.env = env
        self.agent = agent
        self.total_tasks = total_tasks
        self.max_rounds = max_rounds
        self.task_states = [{
            'completed': False,
            'best_score': 0.0,
            'attempts': 0
        } for _ in range(total_tasks)]
        self.history = []

    def run_full_evaluation(self) -> List[float]:
        for round_num in range(1, self.max_rounds + 1):
            round_success = 0
            
            for task_id in range(self.total_tasks):
                if self.task_states[task_id]['completed']:
                    continue
                
                input_text = self.env.generate_input(task_id)
                result = self.agent.run_episode(input_text)
                final_success, final_score = self.env.evaluate_output(result['output'])
                self.agent.backpropagate(result, final_score)
                
                if final_success:
                    self.task_states[task_id]['completed'] = True
                    round_success += 1
            
            completed = sum(1 for t in self.task_states if t['completed'])
            accuracy = completed / self.total_tasks
            self.history.append(accuracy)
            
            print(f"轮次 {round_num}: 正确率 {accuracy*100:.1f}%")
            self.env.save_accuracy(round_num, accuracy)
            
            if completed == self.total_tasks:
                break
        
        return self.history

if __name__ == "__main__":
    env = TaskEnvironment()
    agent = LATS_Agent(env, {
        "num_candidates": 3,
        "memory_size": 30,
        "reflection_depth": 2
    })
    
    coordinator = TaskCoordinator(env, agent, total_tasks=500, max_rounds=10)
    coordinator.run_full_evaluation()