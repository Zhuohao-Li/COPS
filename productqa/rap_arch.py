import json
import numpy as np
import random
from typing import Dict, List, Tuple
from openai import OpenAI

# -------------------------------
# 1. LLM Client: 使用真实的 OpenAI 接口
# -------------------------------
class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://172.17.0.21:50000/v1",
            api_key="None"
        )
    
    def generate(self, prompt: str) -> str:
        try:
            print(f"生成提示词: {prompt[:100]}...")  # 打印提示词的前100个字符
            response = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
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

# -------------------------------
# 2. Environment 环境 (Mock 部分)
# -------------------------------
class RAPEnvironment:
    """
    这里仅对 generate_input 和 evaluate_output 做简单的 Mock。
    其余逻辑(对 LLM 的调用)都在 LLMClient 中真实执行。
    """
    def __init__(self):
        self.success_threshold = 60.0
        self.llm = LLMClient()
        self.current_task_description = ""
        self.qa_data = []
        self.eval_results = []
        
        # Load QA data
        with open("./sample/all_pans/qa.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.qa_data.append(json.loads(line))

    def generate_input(self, task_id: int) -> str:
        """Get question from qa.jsonl"""
        if task_id >= len(self.qa_data):
            return ""
        
        qa_item = self.qa_data[task_id]
        self.current_task_description = qa_item["question"]
        print(f"[Environment] Task input: {self.current_task_description}")
        return self.current_task_description

    def evaluate_output(self, text: str) -> Tuple[bool, float]:
        """Compare RAP's response with ground truth short_answer"""
        qa_item = self.qa_data[len(self.eval_results)]  # Get current QA item
        
        # Get ground truth
        ground_truth = qa_item["short_answer"]
        if isinstance(ground_truth, list):  # Handle search_qa type answers
            ground_truth = [item["title"] for item in ground_truth]
            ground_truth = " | ".join(ground_truth)
        
        # Clean and normalize responses
        text = text.lower().strip()
        ground_truth = str(ground_truth).lower().strip()
        
        # Calculate similarity score
        if text == ground_truth:
            score = 100.0
        else:
            # Simple word overlap score
            text_words = set(text.split())
            truth_words = set(ground_truth.split())
            if len(truth_words) == 0:
                score = 0.0
            else:
                overlap = len(text_words.intersection(truth_words))
                score = (overlap / len(truth_words)) * 100

        success = (score >= self.success_threshold)
        
        # Save evaluation result
        result = {
            "task_id": len(self.eval_results),
            "question": qa_item["question"],
            "rap_response": text,
            "ground_truth": ground_truth,
            "score": round(score, 1),
            "success": success
        }
        self.eval_results.append(result)
        
        # Write to file
        with open("./sample/all_pans/eval_res_rap_arch/evaluation_results.jsonl", "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")
            
        print(f"[Environment] Evaluating output: {text[:100]}...")
        print(f"[Environment] Ground truth: {ground_truth[:100]}...")
        print(f"[Environment] Score: {score:.1f}, Success: {success}")
        
        return success, round(score, 1)

# -------------------------------
# 3. 记忆库 RAPMemory
# -------------------------------
class RAPMemoryEntry:
    """
    存储一次完整的任务及其执行信息
    """
    def __init__(self, task_desc: str, plan: str, output: str,
                 success: bool, score: float, tags: List[str] = None):
        self.task_desc = task_desc
        self.plan = plan
        self.output = output
        self.success = success
        self.score = score
        self.tags = tags or []

    def __repr__(self):
        return f"<RAPMemoryEntry task='{self.task_desc[:15]}...' " \
               f"score={self.score} success={self.success}>"


class RAPMemory:
    """
    用于存储和检索过往经验的记忆库
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.entries: List[RAPMemoryEntry] = []

    def add_entry(self, entry: RAPMemoryEntry):
        # 如果超过大小，先简单移除最早的经验
        if len(self.entries) >= self.max_size:
            self.entries.pop(0)
        self.entries.append(entry)

    def retrieve(self, query: str, top_k=3) -> List[RAPMemoryEntry]:
        """
        在这里根据 query（例如检索关键词等）从记忆库中找最相似的几条。
        这里提供一个简化的基于Jaccard相似度的例子。
        """
        query_set = set(query.split())
        scored_entries = []
        for entry in self.entries:
            entry_set = set(entry.plan.split() + entry.task_desc.split())
            intersection = len(query_set.intersection(entry_set))
            union = len(query_set.union(entry_set)) or 1
            sim_score = intersection / union
            scored_entries.append((entry, sim_score))
        # 按相似度排序
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored_entries[:top_k]]

# -------------------------------
# 4. 推理模块 Reasoner
# -------------------------------
class RAPReasoner:
    """
    根据当前任务产出【整体计划】和【检索关键词】等
    （此处仅做示例的静态实现，也可让LLM生成）
    """
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_plan_and_keywords(self, task: str) -> Tuple[str, str]:
        """
        你也可以在这里调用 LLM 来生成更复杂的plan/keywords。
        此处直接给出示例静态内容。
        """
        plan = "you can think about the task principle and use the keywords to search the memory"
        keywords = "brand, reviews, features, product, specification"
        return plan, keywords

# -------------------------------
# 5. 检索模块 Retriever
# -------------------------------
class RAPRetriever:
    """
    基于Reasoner给出的关键词，到Memory中检索最相似的若干条目
    """
    def __init__(self, memory: RAPMemory):
        self.memory = memory

    def retrieve_experiences(self, keywords: str) -> List[RAPMemoryEntry]:
        relevant_entries = self.memory.retrieve(keywords, top_k=3)
        print(f"[Retriever] 检索到 {len(relevant_entries)} 条相关记忆。")
        return relevant_entries

# -------------------------------
# 6. 执行模块 Executor
# -------------------------------
class RAPExecutor:
    """
    根据检索到的经验 + 计划 + 当前任务描述，执行输出。
    """
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def execute(self, task: str, plan: str, retrieved_entries: List[RAPMemoryEntry]) -> str:
        """
        将历史经验融入到prompt里，让LLM生成最终的输出
        """
        prompt = f"""
you are a professional helper to solve those questions：{task}
this is the plan or think strategy you can use to solve the question: {plan}

reference the following experience:
"""
        for i, entry in enumerate(retrieved_entries):
            prompt += f"\n【experience {i}】\n - task description: {entry.task_desc}\n" \
                      f" - past plan: {entry.plan}\n" \
                      f" - past response: {entry.output}\n" \
                      f" - success or not: {entry.success} (score:{entry.score})\n"

        prompt += """
based on the above information, please output the final answer. The final answer should be in the same language as the question, and keep it as simple as possible (for example, make a choice or justify)
"""
        output = self.llm.generate(prompt)
        return output

# -------------------------------
# 7. 整合 Agent
# -------------------------------
class RAPAgent:
    """
    将 Reasoner、Retriever、Executor 三大模块打通的完整 Agent。
    """
    def __init__(self, env: RAPEnvironment, memory: RAPMemory,
                 reasoner: RAPReasoner, retriever: RAPRetriever, executor: RAPExecutor):
        self.env = env
        self.memory = memory
        self.reasoner = reasoner
        self.retriever = retriever
        self.executor = executor

    def solve_task(self, task_input: str) -> Dict:
        """
        1. 调用 Reasoner 生成计划与检索关键字
        2. 调用 Retriever 根据关键词从 Memory 中检索历史经验
        3. 调用 Executor 生成最终输出
        4. 由 Environment 进行评分
        5. 记录到 Memory
        """
        plan, keywords = self.reasoner.generate_plan_and_keywords(task_input)
        retrieved_entries = self.retriever.retrieve_experiences(keywords)
        output = self.executor.execute(task_input, plan, retrieved_entries)

        success, score = self.env.evaluate_output(output)
        # 将本次经验记忆下来
        entry = RAPMemoryEntry(
            task_desc=task_input,
            plan=plan,
            output=output,
            success=success,
            score=score,
            tags=keywords.split()
        )
        self.memory.add_entry(entry)

        return {
            "plan": plan,
            "keywords": keywords,
            "output": output,
            "score": score,
            "success": success
        }

# -------------------------------
# 8. 任务协调器
# -------------------------------
class RAPTaskCoordinator:
    """
    与原先 LATS 的 Coordinator 类似，管理多任务多轮跑
    """
    def __init__(self, env: RAPEnvironment, agent: RAPAgent,
                 total_tasks=5, max_rounds=3):
        self.env = env
        self.agent = agent
        self.total_tasks = total_tasks
        self.max_rounds = max_rounds
        # 记录任务状态
        self.task_states = [{
            "completed": False,
            "best_score": 0.0,
            "attempts": 0
        } for _ in range(total_tasks)]

        self.history = []

    def run_full_evaluation(self) -> List[float]:
        for r in range(1, self.max_rounds + 1):
            print(f"\n=== [RAP] Round {r} ===")
            round_success = 0

            for task_id in range(self.total_tasks):
                if self.task_states[task_id]["completed"]:
                    print(f"[Coordinator] 任务 {task_id} 已完成，跳过。")
                    continue

                print(f"\n----- 处理任务 {task_id} -----")
                task_input = self.env.generate_input(task_id)
                result = self.agent.solve_task(task_input)

                # 结果判定
                self.task_states[task_id]["attempts"] += 1
                if result["success"]:
                    round_success += 1
                    self.task_states[task_id]["completed"] = True
                    self.task_states[task_id]["best_score"] = result["score"]
                    print(f"[Coordinator] 任务 {task_id} 成功完成！得分 {result['score']}")
                else:
                    # 未成功则记录当前最佳
                    if result["score"] > self.task_states[task_id]["best_score"]:
                        self.task_states[task_id]["best_score"] = result["score"]
                    print(f"[Coordinator] 任务 {task_id} 仍未成功，当前最佳得分: {self.task_states[task_id]['best_score']}")

            # 计算当前完成率
            completed = sum(1 for t in self.task_states if t['completed'])
            accuracy = completed / self.total_tasks
            self.history.append(accuracy)
            print(f"\n[Coordinator] Round {r} 小结：")
            print(f" - 新增成功任务数: {round_success}")
            print(f" - 当前总完成率: {accuracy * 100:.1f}%")

            if completed == self.total_tasks:
                print("[Coordinator] 所有任务均已完成，结束循环。")
                break

        return self.history

# -------------------------------
# 9. 运行主程序 (示例)
# -------------------------------
if __name__ == "__main__":
    # 如果你有自己的OpenAI Key，可以通过参数传入
    # env = RAPEnvironment()
    # env.llm = LLMClient(openai_api_key="YOUR-OPENAI-KEY")
    # 也可在外层先设置 openai.api_key = "..."

    env = RAPEnvironment()
    memory = RAPMemory(max_size=30)
    reasoner = RAPReasoner(env.llm)
    retriever = RAPRetriever(memory)
    executor = RAPExecutor(env.llm)
    agent = RAPAgent(env, memory, reasoner, retriever, executor)

    coordinator = RAPTaskCoordinator(env, agent, total_tasks=500, max_rounds=5)
    history = coordinator.run_full_evaluation()

    print("\n=== 最终结果 ===")
    for i, acc in enumerate(history, 1):
        print(f"Round {i} 的正确率: {acc*100:.1f}%")