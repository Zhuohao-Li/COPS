import json
import numpy as np
import random
from typing import Dict, List, Tuple
from openai import OpenAI
import fuzz
import os

# -------------------------------
# 1. LLM Client: Using real OpenAI interface
# -------------------------------
class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://172.17.0.21:50000/v1",
            api_key="None"
        )
    
    def generate(self, prompt: str) -> str:
        try:
            print(f"Generating prompt: {prompt[:100]}...")  # Print first 100 chars of prompt
            response = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9
            )
            print(f"LLM response: {response.choices[0].message.content[:100]}...")  # Print first 100 chars of response
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API call failed: {str(e)}")
            return ""

# -------------------------------
# 2. Environment (Mock part)
# -------------------------------
class RAPEnvironment:
    """
    Only mock generate_input and evaluate_output here.
    Other logic (LLM calls) are executed in real LLMClient.
    """
    def __init__(self):
        self.success_threshold = 60.0
        self.llm = LLMClient()
        self.current_task_description = ""
        
        # Load QA data
        with open("./sample/all_pans/qa.jsonl", "r", encoding="utf-8") as f:
            self.qa_pairs = [json.loads(line) for line in f][:500]  # Only take first 500
        self.eval_results = []  # Store evaluation results
        
    def generate_input(self, task_id: int) -> str:
        """Return question based on task_id"""
        question = self.qa_pairs[task_id]["question"]
        self.current_task_description = f"Please answer the following question: {question}"
        print(f"[Environment] Generated task input: {self.current_task_description}")
        return self.current_task_description

    def evaluate_output(self, text: str) -> Tuple[bool, float]:
        """
        Evaluate if the generated answer matches the ground truth
        """
        print(f"[Environment] Evaluating output (first 100 chars): {text[:100]} ...")
        
        # Get ground truth for current question
        ground_truth = self.qa_pairs[len(self.eval_results)]["short_answer"]
        
        # Ensure text is not None
        if text is None:
            text = ""
            
        # Normalize generated answer (remove whitespace, convert to uppercase)
        text = text.strip().upper()
        
        # Handle different types of ground truth
        if isinstance(ground_truth, list):
            # Normalize each answer
            ground_truth = [str(ans).strip().upper() for ans in ground_truth]
            # Check if generated answer matches any ground truth
            success = text in ground_truth
        elif isinstance(ground_truth, dict):
            # If dictionary, try to get answer text (adjust based on actual dict structure)
            ground_truth = str(ground_truth.get('text', '')).strip().upper()
            success = text == ground_truth
        else:
            # Convert other cases to string
            ground_truth = str(ground_truth).strip().upper()
            success = text == ground_truth
            
        # Calculate score: 100 for perfect match, 0 otherwise
        score = 100.0 if success else 0.0
        
        # Record evaluation result
        result = {
            "generated_answer": text,
            "ground_truth": ground_truth,
            "score": score,
            "success": success
        }
        self.eval_results.append(result)
        
        # Write results to file
        with open("./sample/all_pans/eval_res_rap/evaluation_result.jsonl", "a", encoding="utf-8") as f:
            json.dump({
                "round": len(self.eval_results),
                "task": len(self.eval_results),
                "result": result
            }, f, ensure_ascii=False)
            f.write("\n")
            
        return success, score

# -------------------------------
# 3. Memory RAPMemory
# -------------------------------
class RAPMemoryEntry:
    """
    Store complete task and execution information
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
    Used to store and retrieve past experience memory
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.entries: List[RAPMemoryEntry] = []

    def add_entry(self, entry: RAPMemoryEntry):
        # If exceeds size, simply remove the earliest experience
        if len(self.entries) >= self.max_size:
            self.entries.pop(0)
        self.entries.append(entry)

    def retrieve(self, query: str, top_k=3) -> List[RAPMemoryEntry]:
        """
        Here, retrieve several similar entries from memory based on query (e.g., search keywords).
        Here, a simplified example based on Jaccard similarity is provided.
        """
        query_set = set(query.split())
        scored_entries = []
        for entry in self.entries:
            entry_set = set(entry.plan.split() + entry.task_desc.split())
            intersection = len(query_set.intersection(entry_set))
            union = len(query_set.union(entry_set)) or 1
            sim_score = intersection / union
            scored_entries.append((entry, sim_score))
        # Sort by similarity
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored_entries[:top_k]]

# -------------------------------
# 4. Reasoner
# -------------------------------
class RAPReasoner:
    """
    Generate overall plan and search keywords based on current task
    (This is a static example implementation, but LLM can also generate)
    """
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_plan_and_keywords(self, task: str) -> Tuple[str, str]:
        """
        You can also call LLM to generate more complex plan/keywords here.
        Here, a static example is directly given.
        """
        plan = "Plan: First highlight product features, emphasize discounts, attract user attention..."
        keywords = "Marketing Features Discounts"
        return plan, keywords

# -------------------------------
# 5. Retriever
# -------------------------------
class RAPRetriever:
    """
    Retrieve several entries from Memory based on keywords from Reasoner
    """
    def __init__(self, memory: RAPMemory):
        self.memory = memory

    def retrieve_experiences(self, keywords: str) -> List[RAPMemoryEntry]:
        relevant_entries = self.memory.retrieve(keywords, top_k=3)
        print(f"[Retriever] Retrieved {len(relevant_entries)} related memories.")
        return relevant_entries

# -------------------------------
# 6. Executor
# -------------------------------
class RAPExecutor:
    """
    Execute output based on retrieved experience + plan + current task description
    """
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def execute(self, task: str, plan: str, retrieved_entries: List[RAPMemoryEntry]) -> str:
        """
        Incorporate historical experience into prompt to generate final output
        """
        prompt = f"""
You are a professional marketing copywriting assistant. The current task is: {task}
This is our high-level plan for the task: {plan}

Reference the following successful experience:
"""
        for i, entry in enumerate(retrieved_entries):
            prompt += f"\n【Experience {i}】\n - Task Description: {entry.task_desc}\n" \
                      f" - Past Plan: {entry.plan}\n" \
                      f" - Past Output: {entry.output}\n" \
                      f" - Success: {entry.success} (Score: {entry.score})\n"

        prompt += """
Please synthesize the above information to generate new marketing copy, and try to showcase creativity and attractiveness in the copy.
"""
        output = self.llm.generate(prompt)
        return output

# -------------------------------
# 7. Agent
# -------------------------------
class RAPAgent:
    """
    Complete Agent by connecting Reasoner, Retriever, Executor
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
        1. Call Reasoner to generate plan and search keywords
        2. Call Retriever to retrieve historical experience from Memory based on keywords
        3. Call Executor to generate final output
        4. Environment scores
        5. Record to Memory
        """
        plan, keywords = self.reasoner.generate_plan_and_keywords(task_input)
        retrieved_entries = self.retriever.retrieve_experiences(keywords)
        output = self.executor.execute(task_input, plan, retrieved_entries)

        success, score = self.env.evaluate_output(output)
        # Record this experience to memory
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
# 8. Task Coordinator
# -------------------------------
class RAPTaskCoordinator:
    """
    Similar to the Coordinator in the original LATS, managing multiple tasks and rounds
    """
    def __init__(self, env: RAPEnvironment, agent: RAPAgent,
                 total_tasks=5, max_rounds=3):
        """Initialize the RAPTaskCoordinator"""
        self.env = env
        self.agent = agent
        self.total_tasks = total_tasks
        self.max_rounds = max_rounds
        self.history = []
        
        # Initialize task status
        self.task_states = [{
            "completed": False,
            "best_score": 0.0,
            "attempts": 0
        } for _ in range(total_tasks)]
        
        # Ensure evaluation result directory exists
        os.makedirs("./sample/all_pans/eval_res_rap", exist_ok=True)
        
        # Clear and initialize evaluation result file
        with open("./sample/all_pans/eval_res_rap/evaluation_result.jsonl", "w", encoding="utf-8") as f:
            f.write("")
        
        # Load QA data
        with open("./sample/all_pans/qa.jsonl", "r", encoding="utf-8") as f:
            self.qa_pairs = [json.loads(line) for line in f][:500]  # Only take first 500
        self.eval_results = []  # Store evaluation results
        
    def generate_input(self, task_id: int) -> str:
        """Return the question based on task_id"""
        question = self.qa_pairs[task_id]["question"]
        self.current_task_description = f"Please answer the following question: {question}"
        print(f"[Environment] Generated task input: {self.current_task_description}")
        return self.current_task_description

    def evaluate_output(self, text: str) -> Tuple[bool, float]:
        """
        Evaluate if the generated answer matches the ground truth
        Also calculate similarity between generated text and ground truth
        """
        print(f"[Environment] Evaluating output (first 100 chars): {text[:100]} ...")
        
        # Get ground truth for current question
        ground_truth = self.qa_pairs[len(self.eval_results)]["short_answer"]
        
        # Ensure text is not None
        if text is None:
            text = ""
            
        # Normalize generated answer
        text = text.strip().upper()
        
        # Handle different types of ground truth
        if isinstance(ground_truth, list):
            ground_truth = [str(ans).strip().upper() for ans in ground_truth]
            success = text in ground_truth
            # Calculate similarity with best matching answer
            similarity = max([fuzz.ratio(text, gt) / 100.0 for gt in ground_truth])
        elif isinstance(ground_truth, dict):
            ground_truth = str(ground_truth.get('text', '')).strip().upper()
            success = text == ground_truth
            similarity = fuzz.ratio(text, ground_truth) / 100.0
        else:
            ground_truth = str(ground_truth).strip().upper()
            success = text == ground_truth
            similarity = fuzz.ratio(text, ground_truth) / 100.0
            
        # Calculate score based on similarity
        score = similarity * 100.0
        
        # Record evaluation result
        result = {
            "generated_answer": text,
            "ground_truth": ground_truth,
            "score": score,
            "success": success,
            "similarity": similarity
        }
        self.eval_results.append(result)
        
        # Save detailed results
        with open("./sample/all_pans/eval_res_rap/evaluation_result.jsonl", "a", encoding="utf-8") as f:
            json.dump({
                "round": len(self.history) + 1,
                "task": len(self.eval_results),
                "result": result
            }, f, ensure_ascii=False)
            f.write("\n")
            
        return success, score

    def run_full_evaluation(self) -> List[float]:
        """Run full evaluation process"""
        for r in range(1, self.max_rounds + 1):
            print(f"\n=== [RAP] Round {r} ===")
            round_success = 0
            round_results = []

            for task_id in range(self.total_tasks):
                if self.task_states[task_id]["completed"]:
                    print(f"[Coordinator] Task {task_id} already completed, skipping.")
                    continue

                print(f"\n----- Processing Task {task_id} -----")
                task_input = self.env.generate_input(task_id)
                result = self.agent.solve_task(task_input)
                
                # Record this result
                round_results.append({
                    "task_id": task_id,
                    "success": result["success"],
                    "score": result["score"]
                })

                # Result judgment
                self.task_states[task_id]["attempts"] += 1
                if result["success"]:
                    round_success += 1
                    self.task_states[task_id]["completed"] = True
                    self.task_states[task_id]["best_score"] = result["score"]
                    print(f"[Coordinator] Task {task_id} completed successfully! Score: {result['score']}")
                else:
                    if result["score"] > self.task_states[task_id]["best_score"]:
                        self.task_states[task_id]["best_score"] = result["score"]
                    print(f"[Coordinator] Task {task_id} not yet successful, best score: {self.task_states[task_id]['best_score']}")

            # Calculate current completion rate
            completed = sum(1 for t in self.task_states if t["completed"])
            accuracy = completed / self.total_tasks
            self.history.append(accuracy)
            
            # Record round evaluation result
            round_summary = {
                "round": r,
                "accuracy": accuracy,
                "completed_tasks": completed,
                "total_tasks": self.total_tasks,
                "round_results": round_results
            }
            
            # Write results to evaluation_result.jsonl
            with open("./sample/all_pans/eval_res_rap/evaluation_result.jsonl", "a", encoding="utf-8") as f:
                json.dump(round_summary, f, ensure_ascii=False)
                f.write("\n")
            
            print(f"\n[Coordinator] Round {r} Summary:")
            print(f" - New successful tasks: {round_success}")
            print(f" - Current total accuracy: {accuracy * 100:.1f}%")

            if completed == self.total_tasks:
                print("[Coordinator] All tasks completed, ending evaluation.")
                break

        return self.history

# -------------------------------
# 9. Run main program (example)
# -------------------------------
if __name__ == "__main__":
    # If you have your own OpenAI Key, you can pass it as a parameter
    # env = RAPEnvironment()
    # env.llm = LLMClient(openai_api_key="YOUR-OPENAI-KEY")
    # You can also set openai.api_key = "..." before the outer layer

    env = RAPEnvironment()
    memory = RAPMemory(max_size=30)
    reasoner = RAPReasoner(env.llm)
    retriever = RAPRetriever(memory)
    executor = RAPExecutor(env.llm)
    agent = RAPAgent(env, memory, reasoner, retriever, executor)

    coordinator = RAPTaskCoordinator(env, agent, total_tasks=100, max_rounds=5)
    history = coordinator.run_full_evaluation()

    print("\n=== Final Results ===")
    for i, acc in enumerate(history, 1):
        print(f"Round {i} Accuracy: {acc*100:.1f}%")