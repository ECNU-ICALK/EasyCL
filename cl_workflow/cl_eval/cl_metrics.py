import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class CLMetricsCalculator:
    """持续学习指标计算器

    计算以下指标:
    - Last: 模型在完成所有 N 个任务后的整体表现。
    - Avg: 模型每学完一个新任务后，在已学任务上的平均准确率，再对所有阶段求平均。
    - BWT (Backward Transfer): 新任务学习对旧任务性能的影响（遗忘程度）。
    - FWT (Forward Transfer): 先前学到的知识对新任务学习的正向迁移影响。

    注意: Intransigence (IM) 指标无法计算，因为它需要每个任务单独训练的基线结果(R_k,k*)，
           而当前的评估流程只提供顺序学习的结果。
    """
    
    def __init__(self, tasks=None):
        """初始化持续学习指标计算器
        
        Args:
            tasks: 任务列表，如果为None则在计算时从结果中推断
        """
        self.tasks = tasks
    
    def calculate(self, base_results_dir: str, task_results_dirs: List[str]) -> Dict[str, Any]:
        """计算持续学习指标
        
        Args:
            base_results_dir: 基础模型评估结果目录 (用于 R_0,i)
            task_results_dirs: 每个任务训练后(从1到N)的评估结果目录列表 (用于 R_k,i)
            
        Returns:
            Dict[str, Any]: 计算得到的指标和详细结果，包含 Last, Avg, BWT, FWT。
                           IM 指标因缺少数据无法计算。
        """
        # 加载评估结果
        # base_results 包含 R_0,i
        base_results = self._load_results(base_results_dir)
        # task_results 列表包含 R_1,i, R_2,i, ..., R_N,i
        task_results = [self._load_results(d) for d in task_results_dirs]
        
        num_tasks = len(task_results) # N
        if num_tasks == 0:
             logger.warning("没有找到任务结果，无法计算持续学习指标。")
             return {"error": "No task results found"}

        # 如果没有提供tasks，从结果中推断
        if self.tasks is None:
            self.tasks = []
            # 从base_results和task_results中收集所有任务名称
            # 优先从 task_results 中获取顺序，因为 base_results 可能不全
            for result in task_results:
                 for task_name in result.keys():
                     if task_name not in self.tasks:
                         self.tasks.append(task_name)
            # 确保 base_results 中的任务也被包含
            for task_name in base_results.keys():
                 if task_name not in self.tasks:
                     logger.warning(f"Base result task '{task_name}' not found in task results sequence. Adding it.")
                     self.tasks.append(task_name) # 可能需要调整顺序逻辑
            
            # 如果task_results为空，则从base_results获取
            if not self.tasks and base_results:
                 self.tasks = list(base_results.keys())

        if not self.tasks:
             logger.error("无法确定任务列表，无法计算指标。")
             return {"error": "Could not determine task list"}

        if len(self.tasks) != num_tasks:
             logger.warning(f"推断的任务数量 ({len(self.tasks)}) 与任务结果目录数量 ({num_tasks}) 不匹配。将使用结果目录数量作为任务数 N。")
             # 修正 N 的值以匹配实际的结果文件数
             num_tasks = len(self.tasks) if len(self.tasks) >= len(task_results) else len(task_results)


        logger.info(f"计算持续学习指标: 找到 {num_tasks} 个任务 (N={num_tasks})")
        logger.info(f"任务列表: {self.tasks}")
        
        # --- 准备 R_k,i 矩阵 (N x N) ---
        # R[k-1][i-1] 代表 R_k,i (因为 Python 索引从0开始)
        # N = num_tasks
        R = np.zeros((num_tasks, num_tasks))
        detailed_results_matrix = [[None for _ in range(num_tasks)] for _ in range(num_tasks)]

        for k_idx, result_k in enumerate(task_results): # k 从 1 到 N
            for i_idx, task_i in enumerate(self.tasks): # i 从 1 到 N
                if task_i in result_k:
                    R[k_idx, i_idx] = result_k[task_i].get("accuracy", 0.0)
                    detailed_results_matrix[k_idx][i_idx] = {
                        "accuracy": result_k[task_i].get("accuracy", 0.0),
                        "total": result_k[task_i].get("total", 0),
                        "correct": result_k[task_i].get("correct", 0)
                    }
                else:
                    # 如果缺少结果，填充为 0 并记录警告
                    R[k_idx, i_idx] = 0.0
                    detailed_results_matrix[k_idx][i_idx] = {"accuracy": 0.0, "total": 0, "correct": 0, "missing": True}
                    logger.warning(f"任务 '{task_i}' 的结果在第 {k_idx+1} 个任务的评估结果中缺失。")

        # --- 准备 R_0,i 向量 ---
        R0 = np.zeros(num_tasks)
        base_model_results_details = {}
        for i_idx, task_i in enumerate(self.tasks):
            if task_i in base_results:
                R0[i_idx] = base_results[task_i].get("accuracy", 0.0)
                base_model_results_details[task_i] = {
                    "accuracy": base_results[task_i].get("accuracy", 0.0),
                    "total": base_results[task_i].get("total", 0),
                    "correct": base_results[task_i].get("correct", 0)
                }
            else:
                 R0[i_idx] = 0.0 # 如果基础模型没有评估该任务，设为0
                 base_model_results_details[task_i] = {"accuracy": 0.0, "total": 0, "correct": 0, "missing": True}
                 logger.warning(f"任务 '{task_i}' 的结果在基础模型评估结果中缺失。")

        # --- 创建详细的评估结果汇总 ---
        detailed_results_output = {
            "task_sequence": self.tasks,
            "num_tasks": num_tasks,
            "base_model_results": base_model_results_details, # R_0,i
            "task_results_matrix": { # R_k,i Matrix (k rows, i columns)
                 f"after_task_{k+1}": {
                     self.tasks[i]: detailed_results_matrix[k][i] for i in range(num_tasks)
                 } for k in range(num_tasks)
            },
             "notes": ["IM (Intransigence) metric not calculated due to missing baseline results (R_k,k*)."]
        }

        # --- 计算持续学习指标 ---
        metrics = {}
        details = {}

        if num_tasks > 0:
             # 计算 Last
             metrics["last"], details["last"] = self._calculate_last_with_details(R, self.tasks)
             
             # 计算 Avg
             metrics["avg"], details["avg"] = self._calculate_avg_with_details(R, self.tasks)
        
        if num_tasks > 1:
            # 计算 BWT (Backward Transfer)
            metrics["bwt"], details["bwt"] = self._calculate_bwt_with_details(R, self.tasks)
            
            # 计算 FWT (Forward Transfer)
            metrics["fwt"], details["fwt"] = self._calculate_fwt_with_details(R, R0, self.tasks)
        else:
            # 对于单任务，BWT 和 FWT 无意义
            metrics["bwt"], details["bwt"] = 0.0, {"explanation": "BWT not applicable for a single task.", "value": 0.0}
            metrics["fwt"], details["fwt"] = 0.0, {"explanation": "FWT not applicable for a single task.", "value": 0.0}

        # 添加指标计算的详细信息
        detailed_results_output["metrics"] = {
            "summary": metrics,
            "details": details
        }
        
        # 记录计算得到的指标和详细结果
        logger.info(f"计算得到的持续学习指标: {metrics}")
        logger.info("详细评估结果汇总已创建。")
        
        return detailed_results_output
        
    def _load_results(self, results_dir: str) -> Dict[str, float]:
        """加载评估结果。优先读取 cl_results.json，然后读取子目录的 results.json。"""
        # 检查 cl_results.json 是否存在
        cl_results_path = os.path.join(results_dir, "cl_results.json")
        if os.path.exists(cl_results_path):
            try:
                with open(cl_results_path, "r") as f:
                    data = json.load(f)
                    # 检查 individual_results 是否存在且是字典
                    if "individual_results" in data and isinstance(data["individual_results"], dict):
                        logger.info(f"Loaded results from {cl_results_path}")
                        return data["individual_results"]
                    else:
                        logger.warning(f"'individual_results' key missing or not a dictionary in {cl_results_path}. Falling back to subdirectory search.")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {cl_results_path}. Falling back to subdirectory search.")
            except Exception as e:
                 logger.error(f"Error reading {cl_results_path}: {e}. Falling back to subdirectory search.")

        # 否则尝试读取每个任务子目录下的 results.json
        logger.info(f"cl_results.json not found or invalid in {results_dir}. Searching for results.json in subdirectories.")
        results = {}
        if not os.path.exists(results_dir):
             logger.warning(f"Results directory not found: {results_dir}")
             return results
             
        try:
            for task_dir_name in os.listdir(results_dir):
                task_path = os.path.join(results_dir, task_dir_name)
                if os.path.isdir(task_path):
                    results_path = os.path.join(task_path, "results.json")
                    if os.path.exists(results_path):
                        try:
                            with open(results_path, "r") as f:
                                task_result = json.load(f)
                                # 使用任务名作为键
                                task_name = task_result.get("task")
                                if task_name:
                                    # 确保返回的是包含 accuracy, total, correct 的字典
                                    results[task_name] = {
                                         "accuracy": task_result.get("accuracy", 0.0),
                                         "total": task_result.get("total", 0),
                                         "correct": task_result.get("correct", 0)
                                    }
                                    logger.debug(f"Loaded result for task '{task_name}' from {results_path}")
                                else:
                                     logger.warning(f"'task' key missing in {results_path}")
                        except json.JSONDecodeError:
                            logger.error(f"Error decoding JSON from {results_path}")
                        except Exception as e:
                             logger.error(f"Error reading {results_path}: {e}")
        except FileNotFoundError:
             logger.error(f"Could not list contents of directory: {results_dir}")
        except Exception as e:
             logger.error(f"Error listing directory {results_dir}: {e}")
             
        if not results:
            logger.warning(f"No valid results found in subdirectories of {results_dir} either.")
            
        return results

    def _calculate_last_with_details(self, R: np.ndarray, tasks: List[str]) -> Tuple[float, Dict]:
        """计算 Last 指标并返回详细信息
        Formula: Last = (1/N) * sum(R_N,i) for i=1 to N
        """
        num_tasks = R.shape[0] # N
        if num_tasks == 0:
            return 0.0, {"explanation": "No tasks results available.", "value": 0.0}
            
        last_accuracies = R[num_tasks - 1, :] # R_N,i for all i
        last_metric = np.mean(last_accuracies)
        
        details = {
            "explanation": "Average accuracy across all tasks after learning the final task.",
            "value": last_metric,
            "formula": "Last = (1/N) * sum(R_N,i)",
            "accuracies_after_last_task": {tasks[i]: acc for i, acc in enumerate(last_accuracies)},
            "num_tasks": num_tasks
        }
        return last_metric, details

    def _calculate_avg_with_details(self, R: np.ndarray, tasks: List[str]) -> Tuple[float, Dict]:
        """计算 Avg 指标并返回详细信息
        Formula: Avg = (1/N) * sum_{k=1 to N} [ (1/k) * sum_{i=1 to k} (R_k,i) ]
        """
        num_tasks = R.shape[0] # N
        if num_tasks == 0:
            return 0.0, {"explanation": "No tasks results available.", "value": 0.0}
            
        avg_acc_per_step = []
        detailed_steps = {}
        for k in range(num_tasks): # k from 0 to N-1 (representing task 1 to N)
            # Accuracies on tasks 1 to k+1 after learning task k+1
            current_step_accuracies = R[k, :k+1] # R_{k+1, i} for i=1 to k+1
            avg_acc_k = np.mean(current_step_accuracies)
            avg_acc_per_step.append(avg_acc_k)
            detailed_steps[f"after_task_{k+1}"] = {
                 "avg_accuracy": avg_acc_k,
                 "accuracies": {tasks[i]: acc for i, acc in enumerate(current_step_accuracies)}
            }

        avg_metric = np.mean(avg_acc_per_step)
        
        details = {
            "explanation": "Average of the mean accuracy across learned tasks at each step.",
            "value": avg_metric,
            "formula": "Avg = (1/N) * sum_{k=1 to N} [ (1/k) * sum_{i=1 to k} (R_k,i) ]",
            "average_accuracy_at_each_step": avg_acc_per_step,
            "details_at_each_step": detailed_steps,
            "num_tasks": num_tasks
        }
        return avg_metric, details

    def _calculate_bwt_with_details(self, R: np.ndarray, tasks: List[str]) -> Tuple[float, Dict]:
        """计算 BWT (Backward Transfer) 指标并返回详细信息
        Formula: BWT = (1/(N-1)) * sum(R_N,i - R_i,i) for i=1 to N-1
        """
        num_tasks = R.shape[0] # N
        if num_tasks <= 1:
            return 0.0, {"explanation": "BWT requires at least 2 tasks.", "value": 0.0}
            
        bwt_values = []
        details_per_task = {}
        for i in range(num_tasks - 1): # i from 0 to N-2 (representing task 1 to N-1)
            r_ni = R[num_tasks - 1, i] # R_N,i+1
            r_ii = R[i, i]             # R_i+1,i+1
            diff = r_ni - r_ii
            bwt_values.append(diff)
            details_per_task[tasks[i]] = {
                "accuracy_after_learning_task_i": r_ii,
                "accuracy_after_learning_all_tasks": r_ni,
                "backward_transfer": diff
            }

        bwt_metric = np.mean(bwt_values)
        
        details = {
            "explanation": "Average difference between accuracy after learning all tasks and accuracy just after learning the task (measures forgetting/transfer from later tasks).",
            "value": bwt_metric,
            "formula": "BWT = (1/(N-1)) * sum(R_N,i - R_i,i)",
            "per_task_details": details_per_task,
            "num_tasks_compared": num_tasks - 1
        }
        return bwt_metric, details
        
    def _calculate_fwt_with_details(self, R: np.ndarray, R0: np.ndarray, tasks: List[str]) -> Tuple[float, Dict]:
        """计算 FWT (Forward Transfer) 指标并返回详细信息
        Formula: FWT = (1/(N-1)) * sum(R_{i-1},i - R_0,i) for i=2 to N
        """
        num_tasks = R.shape[0] # N
        if num_tasks <= 1:
            return 0.0, {"explanation": "FWT requires at least 2 tasks.", "value": 0.0}
            
        fwt_values = []
        details_per_task = {}
        for i in range(1, num_tasks): # i from 1 to N-1 (representing task 2 to N)
            r_prev_i = R[i - 1, i] # R_i, i+1
            r_0i = R0[i]           # R_0, i+1
            diff = r_prev_i - r_0i
            fwt_values.append(diff)
            details_per_task[tasks[i]] = {
                 "accuracy_on_base_model": r_0i,
                 "accuracy_after_learning_previous_tasks": r_prev_i,
                 "forward_transfer": diff
            }
            
        fwt_metric = np.mean(fwt_values)
        
        details = {
            "explanation": "Average difference between accuracy after learning previous tasks and accuracy on the base model (measures knowledge transfer to new tasks).",
            "value": fwt_metric,
            "formula": "FWT = (1/(N-1)) * sum(R_{i-1},i - R_0,i)",
            "per_task_details": details_per_task,
            "num_tasks_compared": num_tasks - 1
        }
        return fwt_metric, details 