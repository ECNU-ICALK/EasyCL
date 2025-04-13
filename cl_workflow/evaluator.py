import os
import json
import copy
import gc
import torch
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Callable, Union
from dataclasses import asdict
from llamafactory.eval.evaluator import Evaluator
from .cl_eval import CLEvalEvaluator
from llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    FinetuningArguments
)
from easycl.hparams import CLEvaluationArguments
from llamafactory.extras import logging

logger = logging.get_logger(__name__)


class CLEvaluator:
    """持续学习评估器类"""
    
    @staticmethod
    def calculate_transfer(results: Dict[str, Dict[str, float]]) -> float:
        """计算迁移能力指标"""
        if not results:
            return 0.0
        accuracies = [task_result["accuracy"] for task_result in results.values()]
        return sum(accuracies) / len(accuracies)

    @staticmethod
    def calculate_bwt(results: Dict[str, Dict[str, float]], tasks: List[str]) -> float:
        """计算向后迁移指标"""
        if len(tasks) <= 1:
            return 0.0
        bwt = 0.0
        n = len(tasks)
        for i in range(n - 1):
            task = tasks[i]
            if task in results:
                bwt += results[task]["accuracy"]
        return bwt / (n - 1) if n > 1 else 0.0

    @staticmethod
    def calculate_fwt(results: Dict[str, Dict[str, float]], tasks: List[str]) -> float:
        """计算向前迁移指标"""
        if len(tasks) <= 1:
            return 0.0
        fwt = 0.0
        n = len(tasks)
        for i in range(1, n):
            task = tasks[i]
            if task in results:
                fwt += results[task]["accuracy"]
        return fwt / (n - 1) if n > 1 else 0.0

    def __init__(self, args: Tuple[ModelArguments, DataArguments, CLEvaluationArguments, FinetuningArguments]):
        """初始化持续学习评估器"""
        self.model_args, self.data_args, self.cl_eval_args, self.finetuning_args = args
        self.args_dict = {
            **asdict(self.model_args),
            **asdict(self.data_args),
            **asdict(self.cl_eval_args),
            **asdict(self.finetuning_args)
        }
        self.tasks = self.cl_eval_args.cl_tasks.split(",") if self.cl_eval_args.cl_tasks else []
        self.dataset_options = self._load_dataset_options()
        
        # 处理多adapter模式
        self.using_multi_adapter = self.cl_eval_args.eval_mode == "multi_adapter"

    def _load_dataset_options(self) -> Dict:
        """加载数据集选项配置"""
        if self.cl_eval_args.dataset_options:
            options_path = self.cl_eval_args.dataset_options
        else:
            options_path = os.path.join("./data", "dataset_options.json")

        if os.path.exists(options_path):
            with open(options_path, "r", encoding="utf-8") as f:
                dataset_options = json.load(f)
        else:
            raise ValueError(f"数据集选项配置文件未找到：{options_path}")

        # 验证所有任务都有对应的配置
        for task in self.tasks:
            if task not in dataset_options:
                raise ValueError(f"任务 {task} 在 dataset_options 中没有找到对应的配置")

        return dataset_options

    def _run_abscl_selector(self, task: str, dataset_path: str) -> bool:
        """运行ABSCL选择器为任务数据集选择最合适的Adapters"""
        try:
            # 导入ABSCL选择器模块
            from easycl.cl.abscl.abscl_selector import select_adapter
            logger.info(f"使用ABSCL选择器为任务: {task} 选择最合适的Adapters")
            
            # 创建副本避免修改原始参数
            model_args_copy = copy.deepcopy(self.model_args)
            data_args_copy = copy.deepcopy(self.data_args)
            eval_args_copy = copy.deepcopy(self.cl_eval_args)
            finetuning_args_copy = copy.deepcopy(self.finetuning_args)
            
            # 设置任务名
            eval_args_copy.task = task
            
            # 运行选择器
            select_adapter(
                model_args=model_args_copy,
                data_args=data_args_copy,
                training_args=eval_args_copy,  # 注意：在select_adapter中training_args实际上是eval_args
                finetuning_args=finetuning_args_copy,
                dataset_path=dataset_path,
                multi_adapter_dir=self.cl_eval_args.multi_adapter_dir,
                task_name=task,
                batch_size=self.cl_eval_args.abscl_selector_batch_size
            )
            
            # 检查是否成功生成了配置文件
            config_path = os.path.join(self.cl_eval_args.multi_adapter_dir, "multiadapter_selected_config.json")
            if not os.path.exists(config_path):
                logger.error(f"ABSCL选择器未生成配置文件: {config_path}")
                return False
                
            logger.info(f"ABSCL选择器已成功为任务 {task} 生成配置文件")
            
            # 打印配置文件内容以便确认
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    logger.info(f"配置文件包含 {len(config_data.get('adapters', {}))} 个adapter")
                    for adapter_name, adapter_info in config_data.get('adapters', {}).items():
                        sample_count = len(adapter_info.get('indices', []))
                        logger.info(f"  - Adapter '{adapter_name}': {sample_count} 个样本")
            except Exception as e:
                logger.warning(f"读取配置文件内容时出错: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"运行ABSCL选择器时出错: {str(e)}")
            return False

    def _run_dynamic_conpet_selector(self, task: str, dataset_path: str) -> bool:
        """运行Dynamic ConPet选择器为任务数据集选择最合适的Adapters"""
        try:
            # 导入Dynamic ConPet选择器模块
            from easycl.cl.dynamic_conpet.dynamic_conpet_selector import select_adapter_dynamic_conpet
            logger.info(f"使用Dynamic ConPet选择器为任务: {task} 选择最合适的Adapters")
            
            # 创建副本避免修改原始参数
            model_args_copy = copy.deepcopy(self.model_args)
            data_args_copy = copy.deepcopy(self.data_args)
            eval_args_copy = copy.deepcopy(self.cl_eval_args)
            finetuning_args_copy = copy.deepcopy(self.finetuning_args)
            
            # 设置任务名
            eval_args_copy.task = task
            
            # 运行选择器
            select_adapter_dynamic_conpet(
                model_args=model_args_copy,
                data_args=data_args_copy,
                training_args=eval_args_copy,  # 在选择器中training_args实际上是eval_args
                finetuning_args=finetuning_args_copy,
                dataset_path=dataset_path,
                multi_adapter_dir=self.cl_eval_args.multi_adapter_dir,
                task_name=task,
                batch_size=self.cl_eval_args.dynamic_conpet_selector_batch_size
            )
            
            # 检查是否成功生成了配置文件
            config_path = os.path.join(self.cl_eval_args.multi_adapter_dir, "multiadapter_selected_config.json")
            if not os.path.exists(config_path):
                logger.error(f"Dynamic ConPet选择器未生成配置文件: {config_path}")
                return False
                
            logger.info(f"Dynamic ConPet选择器已成功为任务 {task} 生成配置文件")
            
            # 打印配置文件内容以便确认
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    logger.info(f"配置文件包含 {len(config_data.get('adapters', {}))} 个adapter")
                    for adapter_name, adapter_info in config_data.get('adapters', {}).items():
                        sample_count = len(adapter_info.get('indices', []))
                        logger.info(f"  - Adapter '{adapter_name}': {sample_count} 个样本")
            except Exception as e:
                logger.warning(f"读取配置文件内容时出错: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"运行Dynamic ConPet选择器时出错: {str(e)}")
            return False

    def evaluate_model(self, model_args_dict: Dict[str, Any]) -> Dict[str, Any]:
        """评估模型在所有任务上的表现"""
        results = {}
        
        for task in self.tasks:
            # 为每个任务创建新的评估器实例前进行深拷贝
            eval_args = copy.deepcopy(model_args_dict)

            # 更新任务特定参数
            eval_args["task"] = task
            task_config = self.dataset_options[task]

            # 为每个任务创建独立的保存目录
            task_save_dir = os.path.join(eval_args["save_dir"], task)
            os.makedirs(task_save_dir, exist_ok=True)
            eval_args["save_dir"] = task_save_dir

            logger.info(f"正在评估任务: {task}")
            logger.info(f"任务结果将保存在: {task_save_dir}")
            
            # 确保multi_adapter_dir正确传递
            if self.using_multi_adapter:
                logger.info(f"使用多Adapter模式评估任务: {task}")
                # 确保multi_adapter_dir参数被传递
                if "multi_adapter_dir" not in eval_args and hasattr(self.cl_eval_args, "multi_adapter_dir"):
                    eval_args["multi_adapter_dir"] = self.cl_eval_args.multi_adapter_dir
                    
                multi_adapter_dir = eval_args.get("multi_adapter_dir", self.cl_eval_args.multi_adapter_dir)
                logger.info(f"多Adapter配置目录: {multi_adapter_dir}")
                
            # 修复：确保adapter_name_or_path是字符串格式
            if "adapter_name_or_path" in eval_args and isinstance(eval_args["adapter_name_or_path"], list):
                eval_args["adapter_name_or_path"] = ",".join(eval_args["adapter_name_or_path"])
                logger.info(f"已将adapter_name_or_path从列表转换为逗号分隔的字符串: {eval_args['adapter_name_or_path']}")

            # 将数据集选项配置保存为临时文件
            task_options = {
                task: {
                    "options": task_config["options"],
                    "description": task_config["description"]
                }
            }
            task_options_path = os.path.join(task_save_dir, f"{task}_options.json")
            with open(task_options_path, "w", encoding="utf-8") as f:
                json.dump(task_options, f, indent=2, ensure_ascii=False)

            # 设置评估参数：使用临时文件路径
            eval_args["dataset_options"] = task_options_path
            
            # 获取测试集路径，用于可能的选择器
            task_name = task.split("_")[0]
            dataset_path = os.path.join("./data", f"{task_name}_test.json")
            
            # 多adapter模式下，运行选择器
            if self.using_multi_adapter:
                if not os.path.exists(dataset_path):
                    logger.warning(f"未找到任务 {task} 的测试集文件: {dataset_path}，无法运行选择器")
                else:
                    selector_run_attempted = False
                    selector_success = False # Default to failure unless a selector runs successfully

                    # 检查 Dynamic ConPet (如果启用)
                    if self.cl_eval_args.use_dynamic_conpet_selector:
                        selector_run_attempted = True
                        logger.info(f"为任务 {task} 运行Dynamic ConPet选择器...")
                        selector_success = self._run_dynamic_conpet_selector(task, dataset_path)
                        if not selector_success:
                            # 如果 Dynamic ConPet 失败，立即报错停止
                            raise ValueError(f"任务 {task} 的 Dynamic ConPet 选择器运行失败。评估已中止。")
                        else:
                            logger.info(f"Dynamic ConPet选择器成功运行，任务 {task} 将使用选择的Adapters进行评估")

                    # 仅当 Dynamic ConPet 未启用 且 ABSCL 启用时，才检查 ABSCL
                    elif self.cl_eval_args.use_abscl_selector:
                        selector_run_attempted = True
                        logger.info(f"为任务 {task} 运行ABSCL选择器...")
                        selector_success = self._run_abscl_selector(task, dataset_path)
                        if not selector_success:
                            # 如果 ABSCL 失败，立即报错停止
                            raise ValueError(f"任务 {task} 的 ABSCL 选择器运行失败。评估已中止。")
                        else:
                            logger.info(f"ABSCL选择器成功运行，任务 {task} 将使用选择的Adapters进行评估")

                    # 如果启用了多Adapter模式，但没有启用任何选择器
                    if not selector_run_attempted:
                         logger.warning(f"多Adapter模式已启用，但没有为任务 {task} 配置或启用任何选择器。将继续执行，但这可能不是预期行为。")

                    # 确保multi_adapter_dir被包含在评估参数中
                    eval_args["multi_adapter_dir"] = self.cl_eval_args.multi_adapter_dir

            # 判断是否是自定义数据集还是标准数据集
            if task_name in ["mmlu", "cmmlu", "ceval"]:
                # 标准数据集使用原始评估器
                logger.info(f"使用标准评估器评估任务: {task}")
                evaluator = Evaluator(eval_args)
                evaluator.eval()
            else:
                # 自定义数据集使用持续学习评估器
                logger.info(f"使用持续学习评估器评估任务: {task}")
                evaluator = CLEvalEvaluator(eval_args)
                
                # 确保在多adapter模式下设置正确的属性
                if self.using_multi_adapter:
                    logger.info(f"任务 {task} 启用多adapter模式评估")
                    if not hasattr(evaluator, "using_multi_adapter") or not evaluator.using_multi_adapter:
                        logger.warning("CLEvalEvaluator的多adapter模式未正确启用，尝试手动启用")
                        evaluator.using_multi_adapter = True
                        
                    # 确保multi_adapter_dir被正确传递
                    if not hasattr(evaluator.eval_args, "multi_adapter_dir") and "multi_adapter_dir" in eval_args:
                        evaluator.eval_args.multi_adapter_dir = eval_args["multi_adapter_dir"]
                        logger.info(f"手动设置evaluator的multi_adapter_dir: {evaluator.eval_args.multi_adapter_dir}")
                
                # 运行评估
                evaluator.evaluate_custom_dataset()

            # 读取评估结果
            results_path = os.path.join(task_save_dir, "results.json")
            with open(results_path, "r") as f:
                results[task] = json.load(f)

            # 清理临时文件
            os.remove(task_options_path)

            # 显式删除 evaluator 并释放内存
            logger.info(f"任务 {task} 评估完成，释放资源")
            del evaluator
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
        
        return results

    def run(self) -> None:
        """运行持续学习评估"""
        # 创建主保存目录
        os.makedirs(self.cl_eval_args.save_dir, exist_ok=True)

        # 记录评估模式
        logger.info(f"评估模式: {self.cl_eval_args.eval_mode}")
        if self.cl_eval_args.eval_mode == "multi_adapter":
            logger.info(f"多Adapter目录: {self.cl_eval_args.multi_adapter_dir}")
            
            # 检查多Adapter配置目录是否存在
            if not os.path.exists(self.cl_eval_args.multi_adapter_dir):
                logger.info(f"创建多Adapter配置目录: {self.cl_eval_args.multi_adapter_dir}")
                os.makedirs(self.cl_eval_args.multi_adapter_dir, exist_ok=True)

        if self.cl_eval_args.eval_mode == "single":
            # 单模型评估
            logger.info("开始单模型评估")
            base_results = self.evaluate_model(self.args_dict)
            final_results = {
                "individual_results": base_results
            }
        elif self.cl_eval_args.eval_mode == "multi_adapter":
            # 多adapter模式评估
            logger.info("开始多Adapter模式评估")
            multi_adapter_args = copy.deepcopy(self.args_dict)
            # 确保multi_adapter_dir被包含在参数中
            if "multi_adapter_dir" not in multi_adapter_args:
                multi_adapter_args["multi_adapter_dir"] = self.cl_eval_args.multi_adapter_dir
            results = self.evaluate_model(multi_adapter_args)
            final_results = {
                "multi_adapter_results": results
            }
        else:
            # 基准-微调对比模式
            original_save_dir = self.cl_eval_args.save_dir

            # 评估基准模型
            base_args = copy.deepcopy(self.args_dict)
            base_args["save_dir"] = os.path.join(original_save_dir, "base_model")
            os.makedirs(base_args["save_dir"], exist_ok=True)
            base_args["adapter_name_or_path"] = None
            base_model_results = self.evaluate_model(base_args)

            # 评估微调模型
            finetuned_args = copy.deepcopy(self.args_dict)
            finetuned_args["save_dir"] = os.path.join(original_save_dir, "finetuned_model")
            os.makedirs(finetuned_args["save_dir"], exist_ok=True)
            if self.cl_eval_args.cl_tuning_type == "lora":
                if not self.cl_eval_args.compared_adapter_name_or_path:
                    raise ValueError("在 compare 模式下使用 lora 类型时必须提供 compared_adapter_name_or_path")
                finetuned_args["adapter_name_or_path"] = self.cl_eval_args.compared_adapter_name_or_path
            else:  # full_model
                if not self.cl_eval_args.compared_model_name_or_path:
                    raise ValueError("在 compare 模式下使用 full_model 类型时必须提供 compared_model_name_or_path")
                finetuned_args["model_name_or_path"] = self.cl_eval_args.compared_model_name_or_path
            
            finetuned_results = self.evaluate_model(finetuned_args)
            
            # 计算改进程度
            improvements = {}
            for task in self.tasks:
                if task in base_model_results and task in finetuned_results:
                    base_acc = base_model_results[task]["accuracy"]
                    finetuned_acc = finetuned_results[task]["accuracy"]
                    improvements[task] = {
                        "base_accuracy": base_acc,
                        "finetuned_accuracy": finetuned_acc,
                        "absolute_improvement": finetuned_acc - base_acc,
                        "relative_improvement": (finetuned_acc - base_acc) / base_acc * 100 if base_acc > 0 else 0
                    }

            final_results = {
                "base_model_results": base_model_results,
                "finetuned_model_results": finetuned_results,
                "improvements": improvements
            }

        # 添加持续学习指标
        if self.cl_eval_args.calculate_cl_metrics:
            metrics = {}
            if self.cl_eval_args.eval_mode == "single":
                results_to_analyze = final_results.get("individual_results", {})
            elif self.cl_eval_args.eval_mode == "multi_adapter":
                results_to_analyze = final_results.get("multi_adapter_results", {})
            else:  # compare模式
                results_to_analyze = final_results.get("finetuned_model_results", {})
                
            # Calculate all metrics if calculate_cl_metrics is True
            metrics["transfer"] = self.calculate_transfer(results_to_analyze)
            metrics["bwt"] = self.calculate_bwt(results_to_analyze, self.tasks)
            metrics["fwt"] = self.calculate_fwt(results_to_analyze, self.tasks)
            final_results["cl_metrics"] = metrics

        # 保存最终结果
        results_path = os.path.join(self.cl_eval_args.save_dir, "cl_results.json")
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2)