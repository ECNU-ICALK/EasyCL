import os
import json
import copy
import random
import numpy as np
import torch
import io
from typing import Any, Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
# Replace original sentence-transformers import
from transformers import AutoModel, AutoTokenizer as HFAutoTokenizer
def debugprint(*args, **kwargs):
    pass

# 分布式训练辅助函数
def is_distributed():
    """检查是否在分布式环境中运行"""
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_rank():
    """获取当前进程在分布式训练中的rank，非分布式环境返回0"""
    if is_distributed():
        return torch.distributed.get_rank()
    return 0

def is_main_process():
    """检查是否为主进程（rank 0）"""
    return get_rank() == 0



def broadcast_object(obj, src=0):
    """
    在分布式环境中广播任意Python对象

    Args:
        obj: 要广播的对象
        src: 源进程的rank

    Returns:
        广播后的对象
    """
    if not is_distributed():
        return obj

    rank = get_rank()
    debugprint(f"进程 rank={rank} 开始广播对象，源进程: {src}")

    # 确保所有进程在开始广播前同步
    torch.distributed.barrier()
    debugprint(f"进程 rank={rank} 在广播前同步完成")

    # 序列化对象
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    data = buffer.getvalue()

    # 广播数据大小
    size = torch.tensor([len(data)], dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")
    torch.distributed.broadcast(size, src=src)
    debugprint(f"进程 rank={rank} 广播数据大小: {size.item()} 字节")

    # 广播数据
    if rank == src:
        # 源进程发送数据
        tensor = torch.ByteTensor(list(data)).to("cuda" if torch.cuda.is_available() else "cpu")
        debugprint(f"进程 rank={rank} (源) 准备发送数据")
    else:
        # 其他进程接收数据
        tensor = torch.empty(size.item(), dtype=torch.uint8, device="cuda" if torch.cuda.is_available() else "cpu")
        debugprint(f"进程 rank={rank} 准备接收 {size.item()} 字节数据")

    # 执行广播操作
    torch.distributed.broadcast(tensor, src=src)
    debugprint(f"进程 rank={rank} 完成数据广播")

    # 反序列化对象
    if rank != src:
        try:
            buffer = io.BytesIO(tensor.cpu().numpy().tobytes())
            obj = torch.load(buffer)
            debugprint(f"进程 rank={rank} 成功反序列化对象")
        except Exception as e:
            debugprint(f"进程 rank={rank} 反序列化对象失败: {str(e)}")
            # 在反序列化失败时，确保不会阻塞其他进程
            torch.distributed.barrier()
            raise

    # 确保所有进程在完成广播后同步
    torch.distributed.barrier()
    debugprint(f"进程 rank={rank} 广播对象完成")

    return obj


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedModel
)

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    FinetuningArguments
)
from easycl.hparams.cl_finetuning_args import CLFinetuningArguments
from llamafactory.data.parser import get_dataset_list

logger = get_logger(__name__)

class SimCSEModel:
    """SimCSE model wrapper for getting text semantic representations"""

    def __init__(self, model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
        rank = get_rank()
        debugprint(f"SimCSEModel 初始化开始，模型名称: {model_name}")
        self.tokenizer = HFAutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        debugprint("SimCSE Tokenizer 和 Model 已加载并设置为评估模式")

        # 在分布式环境中，只在主进程上加载到GPU
        if not is_distributed() or is_main_process():
            # Check CUDA availability
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.device = "cuda"
                debugprint("SimCSE 模型已移动到 CUDA")
            else:
                self.device = "cpu"
                debugprint("SimCSE 模型将使用 CPU")
        else:
            # 非主进程使用CPU
            self.device = "cpu"
            debugprint("非主进程，SimCSE 模型将使用 CPU")

        # 在分布式环境中同步模型参数
        if is_distributed():
            torch.distributed.barrier()
            debugprint(f"进程 rank={rank} 等待 SimCSE 模型初始化同步")

        debugprint("SimCSEModel 初始化完成")

    def encode(self, sentences, batch_size=32, convert_to_numpy=True):
        """Encode input sentences to get their feature representations"""
        debugprint(f"SimCSEModel encode 开始，句子数量: {len(sentences)}, 批次大小: {batch_size}, 转为Numpy: {convert_to_numpy}")
        embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            debugprint(f"处理批次 {i // batch_size + 1}，样本数量: {len(batch_sentences)}")

            # Encode and get attention mask
            inputs = self.tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # Move to appropriate device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            debugprint("输入已移动到设备")

            # No gradient computation
            with torch.no_grad():
                outputs = self.model(**inputs)
                debugprint("模型前向传播完成")

                # Get hidden states corresponding to [CLS]
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                debugprint(f"获取到 [CLS] 嵌入，形状: {cls_embeddings.shape}")

                # Normalize
                cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
                debugprint("嵌入已归一化")

                if convert_to_numpy:
                    cls_embeddings = cls_embeddings.cpu().numpy()
                    debugprint("嵌入已转为 Numpy")

                embeddings.append(cls_embeddings)

        # Merge results from all batches
        if convert_to_numpy:
            final_embeddings = np.vstack(embeddings)
            debugprint(f"SimCSEModel encode 结束，返回 Numpy 嵌入，形状: {final_embeddings.shape}")
            return final_embeddings
        else:
            final_embeddings = torch.cat(embeddings, dim=0)
            debugprint(f"SimCSEModel encode 结束，返回 Torch 嵌入，形状: {final_embeddings.shape}")
            return final_embeddings

class SSR:
    """
    Self-Synthesized Rehearsal (SSR) Implementation

    SSR mitigates catastrophic forgetting by having the model generate its own pseudo-samples for continual learning
    """

    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        finetuning_args: FinetuningArguments,
        cl_finetuning_args: "CLFinetuningArguments"
    ):
        """Initialize SSR method"""
        debugprint(f"SSR 初始化开始")
        debugprint(f"Model Args: {model_args}")
        debugprint(f"Data Args: {data_args}")
        debugprint(f"Finetuning Args: {finetuning_args}")
        debugprint(f"CL Finetuning Args: {cl_finetuning_args}")
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.cl_finetuning_args = cl_finetuning_args

        # SSR specific parameters
        self.use_ssr = cl_finetuning_args.use_ssr
        self.base_model_path = cl_finetuning_args.base_model_path
        self.num_shots = cl_finetuning_args.num_shots
        self.generation_temperature = cl_finetuning_args.generation_temperature
        self.n_clusters = cl_finetuning_args.n_clusters
        self.pseudo_sample_memory = cl_finetuning_args.pseudo_sample_memory
        self.pseudo_samples_dir = cl_finetuning_args.pseudo_samples_dir

        # Set up SimCSE encoder for clustering
        self.sentence_encoder = None
        try:
            # Switch to using SimCSE model
            debugprint("尝试加载 SimCSE 编码器...")
            self.sentence_encoder = SimCSEModel("princeton-nlp/sup-simcse-bert-base-uncased")
            logger.info_rank0("Successfully loaded SimCSE encoder for SSR pseudo-sample diversity selection")
            debugprint("SimCSE 编码器加载成功")
        except Exception as e:
            logger.warning_rank0(f"Failed to load SimCSE encoder: {e}, will use random selection for pseudo-samples")
            debugprint(f"加载 SimCSE 编码器失败: {e}，将使用随机选择")
        debugprint("SSR 初始化完成")

    def setup_base_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load base model for ICL generation
        """
        debugprint("setup_base_model 开始")
        # Create model args copy
        base_model_args = copy.deepcopy(self.model_args)

        # Use specified base model path
        if self.base_model_path:
            debugprint(f"使用指定的基模型路径: {self.base_model_path}")
            base_model_args.model_name_or_path = self.base_model_path
        else:
            debugprint(f"未指定基模型路径，使用原始模型路径: {base_model_args.model_name_or_path}")

        # Remove all adapters, use pure base model for ICL generation
        base_model_args.adapter_name_or_path = None
        debugprint("移除所有适配器，使用纯基模型")

        # Load tokenizer
        debugprint("加载基模型 Tokenizer...")
        tokenizer_module = load_tokenizer(base_model_args)
        tokenizer = tokenizer_module["tokenizer"]
        debugprint("基模型 Tokenizer 加载完成")

        # Load base model - no training, disable adapters
        temp_finetuning_args = copy.deepcopy(self.finetuning_args)
        temp_finetuning_args.finetuning_type = "lora"  # Use lora type but don't load adapters
        debugprint("加载基模型 (不训练，禁用适配器)...")
        base_model = load_model(
            tokenizer,
            base_model_args,
            temp_finetuning_args,
            is_trainable=False
        )
        debugprint("基模型加载完成")

        logger.info_rank0(f"Loaded SSR base model for ICL generation, path: {base_model_args.model_name_or_path}")
        debugprint("setup_base_model 完成")
        return base_model, tokenizer

    def setup_previous_model(self) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """
        Load model from previous task for pseudo-sample refinement
        """
        debugprint("setup_previous_model 开始")
        if not self.cl_finetuning_args.previous_task_model:
            logger.warning_rank0("previous_task_model not specified, will skip pseudo-sample refinement step")
            debugprint("未指定 previous_task_model，跳过伪样本优化步骤")
            return None, None

        # Create model args copy
        prev_model_args = copy.deepcopy(self.model_args)

        # Load adapter or full model from previous task
        if self.finetuning_args.finetuning_type == "lora":
            # If using LoRA, load adapter from previous task
            debugprint(f"LoRA 微调类型，加载上一任务适配器: {self.cl_finetuning_args.previous_task_model}")
            prev_model_args.adapter_name_or_path = [self.cl_finetuning_args.previous_task_model]
        else:
            # If using full-parameter finetuning, load complete model from previous task
            debugprint(f"全参数微调类型，加载上一任务完整模型: {self.cl_finetuning_args.previous_task_model}")
            prev_model_args.model_name_or_path = self.cl_finetuning_args.previous_task_model
            prev_model_args.adapter_name_or_path = None

        # Load tokenizer
        debugprint("加载上一任务模型的 Tokenizer...")
        tokenizer_module = load_tokenizer(prev_model_args)
        tokenizer = tokenizer_module["tokenizer"]
        debugprint("上一任务模型 Tokenizer 加载完成")

        # Load model from previous task - no training
        debugprint("加载上一任务模型 (不训练)...")
        prev_model = load_model(
            tokenizer,
            prev_model_args,
            self.finetuning_args,
            is_trainable=False
        )
        debugprint("上一任务模型加载完成")

        logger.info_rank0(f"Loaded previous task model for pseudo-sample refinement, path: {self.cl_finetuning_args.previous_task_model}")
        debugprint("setup_previous_model 完成")
        return prev_model, tokenizer

    def _find_data_file(self, dataset_name: str, split: str = "train") -> Optional[str]:
        """
        Finds the raw data file for a given dataset name and split, based on dataset_info.json.
        Follows the logic described in finddataset.md.
        """
        debugprint(f"_find_data_file 开始，数据集: {dataset_name}, 拆分: {split}")
        potential_paths = [self.data_args.dataset_dir, os.path.join(os.getcwd(), "data")]
        debugprint(f"潜在的数据目录: {potential_paths}")
        info_path = None
        dataset_info = None
        base_dir_for_file = None

        # Find dataset_info.json
        debugprint("查找 dataset_info.json...")
        for data_dir in potential_paths:
            path = os.path.join(data_dir, "dataset_info.json")
            debugprint(f"检查路径: {path}")
            if os.path.exists(path):
                info_path = path
                base_dir_for_file = data_dir
                debugprint(f"找到 dataset_info.json: {info_path}，基础目录: {base_dir_for_file}")
                break

        if not info_path:
            if split == "train":
                logger.warning_rank0("Cannot find dataset_info.json in provided data_args.dataset_dir or ./data.")
                debugprint("未找到 dataset_info.json，尝试直接查找数据文件...")
                # Attempt to find the file directly based on naming convention
                for data_dir in potential_paths:
                    for ext in [".json", ".jsonl"]:
                        potential_file = os.path.join(data_dir, f"{dataset_name}{ext}")
                        debugprint(f"检查直接文件路径: {potential_file}")
                        if os.path.exists(potential_file):
                            debugprint(f"直接找到数据文件: {potential_file}")
                            return potential_file
                logger.error_rank0(f"Could not find dataset_info.json or a data file for {dataset_name}")
                debugprint(f"错误: 无法找到 dataset_info.json 或 {dataset_name} 的数据文件")
                raise FileNotFoundError("Could not find dataset_info.json or a data file.")
            else:
                debugprint(f"未找到 dataset_info.json，对于非训练拆分 ({split}) 返回 None")
                return None # For other splits, it's okay if info is missing

        try:
            debugprint(f"读取 dataset_info.json: {info_path}")
            with open(info_path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)
            debugprint("dataset_info.json 读取成功")
        except Exception as e:
            logger.error_rank0(f"Error reading dataset_info.json from {info_path}: {e}")
            debugprint(f"读取 dataset_info.json 出错: {e}")
            if split == "train":
                raise
            else:
                 debugprint(f"读取 dataset_info.json 出错，对于非训练拆分 ({split}) 返回 None")
                 return None

        file_name = None
        found_entry = None

        # High priority match
        high_priority_key = f"{dataset_name}_{split}"
        debugprint(f"尝试高优先级匹配键: {high_priority_key}")
        if high_priority_key in dataset_info and "file_name" in dataset_info[high_priority_key]:
            found_entry = dataset_info[high_priority_key]
            file_name = found_entry["file_name"]
            debugprint(f"高优先级匹配成功，文件名: {file_name}")

        # Low priority match
        if not file_name and dataset_name in dataset_info:
            debugprint(f"尝试低优先级匹配键: {dataset_name}")
            entry = dataset_info[dataset_name]
            if entry.get("split") == split and "file_name" in entry:
                found_entry = entry
                file_name = found_entry["file_name"]
                debugprint(f"低优先级匹配成功 (拆分匹配)，文件名: {file_name}")

        if not file_name:
            if split == "train":
                 debugprint("在 dataset_info.json 中未找到文件条目，尝试直接查找文件...")
                 # Fallback: Check if a file named dataset_name.json or .jsonl exists directly
                 for suffix in [".json", ".jsonl"]:
                    potential_file = os.path.join(base_dir_for_file, f"{dataset_name}{suffix}")
                    debugprint(f"检查直接文件路径 (回退): {potential_file}")
                    if os.path.exists(potential_file):
                         logger.info_rank0(f"Found data file directly without explicit dataset_info.json entry: {potential_file}")
                         debugprint(f"直接找到数据文件 (回退): {potential_file}")
                         return potential_file
                 logger.error_rank0(f"Cannot find file entry for {dataset_name} (split: {split}) in {info_path}")
                 debugprint(f"错误: 在 {info_path} 中找不到 {dataset_name} (拆分: {split}) 的文件条目")
                 raise ValueError(f"Cannot find file entry for {dataset_name} (split: {split}) in {info_path}")
            else:
                debugprint(f"在 dataset_info.json 中未找到 {dataset_name} (拆分: {split}) 的文件条目，返回 None")
                return None # Okay for non-train split

        # Construct and validate path
        full_path = os.path.join(base_dir_for_file, file_name)
        debugprint(f"构造完整路径: {full_path}")
        if os.path.exists(full_path):
            debugprint(f"文件存在: {full_path}，返回此路径")
            return full_path

        # Fallback check in ./data if dataset_dir was different
        fallback_data_dir = os.path.join(os.getcwd(), "data")
        if base_dir_for_file != fallback_data_dir:
             fallback_path = os.path.join(fallback_data_dir, file_name)
             debugprint(f"尝试回退路径: {fallback_path}")
             if os.path.exists(fallback_path):
                 debugprint(f"文件在回退路径存在: {fallback_path}，返回此路径")
                 return fallback_path
             else:
                 debugprint(f"文件在回退路径不存在: {fallback_path}")

        if split == "train":
            logger.error_rank0(f"Data file '{file_name}' not found at '{full_path}' or fallback paths.")
            debugprint(f"错误: 数据文件 '{file_name}' 在 '{full_path}' 或回退路径中未找到")
            raise FileNotFoundError(f"Data file '{file_name}' not found.")
        else:
            debugprint(f"数据文件 '{file_name}' 未找到，对于非训练拆分 ({split}) 返回 None")
            return None

    def _load_raw_dataset(self, dataset_name: str, split: str = "train") -> List[Dict[str, Any]]:
        """
        Loads raw data from the file specified by dataset_info.json for the given split.
        Returns a list of dictionaries.
        """
        debugprint(f"_load_raw_dataset 开始，数据集: {dataset_name}, 拆分: {split}")
        file_path = self._find_data_file(dataset_name, split)
        if not file_path:
             if split == "train":
                 # _find_data_file should have raised error already
                 logger.error_rank0(f"Could not find data file for {dataset_name} train split.")
                 # No need for debugprint here as _find_data_file already logged the error
                 return []
             else:
                logger.info_rank0(f"No {split} data file found for {dataset_name}.")
                debugprint(f"未找到 {dataset_name} 的 {split} 数据文件，返回空列表")
                return [] # Return empty list if non-train split file not found

        try:
            debugprint(f"尝试从文件加载原始数据: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.endswith(".jsonl"):
                    debugprint("文件格式为 .jsonl")
                    data = [json.loads(line) for line in f if line.strip()]
                elif file_path.endswith(".json"):
                    debugprint("文件格式为 .json")
                    data = json.load(f)
                    # Ensure it's a list of dicts
                    if not isinstance(data, list):
                        debugprint(f"错误: JSON 文件 {file_path} 不包含对象列表")
                        raise ValueError(f"JSON file {file_path} does not contain a list of objects.")
                    if data and not isinstance(data[0], dict):
                         debugprint(f"错误: JSON 文件 {file_path} 列表中的第一个元素不是字典")
                         raise ValueError(f"JSON file {file_path} does not contain a list of objects.")
                else:
                    debugprint(f"错误: 不支持的文件格式: {file_path}")
                    raise ValueError(f"Unsupported file format: {file_path}. Only .json and .jsonl are supported.")
            logger.info_rank0(f"Successfully loaded raw data from {file_path}")
            debugprint(f"成功从 {file_path} 加载了 {len(data)} 条原始数据")
            return data
        except Exception as e:
            logger.error_rank0(f"Error loading raw data from {file_path}: {e}")
            debugprint(f"从 {file_path} 加载原始数据时出错: {e}")
            raise

    def get_few_shot_examples(self, raw_data: List[Dict[str, Any]], num_shots: Optional[int] = None):
        """
        Randomly select few-shot examples from a list of raw data dictionaries.
        In distributed environment, ensures all processes select the same examples.
        """
        rank = get_rank()
        debugprint(f"get_few_shot_examples 开始，原始数据量: {len(raw_data)}, 请求样本数: {num_shots or self.num_shots}")
        if num_shots is None:
            num_shots = self.num_shots

        if not raw_data:
            logger.warning_rank0("Received empty raw_data list for few-shot examples.")
            debugprint("警告: 收到空的原始数据列表，无法选择 few-shot 样本")
            return []

        # Ensure dataset has enough samples
        effective_num_shots = min(num_shots, len(raw_data))
        if len(raw_data) < num_shots:
            logger.warning_rank0(f"Raw data sample count ({len(raw_data)}) is less than requested examples ({num_shots}), will use all samples")
            debugprint(f"警告: 原始数据量 ({len(raw_data)}) 小于请求的样本数 ({num_shots})，将使用所有样本")
        else:
            debugprint(f"将从 {len(raw_data)} 个样本中随机选择 {effective_num_shots} 个")

        # 在分布式环境中设置相同的随机种子，确保所有进程选择相同的样本
        if is_distributed():
            # 保存当前随机状态
            old_state = random.getstate()
            # 使用固定种子
            random.seed(42 + effective_num_shots)  # 添加effective_num_shots以避免不同调用选择相同样本
            debugprint(f"进程 rank={rank} 设置固定随机种子用于样本选择")

        # Randomly select samples
        indices = random.sample(range(len(raw_data)), effective_num_shots)
        examples = [raw_data[i] for i in indices]

        # 恢复随机状态
        if is_distributed():
            random.setstate(old_state)
            debugprint(f"进程 rank={rank} 已恢复随机状态")

        debugprint(f"已选择 {len(examples)} 个 few-shot 样本，索引: {indices}")
        debugprint("get_few_shot_examples 完成")
        return examples

    def construct_icl_prompt(self, examples, template_type="alpaca"):
        """
        Construct ICL prompt
        """
        debugprint(f"construct_icl_prompt 开始，样本数量: {len(examples)}, 模板类型: {template_type}")
        instruction = 'Create a task sample following examples below.\n\n'

        for i, example in enumerate(examples):
            debugprint(f"处理第 {i+1} 个样本: {example}")
            # Ensure compatibility with different dataset formats
            if "instruction" in example and "input" in example and "output" in example:
                # Alpaca format
                debugprint("检测到 Alpaca 格式")
                input_text = example["input"]
                input_text = "<noinput>" if not input_text or input_text.lower() == "" else input_text

                instruction += (
                    f"Instruction: {example['instruction']}\n" +
                    f"Input: {input_text}\n" +
                    f"Output: {example['output']}\n\n"
                )
            elif "messages" in example:
                # ShareGPT format
                debugprint("检测到 ShareGPT 格式")
                user_message = ""
                assistant_message = ""

                for msg in example["messages"]:
                    if msg["role"] == "user":
                        user_message = msg["content"]
                    elif msg["role"] == "assistant":
                        assistant_message = msg["content"]

                # Use placeholder instruction if actual instruction isn't clearly defined
                derived_instruction = "Answer the following query"
                # Try to find a system message if available
                system_message = next((msg["content"] for msg in example.get("messages", []) if msg.get("role") == "system"), None)
                if system_message:
                     derived_instruction = system_message # Use system message as instruction if present
                elif user_message:
                    # If no system message, use a generic instruction or potentially part of the user message if it looks like one.
                    # Keeping it simple for now.
                     pass


                input_text = user_message
                input_text = "<noinput>" if not input_text or input_text.lower() == "" else input_text

                instruction += (
                    # f"Instruction: Answer the following query\n" +
                    f"Instruction: {derived_instruction}\n" + # Use derived or system instruction
                    f"Input: {input_text}\n" +
                    f"Output: {assistant_message}\n\n"
                )
            else:
                # Generic format - try best guess
                debugprint("未检测到已知格式，尝试通用格式解析")
                current_instruction = None
                current_input = None
                current_output = None
                try:
                    # Heuristic: Find keys containing 'instruction', 'input'/'query', 'output'/'answer'
                    ins_key = next((k for k in example if 'instruction' in k.lower()), None)
                    inp_key = next((k for k in example if any(s in k.lower() for s in ['input', 'query', 'context', 'question'])), None)
                    out_key = next((k for k in example if any(s in k.lower() for s in ['output', 'answer', 'completion', 'response'])), None)

                    if ins_key and inp_key and out_key:
                        debugprint(f"通用格式启发式匹配: 指令键='{ins_key}', 输入键='{inp_key}', 输出键='{out_key}'")
                        current_instruction = example[ins_key]
                        current_input = example[inp_key]
                        current_output = example[out_key]
                    else:
                         debugprint("通用格式启发式匹配失败，按顺序猜测")
                         # Fallback to order if keys aren't suggestive
                         string_values = [v for v in example.values() if isinstance(v, str)]
                         if len(string_values) >= 3:
                            current_instruction = string_values[0]
                            current_input = string_values[1]
                            current_output = string_values[2]
                         elif len(string_values) == 2: # Assume instruction and output
                             current_instruction = string_values[0]
                             current_input = "" # Assign empty input
                             current_output = string_values[1]

                    if current_instruction is not None and current_output is not None:
                        input_text = current_input if current_input else ""
                        input_text = "<noinput>" if not input_text or input_text.lower() == "" else input_text
                        instruction += (
                             f"Instruction: {current_instruction}\n" +
                             f"Input: {input_text}\n" +
                             f"Output: {current_output}\n\n"
                        )
                    else:
                        debugprint(f"警告: 无法为样本 {i+1} 解析通用格式，跳过此样本的 ICL 提示构建")

                except Exception as e:
                     debugprint(f"警告: 解析通用格式时发生错误 {e}，跳过样本 {i+1}")

        instruction += 'Instruction:'
        debugprint(f"构造完成的 ICL 提示 (部分):{instruction[:50000]}...") # Print first 500 chars
        debugprint("construct_icl_prompt 完成")
        return instruction

    def generate_pseudo_samples(self, num_samples: Optional[int] = None):
        """
        Generate pseudo-samples using base model through ICL, loading raw data internally.
        In distributed environment, only rank 0 generates samples and broadcasts to other processes.
        """
        rank = get_rank()
        debugprint(f"generate_pseudo_samples 开始，请求样本数: {num_samples or self.pseudo_sample_memory}")

        # 只在主进程或非分布式环境中生成伪样本
        if not is_distributed() or is_main_process():
            pseudo_samples = self._generate_pseudo_samples_impl(num_samples)
            debugprint(f"主进程生成了 {len(pseudo_samples)} 个伪样本")
        else:
            # 非主进程初始化为空列表
            pseudo_samples = []
            debugprint(f"非主进程跳过伪样本生成")

        # 在分布式环境中，广播伪样本到所有进程
        if is_distributed():
            # 等待主进程完成生成
            torch.distributed.barrier()
            # 将伪样本从rank 0广播到其他进程
            pseudo_samples = broadcast_object(pseudo_samples, src=0)
            debugprint(f"进程 rank={rank} 接收到广播的伪样本，数量: {len(pseudo_samples)}")

        return pseudo_samples

    def _generate_pseudo_samples_impl(self, num_samples: Optional[int] = None):
        """
        实际生成伪样本的内部实现，只在主进程调用
        """
        debugprint(f"_generate_pseudo_samples_impl 开始，请求样本数: {num_samples or self.pseudo_sample_memory}")
        if num_samples is None:
            num_samples = self.pseudo_sample_memory
            debugprint(f"使用默认伪样本内存大小: {num_samples}")

        # Determine the primary dataset name for loading raw examples
        # We assume the first dataset in the list is the target for few-shot examples
        if not self.data_args.dataset:
             debugprint("错误: 未指定 data_args.dataset")
             raise ValueError("data_args.dataset must be specified to load few-shot examples.")
        current_task_dataset_name = self.data_args.dataset[0] if isinstance(self.data_args.dataset, list) else self.data_args.dataset
        debugprint(f"当前任务数据集名称 (用于 few-shot 样本): {current_task_dataset_name}")

        # Load raw data for the current task
        try:
            debugprint(f"加载 {current_task_dataset_name} 的原始训练数据...")
            raw_train_data = self._load_raw_dataset(current_task_dataset_name, split="train")
        except (FileNotFoundError, ValueError) as e:
             logger.error_rank0(f"Failed to load raw training data for {current_task_dataset_name}: {e}")
             debugprint(f"错误: 加载 {current_task_dataset_name} 的原始训练数据失败: {e}")
             raise

        if not raw_train_data:
             logger.error_rank0(f"No raw training data found for {current_task_dataset_name}. Cannot generate pseudo-samples.")
             debugprint(f"错误: 未找到 {current_task_dataset_name} 的原始训练数据，无法生成伪样本")
             return [] # Return empty list if no raw data

        # Load base model
        debugprint("设置基模型...")
        base_model, tokenizer = self.setup_base_model()
        base_model = base_model.cuda()
        tokenizer.padding_side = "left"
        debugprint("基模型和 Tokenizer 设置完成")

        # Select few-shot examples from raw data
        debugprint("选择 few-shot 样本...")
        few_shot_examples = self.get_few_shot_examples(raw_train_data)
        if not few_shot_examples:
             debugprint("警告: 未能选择任何 few-shot 样本，ICL 提示将为空")

        # Construct ICL prompt
        debugprint("构造 ICL 提示...")
        prompt = self.construct_icl_prompt(few_shot_examples)

        # Set generation config
        generation_config = GenerationConfig(
            temperature=self.generation_temperature,
            top_p=0.95,
            top_k=40,
            num_beams=1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        debugprint(f"生成配置: {generation_config}")

        # Encode prompt
        debugprint("编码提示...")
        # Ensure tokenizer uses left padding for generation
        debugprint(f"Tokenizer padding side set to: {tokenizer.padding_side}")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(base_model.device)
        attention_mask = inputs["attention_mask"].to(base_model.device)
        debugprint(f"提示编码完成，输入形状: {input_ids.shape}")

        # Generate pseudo-samples
        generated_texts = []

        # Generate in batches to avoid GPU OOM
        batch_size = min(5, num_samples) # Force batch size to 1
        num_batches = (num_samples + batch_size - 1) // batch_size
        debugprint(f"开始生成伪样本，总批次数: {num_batches}, 每批大小: {batch_size}")

        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size) # Correct calculation for remaining samples
            if current_batch_size <= 0:
                 debugprint("已生成足够数量的样本，停止生成")
                 break

            logger.info_rank0(f"Generating pseudo-sample batch {i+1}/{num_batches}, size {current_batch_size}")
            debugprint(f"生成伪样本批次 {i+1}/{num_batches}，大小: {current_batch_size}")

            # Generate multiple samples
            with torch.no_grad():
                debugprint("调用 base_model.generate()...")
                outputs = base_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    max_new_tokens=512,
                    num_return_sequences=current_batch_size,
                    return_dict_in_generate=True
                )
                debugprint("base_model.generate() 调用完成")

            # Decode generated text
            debugprint("解码生成的文本...")
            if isinstance(outputs, dict) and "sequences" in outputs:
                 sequences = outputs.sequences
                 # sequences shape: [current_batch_size, sequence_length]
                 debugprint(f"生成序列形状: {sequences.shape}")
                 prompt_length = input_ids.shape[1]

                 for j in range(sequences.shape[0]): # Iterate through batch dimension
                    # Only take newly generated tokens
                    new_tokens = sequences[j][prompt_length:]
                    debugprint(f"样本 {j} 的新 Token 数量: {len(new_tokens)}")

                    # Use batch_decode for efficiency
                    # generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

                    # Add generated text (including prompt initially for parsing logic) to result list
                    # We will decode later or let the parser handle it if needed.
                    # For now, just store the full sequence? Let's try decoding here.
                    full_decoded_text = tokenizer.decode(sequences[j], skip_special_tokens=True)

                    # Alternative: Decode only new tokens and prepend prompt
                    # generated_part = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    # full_decoded_text = prompt + generated_part # Approximate, prompt decoding might differ slightly

                    if full_decoded_text:
                         generated_texts.append(full_decoded_text)
                         debugprint(f"添加第 {len(generated_texts)} 条生成的文本 (部分): {full_decoded_text[:100]}...")
                    else:
                         debugprint(f"警告: 样本 {j} 解码结果为空")
            else:
                debugprint("警告: model.generate 未返回预期的 'sequences' 字典输出")

            # Optional: Add small sleep to prevent potential issues? Unlikely needed.

        debugprint(f"生成循环结束，共获得 {len(generated_texts)} 条原始生成文本")

        # Free GPU memory
        debugprint("释放基模型 GPU 内存...")
        del base_model
        del tokenizer
        del inputs
        del input_ids
        del attention_mask
        if 'outputs' in locals(): del outputs
        torch.cuda.empty_cache()
        debugprint("基模型 GPU 内存已释放")

        # Parse generated text into structured pseudo-samples
        debugprint("开始解析生成的文本...")
        pseudo_samples = self.parse_generated_texts(generated_texts)
        debugprint(f"解析完成，获得 {len(pseudo_samples)} 个结构化伪样本")

        debugprint(f"_generate_pseudo_samples_impl 完成，返回 {len(pseudo_samples)} 个伪样本")
        return pseudo_samples

    def parse_generated_texts(self, generated_texts):
        """
        Parse generated text into structured pseudo-samples
        """
        debugprint(f"parse_generated_texts 开始，待解析文本数量: {len(generated_texts)}")
        parsed_samples = []
        parse_errors = 0

        for idx, text in enumerate(generated_texts):
            # debugprint(f"解析第 {idx+1} 条文本:{text}--------------------") # Can be very verbose
            try:
                # Split generated text to extract instruction, input and output
                # Assume the generated part starts after the last "Output:" marker from the prompt
                # Find the *last* occurrence of "Instruction:" which marks the start of the *generated* sample
                last_instruction_marker = text.rfind("Instruction:")
                if last_instruction_marker == -1:
                     debugprint(f"文本 {idx+1} 未找到 'Instruction:' 标记，跳过")
                     parse_errors += 1
                     continue

                # Extract the part assumed to be the generated sample
                generated_part = text[last_instruction_marker:].strip()
                # debugprint(f"提取的生成部分:{generated_part}--------------------")

                # Now parse this generated part
                parts = generated_part.split("Output:", 1)
                if len(parts) != 2:
                    debugprint(f"文本 {idx+1} 生成部分未能按 'Output:' 分割，跳过")
                    parse_errors += 1
                    continue

                output = parts[1].strip()
                remaining = parts[0].strip()
                # debugprint(f"解析出 Output: {output[:100]}...")

                # Split remaining part to get input
                parts = remaining.split("Input:", 1)
                if len(parts) != 2:
                     debugprint(f"文本 {idx+1} 剩余部分未能按 'Input:' 分割，跳过")
                     parse_errors += 1
                     continue

                input_text = parts[1].strip()
                # debugprint(f"解析出 Input: {input_text[:100]}...")
                instruction_parts = parts[0].strip().split("Instruction:", 1)

                if len(instruction_parts) != 2:
                    debugprint(f"文本 {idx+1} 指令部分未能按 'Instruction:' 分割，跳过")
                    parse_errors += 1
                    continue

                instruction = instruction_parts[1].strip()
                # debugprint(f"解析出 Instruction: {instruction[:100]}...")

                # Handle special tokens
                if input_text == "<noinput>":
                    input_text = ""
                    # debugprint("输入为 <noinput>，置为空字符串")

                # Validate parsed fields
                if not instruction:
                    debugprint(f"文本 {idx+1} 解析出的指令为空，跳过")
                    parse_errors += 1
                    continue
                if not output: # Also check if output is empty
                    debugprint(f"文本 {idx+1} 解析出的输出为空，跳过")
                    parse_errors += 1
                    continue

                # Create structured pseudo-sample
                sample = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output
                }
                # debugprint(f"文本 {idx+1} 解析成功: {sample}")

                # Add to result list
                parsed_samples.append(sample)

            except Exception as e:
                logger.debug(f"Error parsing pseudo-sample: {e}")
                debugprint(f"解析文本 {idx+1} 时发生异常: {e}，跳过")
                parse_errors += 1
                continue

        debugprint(f"原始解析得到的样本数: {len(parsed_samples)}, 解析错误数: {parse_errors}")

        # Ensure pseudo-sample diversity, remove exact duplicates
        unique_samples = []
        seen = set()
        duplicates_removed = 0

        debugprint("开始移除重复伪样本...")
        for sample in parsed_samples:
            # Use string representation of sample as unique identifier
            try:
                sample_str = json.dumps(sample, sort_keys=True)
                if sample_str not in seen:
                    seen.add(sample_str)
                    unique_samples.append(sample)
                else:
                    duplicates_removed += 1
                    # debugprint(f"发现重复样本: {sample_str}") # Can be verbose
            except TypeError as e:
                 debugprint(f"序列化样本时出错 (可能包含无法序列化的类型): {sample}, 错误: {e}，跳过此样本")
                 duplicates_removed += 1 # Count as removed due to error

        debugprint(f"移除重复样本完成，移除了 {duplicates_removed} 个重复样本")
        debugprint(f"parse_generated_texts 完成，返回 {len(unique_samples)} 个唯一伪样本")
        return unique_samples

    def refine_pseudo_samples(self, pseudo_samples):
        """
        使用前一任务模型优化伪样本输出
        在分布式环境中，只在rank 0进程上执行优化
        """
        rank = get_rank()
        debugprint(f"进程 rank={rank} refine_pseudo_samples 开始，伪样本数量: {len(pseudo_samples)}")

        # 只在主进程或非分布式环境中优化伪样本
        if not is_distributed() or is_main_process():
            refined_samples = self._refine_pseudo_samples_impl(pseudo_samples)
            debugprint(f"主进程优化了 {len(refined_samples)} 个伪样本")
        else:
            # 非主进程初始化为输入样本的副本
            refined_samples = pseudo_samples
            debugprint(f"非主进程跳过伪样本优化")

        # 在分布式环境中，广播优化后的伪样本到所有进程
        if is_distributed():
            # 等待主进程完成优化
            torch.distributed.barrier()
            # 将优化后的伪样本从rank 0广播到其他进程
            refined_samples = broadcast_object(refined_samples, src=0)
            debugprint(f"进程 rank={rank} 接收到广播的优化后伪样本，数量: {len(refined_samples)}")

        return refined_samples

    def _refine_pseudo_samples_impl(self, pseudo_samples):
        """
        实际优化伪样本的内部实现，只在主进程调用
        """
        debugprint(f"_refine_pseudo_samples_impl 开始，伪样本数量: {len(pseudo_samples)}")

        # 如果未指定上一个任务模型，则跳过优化步骤
        if not self.cl_finetuning_args.previous_task_model:
            logger.warning_rank0("未指定previous_task_model，跳过伪样本优化步骤")
            return pseudo_samples

        # 加载上一个任务的模型
        prev_model, tokenizer = self.setup_previous_model()
        tokenizer.padding_side = "left"
        prev_model = prev_model.cuda()

        if prev_model is None or tokenizer is None:
            logger.warning_rank0("加载上一任务模型失败，跳过伪样本优化步骤")
            return pseudo_samples

        # 设置生成配置
        generation_config = GenerationConfig(
            temperature=0.7,  # 使用较低的温度提高输出质量
            top_p=0.95,
            top_k=40,
            num_beams=2,  # 使用beam search提高生成质量
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # 存储优化后的伪样本
        refined_samples = []

        # 分批次处理以避免GPU内存不足
        batch_size = 5
        num_batches = (len(pseudo_samples) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(pseudo_samples))

            batch_samples = pseudo_samples[start_idx:end_idx]
            batch_prompts = []

            # 为每个伪样本准备输入提示词
            for sample in batch_samples:
                prompt = f"Instruction: {sample['instruction']}\nInput: {sample['input']}\nOutput:"
                batch_prompts.append(prompt)

            # 编码输入提示词
            batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt")
            input_ids = batch_inputs["input_ids"].to(prev_model.device)
            attention_mask = batch_inputs["attention_mask"].to(prev_model.device)

            # 使用上一个任务的模型生成优化后的输出
            with torch.no_grad():
                outputs = prev_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    max_new_tokens=256,
                    return_dict_in_generate=True
                )

            # 解码生成的文本并更新伪样本
            for j, sample in enumerate(batch_samples):
                if isinstance(outputs, dict) and "sequences" in outputs:
                    # 只取新生成的token
                    new_tokens = outputs.sequences[j][len(input_ids[j]):]
                    refined_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

                    # 创建优化后的伪样本
                    refined_sample = copy.deepcopy(sample)
                    refined_sample["output"] = refined_output

                    # 添加到结果列表
                    refined_samples.append(refined_sample)

        # 释放GPU内存
        del prev_model
        torch.cuda.empty_cache()

        logger.info_rank0(f"已优化 {len(refined_samples)} 个伪样本")
        debugprint(f"_refine_pseudo_samples_impl 完成，返回 {len(refined_samples)} 个优化后的伪样本")

        return refined_samples

    def select_diverse_samples(self, pseudo_samples, max_samples=None):
        """
        使用聚类方法选择多样化的伪样本
        在分布式环境中，只在rank 0进程上执行聚类选择
        """
        rank = get_rank()
        debugprint(f"进程 rank={rank} select_diverse_samples 开始，待选样本数: {len(pseudo_samples)}, 最大样本数: {max_samples or self.pseudo_sample_memory}")

        # 只在主进程或非分布式环境中执行聚类选择
        if not is_distributed() or is_main_process():
            selected_samples = self._select_diverse_samples_impl(pseudo_samples, max_samples)
            debugprint(f"主进程选择了 {len(selected_samples)} 个多样化伪样本")
        else:
            # 非主进程初始化为输入样本的副本或截断版本
            if max_samples is not None and len(pseudo_samples) > max_samples:
                # 使用固定随机种子确保所有进程选择相同的样本
                old_state = random.getstate()
                random.seed(42)
                selected_samples = random.sample(pseudo_samples, max_samples)
                random.setstate(old_state)
                debugprint(f"非主进程使用固定随机种子选择了 {len(selected_samples)} 个样本")
            else:
                selected_samples = pseudo_samples
                debugprint(f"非主进程跳过样本选择 (样本数量不超过最大限制)")

        # 在分布式环境中，广播选择后的伪样本到所有进程
        if is_distributed():
            # 等待主进程完成选择
            torch.distributed.barrier()
            # 将选择后的伪样本从rank 0广播到其他进程
            selected_samples = broadcast_object(selected_samples, src=0)
            debugprint(f"进程 rank={rank} 接收到广播的选择后伪样本，数量: {len(selected_samples)}")

        return selected_samples

    def _select_diverse_samples_impl(self, pseudo_samples, max_samples=None):
        """
        实际执行多样化伪样本选择的内部实现，只在主进程调用
        """
        debugprint(f"_select_diverse_samples_impl 开始，待选样本数: {len(pseudo_samples)}, 最大样本数: {max_samples or self.pseudo_sample_memory}")
        if max_samples is None:
            max_samples = self.pseudo_sample_memory
            debugprint(f"使用默认最大样本数: {max_samples}")

        # 如果伪样本数量不足，直接返回所有样本
        if len(pseudo_samples) <= max_samples:
            logger.info_rank0(f"伪样本数量({len(pseudo_samples)})不超过请求的最大样本数({max_samples})，返回所有样本")
            debugprint("伪样本数量不足，无需选择，直接返回")
            return pseudo_samples

        # 如果未能加载句子编码器，则随机选择
        if self.sentence_encoder is None:
            logger.warning_rank0("未加载SimCSE编码器，将随机选择伪样本")
            debugprint("警告: 未加载 SimCSE 编码器，执行随机选择")
            # 使用固定随机种子确保所有进程选择相同的样本
            old_state = random.getstate()
            random.seed(42)
            selected_samples = random.sample(pseudo_samples, max_samples)
            random.setstate(old_state)
            debugprint(f"随机选择了 {len(selected_samples)} 个伪样本")
            debugprint("_select_diverse_samples_impl 完成 (随机选择)")
            return selected_samples

        # 提取伪样本的文本表示
        texts = []
        debugprint("提取用于编码的文本...")
        for i, sample in enumerate(pseudo_samples):
            # Combine instruction, input, and output for embedding
            text = f"Instruction: {sample['instruction']} Input: {sample['input']} Output: {sample['output']}"
            # text = f"{sample['instruction']} {sample['input']} {sample['output']}" # Original simpler version
            texts.append(text)
            # if i < 5: debugprint(f"样本 {i} 文本 (部分): {text[:100]}...") # Debug first few texts
        debugprint(f"共提取了 {len(texts)} 条文本")

        # 使用SimCSE编码器获取嵌入表示
        try:
            debugprint("使用 SimCSE 编码器获取嵌入...")
            # 使用SimCSE模型获取嵌入
            embeddings = self.sentence_encoder.encode(texts, convert_to_numpy=True)
            debugprint(f"获取嵌入完成，形状: {embeddings.shape}")
        except Exception as e:
            logger.warning_rank0(f"获取SimCSE嵌入失败: {e}，将随机选择伪样本")
            debugprint(f"获取 SimCSE 嵌入失败: {e}，执行随机选择")
            # 使用固定随机种子确保所有进程选择相同的样本
            old_state = random.getstate()
            random.seed(42)
            selected_samples = random.sample(pseudo_samples, max_samples)
            random.setstate(old_state)
            debugprint(f"随机选择了 {len(selected_samples)} 个伪样本")
            debugprint("_select_diverse_samples_impl 完成 (SimCSE 失败后随机选择)")
            return selected_samples

        # 设置聚类数量，不超过最大样本数
        n_clusters = min(self.n_clusters, max_samples, len(pseudo_samples))
        debugprint(f"设置聚类数量: {n_clusters}")

        # 使用K-means聚类
        debugprint("开始 K-means 聚类...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') # Use n_init='auto' for newer sklearn
        # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Original
        cluster_labels = kmeans.fit_predict(embeddings)
        debugprint("K-means 聚类完成")

        # 为每个聚类选择最接近中心的样本
        selected_indices = []
        debugprint("开始从聚类中选择样本...")

        for i in range(n_clusters):
            # 获取当前聚类的所有样本索引
            cluster_indices = np.where(cluster_labels == i)[0]
            # cluster_indices = [j for j, label in enumerate(cluster_labels) if label == i] # Original
            debugprint(f"处理聚类 {i}，包含 {len(cluster_indices)} 个样本")

            if len(cluster_indices) == 0:
                 debugprint(f"警告: 聚类 {i} 为空，跳过")
                 continue

            # 如果聚类中只有一个样本，直接选择
            # if len(cluster_indices) == 1:
            #     selected_indices.append(cluster_indices[0])
            #     debugprint(f"聚类 {i} 只有一个样本，选择索引 {cluster_indices[0]}")
            #     continue

            # 获取当前聚类的中心
            cluster_center = kmeans.cluster_centers_[i]

            # 计算每个样本到中心的距离 (using numpy for efficiency)
            cluster_embeddings = embeddings[cluster_indices]
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)

            # # Original distance calculation:
            # distances = []
            # for idx in cluster_indices:
            #     dist = np.linalg.norm(embeddings[idx] - cluster_center)
            #     distances.append((idx, dist))

            # 按距离排序并选择最近的样本
            # distances.sort(key=lambda x: x[1]) # Original sort
            sorted_indices_in_cluster = np.argsort(distances) # Indices relative to cluster_indices

            # Map back to original indices
            sorted_original_indices = cluster_indices[sorted_indices_in_cluster]


            # 从每个聚类中选择多个样本，总数不超过max_samples
            # samples_per_cluster = max(1, max_samples // n_clusters) # Original: might lead to exceeding max_samples if max_samples is not divisible
            # Calculate samples per cluster more carefully to respect max_samples overall
            # We will select the closest one from each cluster first, then fill up if needed.
            # For now, just select the single closest one per cluster
            closest_original_index = sorted_original_indices[0]
            if closest_original_index not in selected_indices: # Avoid adding duplicates if K-means assigns poorly
                 selected_indices.append(closest_original_index)
                 debugprint(f"聚类 {i}: 选择最接近中心的样本，索引 {closest_original_index}")
            else:
                 debugprint(f"聚类 {i}: 最接近中心的样本索引 {closest_original_index} 已被选择，尝试选择下一个最近的")
                 if len(sorted_original_indices) > 1:
                     second_closest_original_index = sorted_original_indices[1]
                     if second_closest_original_index not in selected_indices:
                         selected_indices.append(second_closest_original_index)
                         debugprint(f"聚类 {i}: 选择第二接近中心的样本，索引 {second_closest_original_index}")
                     else:
                         debugprint(f"聚类 {i}: 第二接近中心的样本索引 {second_closest_original_index} 也已被选择，跳过此聚类的额外选择")


            # Original logic for selecting multiple per cluster:
            # samples_this_cluster = 0
            # for k in range(len(distances)):
            #     # idx_to_add = distances[k][0] # Original index
            #     idx_to_add = sorted_original_indices[k]
            #     if idx_to_add not in selected_indices: # Ensure we don't add duplicates
            #          selected_indices.append(idx_to_add)
            #          samples_this_cluster += 1
            #          debugprint(f"聚类 {i}: 选择第 {k+1} 近的样本，索引 {idx_to_add}")
            #          if samples_this_cluster >= samples_per_cluster:
            #              break
            #     if len(selected_indices) >= max_samples: # Stop if we reach max samples overall
            #         break
            # if len(selected_indices) >= max_samples:
            #     break

        debugprint(f"通过聚类中心选择后，共有 {len(selected_indices)} 个选定索引")

        # 如果选择的样本数量不足，补充距离聚类中心最远的样本 (more diverse than random)
        if len(selected_indices) < max_samples:
            debugprint(f"选定样本数 ({len(selected_indices)}) 少于目标数 ({max_samples})，补充样本...")
            # Collect all indices and their distances
            all_distances = []
            for i in range(n_clusters):
                 cluster_indices = np.where(cluster_labels == i)[0]
                 if len(cluster_indices) > 0:
                     cluster_center = kmeans.cluster_centers_[i]
                     cluster_embeddings = embeddings[cluster_indices]
                     distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
                     for j, idx in enumerate(cluster_indices):
                         if idx not in selected_indices:
                             all_distances.append((idx, distances[j]))

            # Sort by distance descending (furthest first)
            all_distances.sort(key=lambda x: x[1], reverse=True)

            needed = max_samples - len(selected_indices)
            debugprint(f"需要补充 {needed} 个样本，从距离中心最远的样本中选择")

            added_count = 0
            for idx, dist in all_distances:
                if added_count < needed:
                     if idx not in selected_indices: # Ensure not already selected
                         selected_indices.append(idx)
                         added_count += 1
                         debugprint(f"补充样本: 索引 {idx} (距离 {dist:.4f})")
                else:
                    break
            debugprint(f"补充后总选定索引数: {len(selected_indices)}")

        # # Original random fill:
        # if len(selected_indices) < max_samples:
        #     debugprint(f"选定样本数 ({len(selected_indices)}) 少于目标数 ({max_samples})，随机补充剩余样本...")
        #     remaining_indices = [i for i in range(len(pseudo_samples)) if i not in selected_indices]
        #     needed = max_samples - len(selected_indices)
        #     debugprint(f"需要补充 {needed} 个样本，从 {len(remaining_indices)} 个剩余样本中随机选择")
        #     additional_indices = random.sample(
        #         remaining_indices,
        #         min(needed, len(remaining_indices))
        #     )
        #     selected_indices.extend(additional_indices)
        #     debugprint(f"补充了 {len(additional_indices)} 个随机样本，补充后总选定索引数: {len(selected_indices)}")

        # 确保不超过max_samples (should already be handled, but as a safeguard)
        if len(selected_indices) > max_samples:
            debugprint(f"警告: 选定索引数 ({len(selected_indices)}) 超过目标数 ({max_samples})，截断至目标数")
            selected_indices = selected_indices[:max_samples]

        # 选择对应的伪样本
        debugprint(f"根据最终选定的 {len(selected_indices)} 个索引提取伪样本...")
        selected_samples = [pseudo_samples[i] for i in selected_indices]

        logger.info_rank0(f"通过聚类选择了 {len(selected_samples)} 个多样化伪样本")
        debugprint(f"_select_diverse_samples_impl 完成 (聚类选择)，返回 {len(selected_samples)} 个样本")
        return selected_samples

    def save_pseudo_samples(self, samples, task_id, prev_task_id=None):
        """
        保存伪样本到指定目录
        在分布式环境中，只在rank 0进程上执行保存
        """
        rank = get_rank()
        debugprint(f"进程 rank={rank} save_pseudo_samples 开始，当前任务ID: {task_id}, 上一任务ID: {prev_task_id}, 待保存样本数: {len(samples)}")

        # 只在主进程或非分布式环境中执行文件操作
        if not is_distributed() or is_main_process():
            task_dir = self._save_pseudo_samples_impl(samples, task_id, prev_task_id)
            debugprint(f"主进程保存了伪样本，目录: {task_dir}")
        else:
            # 非主进程初始化为None
            task_dir = None
            debugprint(f"非主进程跳过伪样本保存")

        # 在分布式环境中，广播保存路径到所有进程
        if is_distributed():
            # 等待主进程完成保存
            torch.distributed.barrier()
            # 将保存路径从rank 0广播到其他进程
            task_dir = broadcast_object(task_dir, src=0)
            debugprint(f"进程 rank={rank} 接收到广播的伪样本保存路径: {task_dir}")

        return task_dir

    def _save_pseudo_samples_impl(self, samples, task_id, prev_task_id=None):
        """
        实际保存伪样本的内部实现，只在主进程调用
        """
        debugprint(f"_save_pseudo_samples_impl 开始，当前任务ID: {task_id}, 上一任务ID: {prev_task_id}, 待保存样本数: {len(samples)}")
        # 构建保存路径
        output_dir = self.pseudo_samples_dir
        debugprint(f"配置的伪样本目录 (相对/绝对): {output_dir}")
        if not os.path.isabs(output_dir):
            # 如果是相对路径，则相对于当前工作目录
            output_dir = os.path.join(os.getcwd(), output_dir)
            debugprint(f"转换为绝对路径: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        debugprint(f"确保根目录存在: {output_dir}")

        task_dir = os.path.join(output_dir, str(task_id)) # Ensure task_id is string for path join
        os.makedirs(task_dir, exist_ok=True)
        debugprint(f"确保当前任务目录存在: {task_dir}")

        # 如果存在上一任务的伪样本，先加载它们
        prev_samples = []
        if prev_task_id and str(prev_task_id) != str(task_id): # Ensure comparison is robust
            debugprint(f"存在上一任务ID ({prev_task_id})，尝试加载历史伪样本...")
            prev_task_dir = os.path.join(output_dir, str(prev_task_id))
            debugprint(f"检查上一任务目录: {prev_task_dir}")
            if os.path.exists(prev_task_dir):
                # 检查前序任务的伪样本文件
                # Look for the specific file pseudo_{prev_task_id}.json first
                prev_file_name = f"pseudo_{prev_task_id}.json"
                prev_file_path = os.path.join(prev_task_dir, prev_file_name)
                debugprint(f"检查特定历史文件: {prev_file_path}")
                if os.path.exists(prev_file_path):
                    try:
                        with open(prev_file_path, 'r', encoding='utf-8') as f:
                            prev_data = json.load(f)
                            prev_samples.extend(prev_data)
                            debugprint(f"成功加载 {len(prev_data)} 条历史伪样本自 {prev_file_path}")
                    except Exception as e:
                        logger.warning_rank0(f"加载前序任务伪样本文件 {prev_file_path} 出错: {e}")
                        debugprint(f"加载历史伪样本文件 {prev_file_path} 时出错: {e}")
                else:
                     debugprint(f"未找到特定的历史伪样本文件 {prev_file_path}，将检查目录中其他可能的 .json 文件 (旧逻辑回退，不推荐)")
                     # Fallback to checking all json files (less precise, original behavior)
                     for prev_file in os.listdir(prev_task_dir):
                         if prev_file.endswith(".json") and prev_file != "dataset_info.json":
                             prev_file_path_fallback = os.path.join(prev_task_dir, prev_file)
                             debugprint(f"回退检查文件: {prev_file_path_fallback}")
                             try:
                                 with open(prev_file_path_fallback, 'r', encoding='utf-8') as f:
                                     prev_data = json.load(f)
                                     # Avoid adding duplicates if multiple files somehow exist
                                     current_prev_ids = {json.dumps(s, sort_keys=True) for s in prev_samples}
                                     added_count = 0
                                     for sample in prev_data:
                                         sample_str = json.dumps(sample, sort_keys=True)
                                         if sample_str not in current_prev_ids:
                                             prev_samples.append(sample)
                                             current_prev_ids.add(sample_str)
                                             added_count +=1
                                     if added_count > 0:
                                          debugprint(f"成功加载 {added_count} 条 (唯一) 历史伪样本自 {prev_file_path_fallback}")
                             except Exception as e:
                                 logger.warning_rank0(f"加载前序任务伪样本出错 (回退): {e}")
                                 debugprint(f"加载历史伪样本文件 {prev_file_path_fallback} 时出错 (回退): {e}")
            else:
                 debugprint(f"上一任务目录 {prev_task_dir} 不存在")
        else:
             debugprint("没有上一任务ID或与当前任务ID相同，不加载历史伪样本")

        # 合并当前任务生成的伪样本和历史伪样本
        debugprint(f"合并当前任务样本 ({len(samples)}) 和历史样本 ({len(prev_samples)})...")
        # Ensure no duplicates between current and previous samples
        prev_samples_set = {json.dumps(s, sort_keys=True) for s in prev_samples}
        unique_new_samples = []
        duplicates_with_prev = 0
        for sample in samples:
            sample_str = json.dumps(sample, sort_keys=True)
            if sample_str not in prev_samples_set:
                unique_new_samples.append(sample)
            else:
                duplicates_with_prev += 1

        all_samples = unique_new_samples + prev_samples
        debugprint(f"当前任务与历史样本重复 {duplicates_with_prev} 个。合并后总样本数: {len(all_samples)}")

        # 保存合并后的伪样本
        pseudo_file_name = f"pseudo_{task_id}.json"
        pseudo_file_path = os.path.join(task_dir, pseudo_file_name)
        debugprint(f"准备保存合并后的伪样本到: {pseudo_file_path}")

        with open(pseudo_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        debugprint("合并后的伪样本保存完成")

        # 从原始数据集获取格式信息
        debugprint("获取原始数据集格式信息...")
        dataset_format = self.get_dataset_format()
        debugprint(f"获取到的数据集格式信息: {dataset_format}")

        # 构建数据集注册信息
        dataset_info_key = f"pseudo_{task_id}"
        dataset_info = {
            dataset_info_key: {
                "file_name": pseudo_file_name,
                "formatting": dataset_format.get("formatting", "alpaca"), # Use .get with default
                "split": "train" # Always train split for pseudo samples
            }
        }
        debugprint(f"构建 dataset_info.json 条目，键: {dataset_info_key}")

        # 如果有其他格式特定配置，添加到注册信息中
        if "columns" in dataset_format:
            dataset_info[dataset_info_key]["columns"] = dataset_format["columns"]
            debugprint(f"添加 columns 信息: {dataset_format['columns']}")
        if "tags" in dataset_format:
            dataset_info[dataset_info_key]["tags"] = dataset_format["tags"]
            debugprint(f"添加 tags 信息: {dataset_format['tags']}")

        # 保存数据集注册信息
        dataset_info_path = os.path.join(task_dir, "dataset_info.json")
        debugprint(f"准备保存 dataset_info.json 到: {dataset_info_path}")
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        debugprint("dataset_info.json 保存完成")

        logger.info_rank0(f"已保存当前任务和历史任务的合并伪样本到 {pseudo_file_path}")
        logger.info_rank0(f"包含当前任务伪样本: {len(unique_new_samples)}个 (去除与历史重复后)")
        logger.info_rank0(f"包含历史任务伪样本: {len(prev_samples)}个")
        logger.info_rank0(f"总伪样本数量: {len(all_samples)}个")
        debugprint(f"_save_pseudo_samples_impl 完成，返回任务目录: {task_dir}")
        return task_dir

    def get_dataset_format(self):
        """
        从原始数据集获取格式信息，用于创建与之兼容的dataset_info.json
        """
        debugprint("get_dataset_format 开始")
        try:
            # 获取当前任务的数据集信息
            debugprint(f"尝试使用 data_args 获取数据集列表: dataset={self.data_args.dataset}, dataset_dir={self.data_args.dataset_dir}")
            # dataset_list = get_dataset_list(self.data_args.dataset, self.data_args.dataset_dir) # get_dataset_list might not exist or work as expected here
            # Let's try reading dataset_info.json directly as the primary source

            dataset_info_path = None
            potential_paths = [self.data_args.dataset_dir, os.path.join(os.getcwd(), "data")]
            for data_dir in potential_paths:
                 path = os.path.join(data_dir, "dataset_info.json")
                 if os.path.exists(path):
                     dataset_info_path = path
                     debugprint(f"找到 dataset_info.json: {dataset_info_path}")
                     break

            if dataset_info_path:
                try:
                    with open(dataset_info_path, 'r', encoding='utf-8') as f:
                        dataset_info = json.load(f)
                        debugprint(f"成功加载 dataset_info.json: {dataset_info_path}")
                        # 获取当前任务的数据集名称 (use the first one specified)
                        dataset_name = self.data_args.dataset[0] if isinstance(self.data_args.dataset, list) else self.data_args.dataset
                        debugprint(f"参考的数据集名称: {dataset_name}")

                        # Find the entry matching the dataset name (handle potential split suffix)
                        target_entry = None
                        if dataset_name in dataset_info:
                             target_entry = dataset_info[dataset_name]
                             debugprint(f"在 dataset_info 中找到键 '{dataset_name}'")
                        else:
                            # Check if the name includes a split like 'dataset_name_train'
                            base_name = dataset_name.rsplit('_', 1)[0]
                            if base_name in dataset_info:
                                target_entry = dataset_info[base_name]
                                debugprint(f"在 dataset_info 中找到基础键 '{base_name}'")
                            else:
                                 # Also check for keys like 'dataset_name_split' if the user provided that
                                 if f"{dataset_name}_train" in dataset_info:
                                      target_entry = dataset_info[f"{dataset_name}_train"]
                                      debugprint(f"在 dataset_info 中找到键 '{dataset_name}_train'")

                        if target_entry:
                            # 提取该数据集的格式信息
                            format_info = {}
                            src_info = target_entry

                            # 复制基本格式信息
                            format_info["formatting"] = src_info.get("formatting", "alpaca")
                            # format_info["split"] = src_info.get("split", "train") # Split is always train for pseudo

                            # 复制额外的格式信息
                            if "columns" in src_info:
                                format_info["columns"] = src_info["columns"]
                            if "tags" in src_info:
                                format_info["tags"] = src_info["tags"]

                            debugprint(f"提取到格式信息: {format_info}")
                            debugprint("get_dataset_format 完成 (从 dataset_info.json 获取)")
                            return format_info
                        else:
                            debugprint(f"警告: 在 dataset_info.json 中未找到数据集 '{dataset_name}' 或其变体的条目")
                except Exception as e:
                    logger.warning_rank0(f"读取dataset_info.json出错: {e}")
                    debugprint(f"读取 {dataset_info_path} 出错: {e}")
            else:
                debugprint("警告: 未找到 dataset_info.json")
        except Exception as e:
            logger.warning_rank0(f"获取数据集格式信息出错: {e}")
            debugprint(f"获取数据集格式信息时发生异常: {e}")

        # 如果无法获取格式信息，返回默认格式
        default_format = {"formatting": "alpaca"} # Removed split here, as it's fixed to train later
        debugprint("无法获取特定格式信息，返回默认 alpaca 格式")
        debugprint("get_dataset_format 完成 (返回默认格式)")
        return default_format
