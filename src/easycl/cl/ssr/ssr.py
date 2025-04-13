import os
import json
import copy
import random
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
# Replace original sentence-transformers import
from transformers import AutoModel, AutoTokenizer as HFAutoTokenizer

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

logger = get_logger(__name__)

class SimCSEModel:
    """SimCSE model wrapper for getting text semantic representations"""
    
    def __init__(self, model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
        self.tokenizer = HFAutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Check CUDA availability
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"
    
    def encode(self, sentences, batch_size=32, convert_to_numpy=True):
        """Encode input sentences to get their feature representations"""
        embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            
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
            
            # No gradient computation
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get hidden states corresponding to [CLS]
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                
                # Normalize
                cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
                
                if convert_to_numpy:
                    cls_embeddings = cls_embeddings.cpu().numpy()
                
                embeddings.append(cls_embeddings)
        
        # Merge results from all batches
        if convert_to_numpy:
            return np.vstack(embeddings)
        else:
            return torch.cat(embeddings, dim=0)

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
            self.sentence_encoder = SimCSEModel("princeton-nlp/sup-simcse-bert-base-uncased")
            logger.info_rank0("Successfully loaded SimCSE encoder for SSR pseudo-sample diversity selection")
        except Exception as e:
            logger.warning_rank0(f"Failed to load SimCSE encoder: {e}, will use random selection for pseudo-samples")
    
    def setup_base_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load base model for ICL generation
        """
        # Create model args copy
        base_model_args = copy.deepcopy(self.model_args)
        
        # Use specified base model path
        if self.base_model_path:
            base_model_args.model_name_or_path = self.base_model_path
        
        # Remove all adapters, use pure base model for ICL generation
        base_model_args.adapter_name_or_path = None
        
        # Load tokenizer
        tokenizer_module = load_tokenizer(base_model_args)
        tokenizer = tokenizer_module["tokenizer"]
        
        # Load base model - no training, disable adapters
        temp_finetuning_args = copy.deepcopy(self.finetuning_args)
        temp_finetuning_args.finetuning_type = "lora"  # Use lora type but don't load adapters
        base_model = load_model(
            tokenizer, 
            base_model_args, 
            temp_finetuning_args,
            is_trainable=False
        )
        
        logger.info_rank0(f"Loaded SSR base model for ICL generation, path: {base_model_args.model_name_or_path}")
        
        return base_model, tokenizer
    
    def setup_previous_model(self) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """
        Load model from previous task for pseudo-sample refinement
        """
        if not self.finetuning_args.previous_task_model:
            logger.warning_rank0("previous_task_model not specified, will skip pseudo-sample refinement step")
            return None, None
        
        # Create model args copy
        prev_model_args = copy.deepcopy(self.model_args)
        
        # Load adapter or full model from previous task
        if self.finetuning_args.finetuning_type == "lora":
            # If using LoRA, load adapter from previous task
            prev_model_args.adapter_name_or_path = [self.finetuning_args.previous_task_model]
        else:
            # If using full-parameter finetuning, load complete model from previous task
            prev_model_args.model_name_or_path = self.finetuning_args.previous_task_model
            prev_model_args.adapter_name_or_path = None
        
        # Load tokenizer
        tokenizer_module = load_tokenizer(prev_model_args)
        tokenizer = tokenizer_module["tokenizer"]
        
        # Load model from previous task - no training
        prev_model = load_model(
            tokenizer, 
            prev_model_args, 
            self.finetuning_args,
            is_trainable=False
        )
        
        logger.info_rank0(f"Loaded previous task model for pseudo-sample refinement, path: {self.finetuning_args.previous_task_model}")
        
        return prev_model, tokenizer
    
    def get_few_shot_examples(self, dataset, num_shots=None):
        """
        Randomly select few-shot examples from dataset
        """
        if num_shots is None:
            num_shots = self.num_shots
        
        # Ensure dataset has enough samples
        if len(dataset) < num_shots:
            logger.warning_rank0(f"Dataset sample count ({len(dataset)}) is less than requested examples ({num_shots}), will use all samples")
            num_shots = len(dataset)
        
        # Randomly select samples
        indices = random.sample(range(len(dataset)), num_shots)
        examples = [dataset[i] for i in indices]
        
        return examples
    
    def construct_icl_prompt(self, examples, template_type="alpaca"):
        """
        Construct ICL prompt
        """
        instruction = 'Create task samples following examples below.\n\n'
        
        for example in examples:
            # Ensure compatibility with different dataset formats
            if "instruction" in example and "input" in example and "output" in example:
                # Alpaca format
                input_text = example["input"]
                input_text = "<noinput>" if input_text.lower() == "" else input_text
                
                instruction += (
                    f"Instruction: {example['instruction']}\n" +
                    f"Input: {input_text}\n" +
                    f"Output: {example['output']}\n\n"
                )
            elif "messages" in example:
                # ShareGPT format
                user_message = ""
                assistant_message = ""
                
                for msg in example["messages"]:
                    if msg["role"] == "user":
                        user_message = msg["content"]
                    elif msg["role"] == "assistant":
                        assistant_message = msg["content"]
                
                instruction += (
                    f"Instruction: Answer the following query\n" +
                    f"Input: {user_message}\n" +
                    f"Output: {assistant_message}\n\n"
                )
            else:
                # Generic format - try best guess
                for key, value in example.items():
                    if isinstance(value, str):
                        # Assume first string field is instruction, second is input, third is output
                        if "instruction" not in locals():
                            instruction += f"Instruction: {value}\n"
                        elif "input_text" not in locals():
                            input_text = "<noinput>" if value == "" else value
                            instruction += f"Input: {input_text}\n"
                        elif "output" not in locals():
                            instruction += f"Output: {value}\n\n"
                            break
        
        instruction += 'Instruction:'
        
        return instruction
    
    def generate_pseudo_samples(self, dataset, num_samples=None):
        """
        Generate pseudo-samples using base model through ICL
        """
        if num_samples is None:
            num_samples = self.pseudo_sample_memory
            
        # Load base model
        base_model, tokenizer = self.setup_base_model()
        
        # Select few-shot examples
        few_shot_examples = self.get_few_shot_examples(dataset)
        
        # Construct ICL prompt
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
        
        # Encode prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(base_model.device)
        
        # Generate pseudo-samples
        generated_texts = []
        
        # Generate in batches to avoid GPU OOM
        batch_size = min(5, num_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i*batch_size)
            if current_batch_size <= 0:
                break
                
            logger.info_rank0(f"Generating pseudo-sample batch {i+1}/{num_batches}")
            
            # Generate multiple samples
            with torch.no_grad():
                outputs = base_model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    max_new_tokens=512,
                    num_return_sequences=current_batch_size,
                    return_dict_in_generate=True
                )
            
            # Decode generated text
            for j in range(current_batch_size):
                if isinstance(outputs, dict) and "sequences" in outputs:
                    # Only take newly generated tokens
                    new_tokens = outputs.sequences[j][input_ids.shape[1]:]
                    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    # Process and add generated text to result list
                    if generated_text:
                        generated_texts.append(prompt + " " + generated_text)
        
        # Free GPU memory
        del base_model
        torch.cuda.empty_cache()
        
        # Parse generated text into structured pseudo-samples
        pseudo_samples = self.parse_generated_texts(generated_texts)
        
        logger.info_rank0(f"Successfully generated {len(pseudo_samples)} valid pseudo-samples")
        
        return pseudo_samples
    
    def parse_generated_texts(self, generated_texts):
        """
        Parse generated text into structured pseudo-samples
        """
        parsed_samples = []
        
        for text in generated_texts:
            try:
                # Split generated text to extract instruction, input and output
                parts = text.split("Output:", 1)
                if len(parts) != 2:
                    continue
                    
                output = parts[1].strip()
                remaining = parts[0].strip()
                
                # Split remaining part to get input
                parts = remaining.split("Input:", 1)
                if len(parts) != 2:
                    continue
                    
                input_text = parts[1].strip()
                instruction_parts = parts[0].strip().split("Instruction:", 1)
                
                if len(instruction_parts) != 2:
                    continue
                    
                instruction = instruction_parts[1].strip()
                
                # Handle special tokens
                if input_text == "<noinput>":
                    input_text = ""
                
                # Validate parsed fields
                if not instruction:
                    continue
                
                # Create structured pseudo-sample
                sample = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output
                }
                
                # Add to result list
                parsed_samples.append(sample)
                
            except Exception as e:
                logger.debug(f"Error parsing pseudo-sample: {e}")
                continue
        
        # Ensure pseudo-sample diversity, remove exact duplicates
        unique_samples = []
        seen = set()
        
        for sample in parsed_samples:
            # Use string representation of sample as unique identifier
            sample_str = json.dumps(sample, sort_keys=True)
            if sample_str not in seen:
                seen.add(sample_str)
                unique_samples.append(sample)
        
        return unique_samples
    
    def refine_pseudo_samples(self, pseudo_samples):
        """
        使用前一任务模型优化伪样本输出
        """
        # 如果未指定上一个任务模型，则跳过优化步骤
        if not self.finetuning_args.previous_task_model:
            logger.warning_rank0("未指定previous_task_model，跳过伪样本优化步骤")
            return pseudo_samples
        
        # 加载上一个任务的模型
        prev_model, tokenizer = self.setup_previous_model()
        
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
        
        return refined_samples
    
    def select_diverse_samples(self, pseudo_samples, max_samples=None):
        """
        使用聚类方法选择多样化的伪样本
        """
        if max_samples is None:
            max_samples = self.pseudo_sample_memory
        
        # 如果伪样本数量不足，直接返回所有样本
        if len(pseudo_samples) <= max_samples:
            logger.info_rank0(f"伪样本数量({len(pseudo_samples)})不超过请求的最大样本数({max_samples})，返回所有样本")
            return pseudo_samples
        
        # 如果未能加载句子编码器，则随机选择
        if self.sentence_encoder is None:
            logger.warning_rank0("未加载SimCSE编码器，将随机选择伪样本")
            return random.sample(pseudo_samples, max_samples)
        
        # 提取伪样本的文本表示
        texts = []
        for sample in pseudo_samples:
            text = f"{sample['instruction']} {sample['input']} {sample['output']}"
            texts.append(text)
        
        # 使用SimCSE编码器获取嵌入表示
        try:
            # 使用SimCSE模型获取嵌入
            embeddings = self.sentence_encoder.encode(texts, convert_to_numpy=True)
        except Exception as e:
            logger.warning_rank0(f"获取SimCSE嵌入失败: {e}，将随机选择伪样本")
            return random.sample(pseudo_samples, max_samples)
        
        # 设置聚类数量，不超过最大样本数
        n_clusters = min(self.n_clusters, max_samples, len(pseudo_samples))
        
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 为每个聚类选择最接近中心的样本
        selected_indices = []
        
        for i in range(n_clusters):
            # 获取当前聚类的所有样本索引
            cluster_indices = [j for j, label in enumerate(cluster_labels) if label == i]
            
            if not cluster_indices:
                continue
                
            # 如果聚类中只有一个样本，直接选择
            if len(cluster_indices) == 1:
                selected_indices.append(cluster_indices[0])
                continue
                
            # 获取当前聚类的中心
            cluster_center = kmeans.cluster_centers_[i]
            
            # 计算每个样本到中心的距离
            distances = []
            for idx in cluster_indices:
                dist = np.linalg.norm(embeddings[idx] - cluster_center)
                distances.append((idx, dist))
            
            # 按距离排序并选择最近的样本
            distances.sort(key=lambda x: x[1])
            
            # 从每个聚类中选择多个样本，总数不超过max_samples
            samples_per_cluster = max(1, max_samples // n_clusters)
            for k in range(min(samples_per_cluster, len(distances))):
                selected_indices.append(distances[k][0])
        
        # 如果选择的样本数量不足，补充随机样本
        if len(selected_indices) < max_samples:
            remaining_indices = [i for i in range(len(pseudo_samples)) if i not in selected_indices]
            additional_indices = random.sample(
                remaining_indices, 
                min(max_samples - len(selected_indices), len(remaining_indices))
            )
            selected_indices.extend(additional_indices)
        
        # 确保不超过max_samples
        selected_indices = selected_indices[:max_samples]
        
        # 选择对应的伪样本
        selected_samples = [pseudo_samples[i] for i in selected_indices]
        
        logger.info_rank0(f"通过聚类选择了 {len(selected_samples)} 个多样化伪样本")
        
        return selected_samples
    
    def save_pseudo_samples(self, samples, task_id, prev_task_id=None):
        """
        保存伪样本到指定目录
        """
        # 构建保存路径
        output_dir = self.pseudo_samples_dir
        if not os.path.isabs(output_dir):
            # 如果是相对路径，则相对于当前工作目录
            output_dir = os.path.join(os.getcwd(), output_dir)
            
        os.makedirs(output_dir, exist_ok=True)
        
        task_dir = os.path.join(output_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        # 如果存在上一任务的伪样本，先加载它们
        prev_samples = []
        if prev_task_id and prev_task_id != task_id:
            prev_task_dir = os.path.join(output_dir, prev_task_id)
            if os.path.exists(prev_task_dir):
                # 检查前序任务的伪样本文件
                for prev_file in os.listdir(prev_task_dir):
                    if prev_file.startswith("pseudo_") and prev_file.endswith(".json"):
                        prev_file_path = os.path.join(prev_task_dir, prev_file)
                        try:
                            with open(prev_file_path, 'r', encoding='utf-8') as f:
                                prev_data = json.load(f)
                                prev_samples.extend(prev_data)
                        except Exception as e:
                            logger.warning_rank0(f"加载前序任务伪样本出错: {e}")
        
        # 合并当前任务生成的伪样本和历史伪样本
        all_samples = samples + prev_samples
        
        # 保存合并后的伪样本
        pseudo_file_name = f"pseudo_{task_id}.json"
        pseudo_file_path = os.path.join(task_dir, pseudo_file_name)
        
        with open(pseudo_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        
        # 从原始数据集获取格式信息
        dataset_format = self.get_dataset_format()
        
        # 构建数据集注册信息
        dataset_info = {
            f"pseudo_{task_id}": {
                "file_name": pseudo_file_name,
                "formatting": dataset_format["formatting"],
                "split": "train"
            }
        }
        
        # 如果有其他格式特定配置，添加到注册信息中
        if "columns" in dataset_format:
            dataset_info[f"pseudo_{task_id}"]["columns"] = dataset_format["columns"]
        if "tags" in dataset_format:
            dataset_info[f"pseudo_{task_id}"]["tags"] = dataset_format["tags"]
        
        # 保存数据集注册信息
        dataset_info_path = os.path.join(task_dir, "dataset_info.json")
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
            
        logger.info_rank0(f"已保存当前任务和历史任务的合并伪样本到 {pseudo_file_path}")
        logger.info_rank0(f"包含当前任务伪样本: {len(samples)}个")
        logger.info_rank0(f"包含历史任务伪样本: {len(prev_samples)}个")
        logger.info_rank0(f"总伪样本数量: {len(all_samples)}个")
        
        return task_dir
    
    def get_dataset_format(self):
        """
        从原始数据集获取格式信息，用于创建与之兼容的dataset_info.json
        """
        try:
            from llamafactory.data.parser import get_dataset_list
            
            # 获取当前任务的数据集信息
            dataset_list = get_dataset_list(self.data_args.dataset, self.data_args.dataset_dir)
            if not dataset_list:
                # 使用默认的alpaca格式
                return {"formatting": "alpaca", "split": "train"}
            
            # 假设我们使用第一个数据集作为参考
            dataset_info_path = os.path.join(self.data_args.dataset_dir, "dataset_info.json")
            if os.path.exists(dataset_info_path):
                try:
                    with open(dataset_info_path, 'r', encoding='utf-8') as f:
                        dataset_info = json.load(f)
                        # 获取当前任务的数据集名称
                        dataset_name = self.data_args.dataset[0] if isinstance(self.data_args.dataset, list) else self.data_args.dataset
                        if dataset_name in dataset_info:
                            # 提取该数据集的格式信息
                            format_info = {}
                            src_info = dataset_info[dataset_name]
                            
                            # 复制基本格式信息
                            format_info["formatting"] = src_info.get("formatting", "alpaca")
                            format_info["split"] = src_info.get("split", "train")
                            
                            # 复制额外的格式信息
                            if "columns" in src_info:
                                format_info["columns"] = src_info["columns"]
                            if "tags" in src_info:
                                format_info["tags"] = src_info["tags"]
                            
                            return format_info
                except Exception as e:
                    logger.warning_rank0(f"读取dataset_info.json出错: {e}")
        except Exception as e:
            logger.warning_rank0(f"获取数据集格式信息出错: {e}")
        
        # 如果无法获取格式信息，返回默认格式
        return {"formatting": "alpaca", "split": "train"}
