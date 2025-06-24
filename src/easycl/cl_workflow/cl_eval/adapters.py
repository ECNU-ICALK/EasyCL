"""持续学习评估适配器模块，用于适配不同格式的数据集"""

import re
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class CustomDatasetAdapter:
    """自定义数据集的评估适配器，直接比较答案文本"""
    
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """规范化答案文本"""
        # 1. 基础清理
        answer = answer.strip()
        answer = re.sub(r'\s+', ' ', answer)  # 统一空白字符
        answer = ''.join(char for char in answer if char.isprintable())  # 移除不可见字符
        
        # 2. 标点处理
        answer = re.sub(r'[.,!?;:\'"`\(\)\[\]\{\}]', '', answer)  # 移除标点
        answer = re.sub(r'[-_]', ' ', answer)  # 处理连字符
        
        # 3. 格式规范
        answer = answer.lower()  # 统一小写
        answer = re.sub(r'<[^>]+>', '', answer)  # 移除HTML标签
        
        # 4. 特殊情况处理
        answer = re.sub(r'^[A-Da-d]\.\s*', '', answer)  # 移除选项标签
        answer = re.sub(r'^\s*["\']|["\']\s*$', '', answer)  # 移除引号
        answer = re.sub(r'^\s*[\(\[\{]|[\)\]\}]\s*$', '', answer)  # 移除括号
        
        return answer.strip()

    @staticmethod
    def validate_answer(
        predicted: str,
        expected: str
    ) -> Tuple[bool, str, str]:
        """验证答案是否正确"""
        # 规范化答案
        norm_predicted = CustomDatasetAdapter.normalize_answer(predicted)
        norm_expected = CustomDatasetAdapter.normalize_answer(expected)
        
        # 1. 精确匹配
        if norm_predicted == norm_expected:
            return True, "exact", "Exact match"
        else:
            return False, "wrong", "Wrong answer"

    @staticmethod
    def format_example(
        example: Dict,
        template: str,
        subject_name: str,
        dataset_name: str,
        support_set: Optional[List[Dict]] = None,
        dataset_specific_info: Optional[Dict] = None
    ) -> List[Dict[str, str]]:
        """格式化样本数据为消息列表格式"""
        messages_to_log_and_return = []

        # 获取数据集配置信息
        if dataset_specific_info:
            formatting = dataset_specific_info.get("formatting", "alpaca")
            columns = dataset_specific_info.get("columns", {})
            tags = dataset_specific_info.get("tags", {})
        else:
            formatting = "alpaca"
            columns = {}
            tags = {}

        # 获取标签映射
        role_tag = tags.get("role_tag", "from")
        content_tag = tags.get("content_tag", "value")
        user_tag = tags.get("user_tag", "human")
        assistant_tag = tags.get("assistant_tag", "gpt")
        system_tag = tags.get("system_tag", "system")

        # 获取列映射
        messages_key = columns.get("messages", "conversations")
        images_key = columns.get("images", "image")

        # 1. 处理Few-Shot示例
        if support_set:
            for s_ex in support_set:
                if formatting == "sharegpt" and messages_key in s_ex:
                    # ShareGPT格式
                    for turn in s_ex[messages_key]:
                        if not isinstance(turn, dict):
                            continue
                        
                        # 直接复制turn内容，保持原始结构
                        message = dict(turn)  # 创建副本
                        messages_to_log_and_return.append(message)
                else:
                    # Alpaca格式
                    prompt = s_ex.get("instruction", "")
                    if s_ex.get("input"):
                        prompt = f"{prompt}\n{s_ex['input']}"
                    response = CustomDatasetAdapter.get_response_from_item(s_ex, dataset_specific_info, "output")
                    
                    messages_to_log_and_return.append({
                        role_tag: user_tag,
                        content_tag: f"{prompt}\nAnswer:"
                    })
                    messages_to_log_and_return.append({
                        role_tag: assistant_tag,
                        content_tag: response
                    })

        # 2. 处理主要示例
        if formatting == "sharegpt" and messages_key in example:
            # ShareGPT格式 - 直接复制对话结构，包含所有多模态信息
            example_messages = example[messages_key]
            
            for turn in example_messages:
                if not isinstance(turn, dict):
                    continue
                
                # 直接复制turn内容，保持原始结构和多模态信息
                message = dict(turn)  # 创建副本
                messages_to_log_and_return.append(message)
            
            # 如果有独立的图像信息，添加到第一个用户消息中
            if images_key in example and example[images_key]:
                # 查找第一个用户消息
                for msg in messages_to_log_and_return:
                    if msg.get(role_tag) == user_tag:
                        # 如果消息中还没有图像信息，添加
                        if "images" not in msg:
                            images = example[images_key]
                            if isinstance(images, str):
                                msg["images"] = [images]
                            elif isinstance(images, list):
                                msg["images"] = images
                        break
        else:
            # Alpaca格式或其他格式
            question = example.get("instruction", "")
            if example.get("input"):
                question = f"{question}\n{example['input']}"
            
            # 创建用户消息
            user_message = {
                role_tag: user_tag,
                content_tag: f"{question}\nAnswer:"
            }
            
            # 处理多模态信息
            if images_key in example and example[images_key]:
                images = example[images_key]
                if isinstance(images, str):
                    user_message["images"] = [images]
                elif isinstance(images, list):
                    user_message["images"] = images
            
            messages_to_log_and_return.append(user_message)
            
            # 添加助手响应（用于训练或评估的ground truth）
            response = CustomDatasetAdapter.get_response_from_item(example, dataset_specific_info, "output")
            messages_to_log_and_return.append({
                role_tag: assistant_tag,
                content_tag: response
            })

        return messages_to_log_and_return

    # Make sure get_response_from_item is correctly defined within the class or accessible
    # (It was defined as a local helper in the previous diff, assuming it's made a static method or similar)
    @staticmethod
    def get_response_from_item(item_data: Dict, dsi: Optional[Dict], fallback_key: str) -> str:
        response = ""
        if dsi and dsi.get("formatting") == "sharegpt":
            cols = dsi.get("columns", {})
            item_specific_tags = dsi.get("tags", {})
            msg_key = cols.get("messages", "conversations")
            r_tag = item_specific_tags.get("role_tag", "from") 
            c_tag = item_specific_tags.get("content_tag", "value")
            assist_tag_val = item_specific_tags.get("assistant_tag", "gpt")
            if msg_key in item_data and isinstance(item_data[msg_key], list):
                for turn_item in reversed(item_data[msg_key]):
                    if isinstance(turn_item, dict) and turn_item.get(r_tag) == assist_tag_val:
                        response = turn_item.get(c_tag, "")
                        break
        if not response and fallback_key in item_data:
            response = item_data.get(fallback_key, "")
        return response

    @staticmethod
    def process_result(
        predicted: str,
        example: Dict,
        dataset_name: str,
        dataset_specific_info: Optional[Dict] = None
    ) -> Dict:
        """处理评估结果"""
        expected_answer_raw = ""

        # --- Enhanced ShareGPT/Custom Format Detection using dataset_specific_info ---
        if dataset_specific_info and dataset_specific_info.get("formatting") == "sharegpt":
            columns = dataset_specific_info.get("columns", {})
            tags = dataset_specific_info.get("tags", {})

            messages_key = columns.get("messages", "conversations") # Default to "conversations"
            role_tag = tags.get("role_tag", "from") # Default to "from" (common in older sharegpt)
            content_tag = tags.get("content_tag", "value") # Default to "value" (common in older sharegpt)
            assistant_tag_value = tags.get("assistant_tag", "gpt") # Default to "gpt"

            if messages_key in example and isinstance(example[messages_key], list):
                # Iterate in reverse to find the last assistant message
                for turn in reversed(example[messages_key]):
                    if isinstance(turn, dict) and turn.get(role_tag) == assistant_tag_value:
                        expected_answer_raw = turn.get(content_tag, "")
                        break 
                if not expected_answer_raw:
                    print(f"Warning: Assistant turn with role '{assistant_tag_value}' and content_tag '{content_tag}' not found or 'value' is missing in '{messages_key}' for example: {example.get('id', 'unknown_id')} in dataset {dataset_name} using specific info.")
            else:
                print(f"Warning: Messages key '{messages_key}' not found or not a list in example: {example.get('id', 'unknown_id')} for dataset {dataset_name} when using specific info.")

        # --- Fallback to Original ShareGPT Format Detection (if no specific info or specific failed) ---
        if not expected_answer_raw and "conversations" in example: # Original check
            for turn in example["conversations"]:
                if turn.get("from") == "gpt": # Original hardcoded values
                    expected_answer_raw = turn.get("value", "")
                    break
            if not expected_answer_raw:
                print(f"Warning: GPT turn not found or 'value' is missing in conversations for example: {example.get('id', 'unknown_id')} (fallback).")
        # --- Fallback to Original Alpaca Format Detection ---
        elif not expected_answer_raw and "output" in example: # Original check
            expected_answer_raw = example["output"]
        # --- Multimodal and Final Fallback ---
        elif not expected_answer_raw: # If still no answer
            if any(key in example for key in ["image", "images", "video"]):
                if "output" not in example and "conversations" not in example and (not dataset_specific_info or messages_key not in example) :
                     print(f"Warning: Example {example.get('id', 'unknown_id')} contains multimodal keys but no recognized text answer field.")
            else:
                print(f"Warning: Could not determine expected answer format for example: {example.get('id', 'unknown_id')}")
        # ---

        # 验证答案
        is_correct, match_type, message = CustomDatasetAdapter.validate_answer(
            predicted,
            expected_answer_raw
        )
        
        # 返回结果
        return {
            "question": example.get("instruction", example.get("input", "")), # 尝试获取instruction或input作为问题
            "input": example.get("input", ""),
            "expected": expected_answer_raw, # 使用原始的预期答案
            "predicted": predicted,
            "normalized_predicted": CustomDatasetAdapter.normalize_answer(predicted),
            "is_correct": is_correct,
            "match_type": match_type,
            "message": message
        }
