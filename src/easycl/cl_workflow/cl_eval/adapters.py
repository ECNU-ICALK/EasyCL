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
        messages_to_log_and_return = []

        # --- Get base tags for the output message structure (for LlamaFactory model) ---
        output_struct_tags = dataset_specific_info.get("tags", {}) if dataset_specific_info else {}
        output_role_tag = output_struct_tags.get("role_tag", "role")
        output_content_tag = output_struct_tags.get("content_tag", "content")
        output_user_role_val = output_struct_tags.get("user_tag", "user")
        output_assistant_role_val = output_struct_tags.get("assistant_tag", "assistant")
        # Allow for a system role in the output structure if defined
        output_system_role_val = output_struct_tags.get("system_tag", "system")

        # --- Get keys/tags for parsing the *input data* (e.g., ShareGPT fields) ---
        ds_columns = dataset_specific_info.get("columns", {}) if dataset_specific_info else {}
        ds_messages_key = ds_columns.get("messages", "conversations") # Key for the list of turns

        ds_data_tags = dataset_specific_info.get("tags", {}) if dataset_specific_info else {}
        # Keys *within each turn* of a ShareGPT message list
        ds_role_key_in_turn = ds_data_tags.get("role_tag", "from") # e.g., "role" or "from"
        ds_content_key_in_turn = ds_data_tags.get("content_tag", "value") # e.g., "content" or "value"
        # Actual role values *within each turn* of ShareGPT data
        ds_user_role_val_in_turn = ds_data_tags.get("user_tag", "human")
        ds_assistant_role_val_in_turn = ds_data_tags.get("assistant_tag", "gpt")
        ds_system_role_val_in_turn = ds_data_tags.get("system_tag", "system") # Role value for system in input data

        # Helper to get assistant response from an example item (defined in the previous step)
        # This helper should be part of the class or accessible here.
        # For this edit, we assume `get_response_from_item` is correctly defined as before.
        # staticmethod def get_response_from_item(item_data: Dict, dsi: Optional[Dict], fallback_key: str) -> str: ...

        # 1. Process Few-Shot Examples (support_set)
        if support_set:
            for s_ex in support_set:
                is_support_sharegpt = (dataset_specific_info and
                                       dataset_specific_info.get("formatting") == "sharegpt" and
                                       ds_messages_key in s_ex and
                                       isinstance(s_ex[ds_messages_key], list))

                if is_support_sharegpt:
                    for turn in s_ex[ds_messages_key]:
                        if not isinstance(turn, dict) or ds_role_key_in_turn not in turn:
                            continue
                        role_from_data = turn[ds_role_key_in_turn]
                        content_from_data = turn.get(ds_content_key_in_turn, "")
                        
                        mapped_role = output_user_role_val # Default
                        if role_from_data == ds_assistant_role_val_in_turn:
                            mapped_role = output_assistant_role_val
                        elif role_from_data == ds_system_role_val_in_turn:
                            mapped_role = output_system_role_val
                        # else it's user (ds_user_role_val_in_turn) or unmapped, defaults to output_user_role_val

                        messages_to_log_and_return.append({output_role_tag: mapped_role, output_content_tag: content_from_data})
                else: # Not ShareGPT or malformed for support example
                    s_prompt_text = s_ex.get("instruction", "")
                    if s_ex.get("input"):
                        s_prompt_text = f"{s_prompt_text}\\n{s_ex['input']}"
                    s_response = CustomDatasetAdapter.get_response_from_item(s_ex, dataset_specific_info, "output")
                    messages_to_log_and_return.append({output_role_tag: output_user_role_val, output_content_tag: f"{s_prompt_text}\\nAnswer:"})
                    messages_to_log_and_return.append({output_role_tag: output_assistant_role_val, output_content_tag: s_response})

        # 2. Process the Main Example
        # This is the ground truth assistant response for the main example.
        main_example_ground_truth_response = CustomDatasetAdapter.get_response_from_item(example, dataset_specific_info, "output")

        is_main_sharegpt = (dataset_specific_info and
                            dataset_specific_info.get("formatting") == "sharegpt" and
                            ds_messages_key in example and
                            isinstance(example[ds_messages_key], list))

        if is_main_sharegpt:
            # For ShareGPT, the example[ds_messages_key] contains the full interaction,
            # including the user's query and often the assistant's ground truth response as the last turn.
            # LlamaFactory's template.encode_oneturn expects the full conversation, where the
            # last assistant message is treated as the label.
            for turn_idx, turn in enumerate(example[ds_messages_key]):
                if not isinstance(turn, dict) or ds_role_key_in_turn not in turn:
                    continue
                role_from_data = turn[ds_role_key_in_turn]
                content_from_data = turn.get(ds_content_key_in_turn, "")
                
                mapped_role = output_user_role_val # Default to user
                if role_from_data == ds_assistant_role_val_in_turn:
                    mapped_role = output_assistant_role_val
                elif role_from_data == ds_system_role_val_in_turn:
                    mapped_role = output_system_role_val
                # else it's user (ds_user_role_val_in_turn) or unmapped, defaults to output_user_role_val

                messages_to_log_and_return.append({output_role_tag: mapped_role, output_content_tag: content_from_data})
            
            # Ensure the last message is indeed the assistant's ground truth if the ShareGPT data structure implies it.
            # If the ShareGPT data ends with a user message (i.e., it's a prompt for generation and ground truth is separate),
            # then `main_example_ground_truth_response` (extracted earlier) must be appended.
            # However, typical ShareGPT for SFT/eval includes the assistant's response in the list.
            # The `get_response_from_item` should correctly extract this last assistant response if it's part of the turns.
            # If `example[ds_messages_key]` does NOT contain the final assistant response (e.g. if it's only a prompt),
            # we might need to append `main_example_ground_truth_response` manually.
            # For now, assume ShareGPT data for eval includes the target assistant response.
            # If the last message in messages_to_log_and_return is not an assistant message,
            # or if its content doesn't match main_example_ground_truth_response, this could be an issue.
            # This logic relies on template.encode_oneturn correctly identifying the label from the last assistant message.

        else: # Not ShareGPT or malformed for main example
            question_text = example.get("instruction", "")
            if example.get("input"):
                question_text = f"{question_text}\\n{example['input']}"
            messages_to_log_and_return.append({output_role_tag: output_user_role_val, output_content_tag: f"{question_text}\\nAnswer:"})
            # The ground truth (label) for the model
            messages_to_log_and_return.append({output_role_tag: output_assistant_role_val, output_content_tag: main_example_ground_truth_response})

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