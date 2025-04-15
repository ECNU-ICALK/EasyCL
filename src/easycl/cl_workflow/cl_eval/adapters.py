"""持续学习评估适配器模块，用于适配不同格式的数据集"""

import re
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class AlpacaEvalAdapter:
    """将Alpaca格式数据适配到MMLU评估格式"""
    
    @staticmethod
    def convert_example(example: Dict, dataset_options: Dict[str, Dict], dataset_name: str) -> Dict:
        """将Alpaca格式转换为评估器需要的格式"""
        options = dataset_options[dataset_name]["options"]
        # 构造A/B/C/D到实际选项的映射
        option_map = {chr(65 + i): opt for i, opt in enumerate(options)}
        # 找到答案对应的选项标签
        answer_label = None
        for label, opt in option_map.items():
            if opt == example["answer"]:
                answer_label = label
                break
        
        if answer_label is None:
            raise ValueError(
                f"Answer '{example['answer']}' not found in options: {options}"
            )
        
        result = {
            "question": example["question"],
            "answer": answer_label
        }
        # 添加选项（如果问题中没有包含选项说明）
        if "Choose one from the option:" not in example["question"]:
            for label, opt in option_map.items():
                result[label] = opt
            
        return result

    @staticmethod
    def format_example(
        example: Dict,
        template: str,
        subject_name: str,
        dataset_options: Dict[str, Dict],
        dataset_name: str,
        support_set: Optional[List[Dict]] = None
    ) -> List[Dict[str, str]]:
        """格式化示例为评估模板格式"""
        converted = AlpacaEvalAdapter.convert_example(example, dataset_options, dataset_name)
        
        # 构造问题文本
        question_text = converted["question"]
        
        # 只有当问题中没有包含选项说明时才添加选项
        if "Choose one from the option:" not in question_text:
            options_text = "\n".join([
                f"{label}. {converted[label]}"
                for label in sorted(converted.keys())
                if label != "question" and label != "answer"
            ])
            question_text = f"{question_text}\n{options_text}"
        
        # 返回消息列表
        return [
            {"role": "user", "content": f"{question_text}\nAnswer:"},
            {"role": "assistant", "content": converted["answer"]}
        ]

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
        expected: str,
        valid_options: List[str]
    ) -> Tuple[bool, str, str]:
        """验证答案是否正确"""
        # 规范化答案
        norm_predicted = CustomDatasetAdapter.normalize_answer(predicted)
        norm_expected = CustomDatasetAdapter.normalize_answer(expected)
        norm_options = [CustomDatasetAdapter.normalize_answer(opt) for opt in valid_options]
        
        # 1. 完全匹配
        if norm_predicted == norm_expected:
            return True, "exact", "Exact match"
            
        # 2. 选项匹配
        for opt in norm_options:
            if opt in norm_predicted:
                if opt == norm_expected:
                    return True, "option", "Option match"
                    
        # 3. 子字符串匹配 (替换原来的关键词匹配)
        # 检查规范化后的预期答案是否是规范化后的预测答案的子集
        if norm_expected in norm_predicted:
             return True, "substring", "Substring match (expected in predicted)"
            
        # 4. 模糊匹配
        for opt in norm_options:
            if opt in norm_predicted or norm_predicted in opt:
                if opt == norm_expected:
                    return True, "fuzzy", "Fuzzy match"
        
        return False, "wrong", "Wrong answer"

    @staticmethod
    def format_example(
        example: Dict,
        template: str,
        subject_name: str,
        dataset_options: Dict[str, Dict],
        dataset_name: str,
        support_set: Optional[List[Dict]] = None
    ) -> List[Dict[str, str]]:
        """格式化示例为评估格式"""
        messages = []
        
        # 添加few-shot样例
        if support_set:
            # 构造few-shot示例
            for support_example in support_set:
                # 构造few-shot示例文本
                support_text = support_example.get("instruction", "")
                if support_example.get("input"):
                    support_text = f"{support_text}\n{support_example['input']}"
                messages.extend([
                    {"role": "user", "content": f"{support_text}\nAnswer:"},
                    {"role": "assistant", "content": support_example["output"]}
                ])
        
        # 构造当前问题文本
        question_text = example.get("instruction", "")
        if example.get("input"):
            question_text = f"{question_text}\n{example['input']}"
        
        # 添加当前问题
        messages.extend([
            {"role": "user", "content": f"{question_text}\nAnswer:"},
            {"role": "assistant", "content": example["output"]}
        ])
        
        return messages

    @staticmethod
    def process_result(
        predicted: str,
        example: Dict,
        dataset_options: Dict[str, Dict],
        dataset_name: str
    ) -> Dict:
        """处理评估结果"""
        # 获取有效选项
        valid_options = dataset_options[dataset_name]["options"]
        
        # 验证答案
        is_correct, match_type, message = CustomDatasetAdapter.validate_answer(
            predicted,
            example["output"],
            valid_options
        )
        
        # 返回结果
        return {
            "question": example["instruction"],
            "input": example.get("input", ""),
            "expected": example["output"],
            "predicted": predicted,
            "normalized_predicted": CustomDatasetAdapter.normalize_answer(predicted),
            "is_correct": is_correct,
            "match_type": match_type,
            "message": message
        }