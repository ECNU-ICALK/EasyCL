"""
持续学习评估框架模块
"""

from easycl.cl_workflow.cl_eval.evaluator import CLEvalEvaluator
from easycl.cl_workflow.cl_eval.adapters import AlpacaEvalAdapter, CustomDatasetAdapter
from easycl.cl_workflow.cl_eval.cl_metrics import CLMetricsCalculator

__all__ = [
    "CLEvalEvaluator",
    "AlpacaEvalAdapter", 
    "CustomDatasetAdapter",
    "CLMetricsCalculator"
]
