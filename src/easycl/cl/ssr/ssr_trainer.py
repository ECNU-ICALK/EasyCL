import torch
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from ...hparams import CLFinetuningArguments
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




class SSRTrainer(CustomSeq2SeqTrainer):
    """
    SSR (Self-Synthesized Rehearsal) Trainer

    Simply inherits from CustomSeq2SeqTrainer without adding additional functionality
    """
    def __init__(self, *args, **kwargs):
        rank = get_rank()
        debugprint(f"SSRTrainer 初始化开始")
        # Extract cl_finetuning_args before calling super().__init__
        self.cl_finetuning_args = kwargs.pop("cl_finetuning_args", None)
        debugprint(f"从 kwargs 提取的 CL Finetuning Args: {self.cl_finetuning_args}")
        super().__init__(*args, **kwargs)
        debugprint(f"SSRTrainer 初始化完成 (父类 CustomSeq2SeqTrainer 初始化已调用)")

    def save_model(self, output_dir=None, _internal_call=False):
        """
        在分布式环境中，确保只有主进程保存模型
        """
        if not is_main_process() and not _internal_call:
            debugprint(f"非主进程 (rank={get_rank()})，跳过保存模型")
            # 等待主进程完成保存
            if is_distributed():
                torch.distributed.barrier()
            return

        # 调用父类的保存方法
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        debugprint(f"模型保存完成，输出目录: {output_dir or self.args.output_dir}")

        # 等待主进程完成保存
        if is_distributed() and not _internal_call:
            torch.distributed.barrier()
            debugprint(f"进程 rank={get_rank()} 等待模型保存完成")
