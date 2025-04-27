import torch
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from ...hparams import CLFinetuningArguments

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

class PseudoReplayTrainer(CustomSeq2SeqTrainer):
    """
    Pseudo Replay Trainer

    Inherits from CustomSeq2SeqTrainer with additional distributed training support
    """
    def __init__(self, *args, **kwargs):
        self.cl_finetuning_args = kwargs.pop("cl_finetuning_args", None)
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        """
        在分布式环境中，确保只有主进程保存模型
        """
        if not is_main_process() and not _internal_call:
            # 非主进程等待主进程完成保存
            if is_distributed():
                torch.distributed.barrier()
            return

        # 调用父类的保存方法
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)

        # 等待主进程完成保存
        if is_distributed() and not _internal_call:
            torch.distributed.barrier()
