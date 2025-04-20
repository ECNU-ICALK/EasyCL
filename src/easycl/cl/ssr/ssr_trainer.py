from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from ...hparams import CLFinetuningArguments

# Add debugprint import

def debugprint(*args, **kwargs):
    pass


class SSRTrainer(CustomSeq2SeqTrainer):
    """
    SSR (Self-Synthesized Rehearsal) Trainer
    
    Simply inherits from CustomSeq2SeqTrainer without adding additional functionality
    """
    def __init__(self, *args, **kwargs):
        debugprint(f"SSRTrainer 初始化开始")
        # Extract cl_finetuning_args before calling super().__init__
        self.cl_finetuning_args = kwargs.pop("cl_finetuning_args", None)
        debugprint(f"从 kwargs 提取的 CL Finetuning Args: {self.cl_finetuning_args}")
        super().__init__(*args, **kwargs)
        debugprint(f"SSRTrainer 初始化完成 (父类 CustomSeq2SeqTrainer 初始化已调用)")
