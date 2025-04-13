from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from ...hparams import CLFinetuningArguments

class SSRTrainer(CustomSeq2SeqTrainer):
    """
    SSR (Self-Synthesized Rehearsal) Trainer
    
    Simply inherits from CustomSeq2SeqTrainer without adding additional functionality
    """
    def __init__(self, *args, **kwargs):
        self.cl_finetuning_args = kwargs.pop("cl_finetuning_args", None)
        super().__init__(*args, **kwargs)
