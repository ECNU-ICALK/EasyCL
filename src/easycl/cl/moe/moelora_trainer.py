# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, Dict, Optional

from llamafactory.extras import logging
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from llamafactory.hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class MoELoRATrainer(CustomSeq2SeqTrainer):
    """
    Trainer for MoE-LoRA models.
    
    This trainer extends the CustomSeq2SeqTrainer to handle MoE-LoRA specific configurations.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            finetuning_args=finetuning_args,
            processor=processor,
            gen_kwargs=gen_kwargs,
            **kwargs
        )
        self.expert_num = getattr(finetuning_args, "expert_num", None)
        self.task_embedding_dim = getattr(finetuning_args, "task_embedding_dim", None)
        
        # Log MoE-LoRA specific configuration
        if self.expert_num is not None and self.expert_num > 1:
            logger.info_rank0(
                f"MoE-LoRA configuration: expert_num={self.expert_num}, "
                f"task_embedding_dim={self.task_embedding_dim}"
            )