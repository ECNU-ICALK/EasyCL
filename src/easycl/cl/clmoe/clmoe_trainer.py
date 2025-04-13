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
    from easycl.hparams import CLFinetuningArguments


logger = logging.get_logger(__name__)


class CLMoETrainer(CustomSeq2SeqTrainer): # Renamed from MoELoRATrainer
    """
    Trainer for CL-MoE models.
    
    This trainer extends the CustomSeq2SeqTrainer to handle CL-MoE specific configurations.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        cl_finetuning_args: "CLFinetuningArguments",
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
        
        # Store cl_finetuning_args
        self.cl_finetuning_args = cl_finetuning_args
        
        # Get CL-MoE specific parameters from cl_finetuning_args
        self.use_cl_moe = cl_finetuning_args.use_cl_moe
        self.expert_num = cl_finetuning_args.expert_num
        self.task_embedding_dim = cl_finetuning_args.task_embedding_dim
        
        # Log CL-MoE specific configuration if enabled
        if self.use_cl_moe and self.expert_num is not None and self.expert_num > 1:
            logger.info_rank0(
                f"CL-MoE configuration: expert_num={self.expert_num}, "
                f"task_embedding_dim={self.task_embedding_dim}"
            )
        # Log standard MoE-LoRA config if that's used (though args check should prevent both)
        elif cl_finetuning_args.use_moe and self.expert_num is not None and self.expert_num > 1:
             logger.info_rank0(
                 f"MoE-LoRA configuration: expert_num={self.expert_num}, "
                 f"task_embedding_dim={self.task_embedding_dim}"
             ) 