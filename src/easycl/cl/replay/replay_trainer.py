# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
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

from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.extras.logging import get_logger
from llamafactory.hparams import FinetuningArguments
from easycl.hparams.cl_finetuning_args import CLFinetuningArguments

logger = get_logger(__name__)

class ReplaySeq2SeqTrainer(CustomSeq2SeqTrainer):
    """
    Seq2SeqTrainer for Experience Replay
    Inherits from base CustomSeq2SeqTrainer without additional functionality
    Replay mechanism is implemented through dataset merging in the workflow
    """
    
    def __init__(self, finetuning_args: "FinetuningArguments", cl_finetuning_args: "CLFinetuningArguments", **kwargs):
        super().__init__(finetuning_args=finetuning_args, **kwargs)
        self.cl_finetuning_args = cl_finetuning_args