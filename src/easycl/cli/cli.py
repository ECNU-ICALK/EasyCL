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

import os
import random
import subprocess
import sys
from enum import Enum, unique

# EasyCL specific imports
from easycl.cl_workflow.evaluator import CLEvaluator
from easycl.hparams.parser import get_cl_eval_args

# LlamaFactory core imports (potentially needed by CL workflows)
from llamafactory.extras import logging
from llamafactory.extras.misc import get_device_count, is_env_enabled, use_ray
# Assuming tuner.py is still in the same directory for now
from .tuner import run_exp


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                            |\n"
    + "|   easycl-cli <command> [options]                                  |\n"
    + "| Commands:                                                         |\n"
    + "|   cl_train      train models with continual learning                |\n"
    + "|   cl_eval        evaluate models with continual learning           |\n"
    + "|   cl_workflow   run continual learning workflow                 |\n"
    + "| Use 'easycl-cli <command> -h' for more information on a command.   |\n"
    + "-" * 70
)


logger = logging.get_logger(__name__)


@unique
class Command(str, Enum):
    CL_TRAIN = "cl_train"
    CL_EVAL = "cl_eval"
    CL_WORKFLOW = "cl_workflow"


def main():
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0) # Exit normally for help requests

    command_name = sys.argv.pop(1)

    try:
        command = Command(command_name)
    except ValueError:
        print(f"Unknown command: {command_name}")
        print(USAGE)
        sys.exit(1)

    # At this point, command is valid. Let HfArgumentParser handle sub-command help.
    if command == Command.CL_EVAL:
        # We pass remaining sys.argv to the parser which handles -h/--help
        args = get_cl_eval_args()
        evaluator = CLEvaluator(args)
        evaluator.run()
    elif command == Command.CL_WORKFLOW:
        # Import locally and let the underlying script handle args/help
        from easycl.cl_workflow.cl_train_and_eval import main as run_cl_workflow
        run_cl_workflow()
    elif command == Command.CL_TRAIN:
        # run_exp and torchrun will receive remaining args and handle help
        force_torchrun = is_env_enabled("FORCE_TORCHRUN")
        if force_torchrun or (get_device_count() > 1 and not use_ray()):
            master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
            master_port = os.getenv("MASTER_PORT", str(random.randint(20001, 29999)))
            logger.info_rank0(f"Initializing distributed tasks at: {master_addr}:{master_port}")

            try:
                import llamafactory.launcher as llamafactory_launcher
            except ImportError:
                logger.error_rank0("Could not import llamafactory.launcher. Make sure llamafactory is installed correctly.")
                sys.exit(1)

            process = subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                )
                .format(
                    nnodes=os.getenv("NNODES", "1"),
                    node_rank=os.getenv("NODE_RANK", "0"),
                    nproc_per_node=os.getenv("NPROC_PER_NODE", str(get_device_count())),
                    master_addr=master_addr,
                    master_port=master_port,
                    file_name=llamafactory_launcher.__file__,
                    args=" ".join(sys.argv[1:]), # Pass remaining args to torchrun script
                )
                .split()
            )
            sys.exit(process.returncode)
        else:
            # run_exp itself uses HfArgumentParser, which handles help
            run_exp()


if __name__ == "__main__":
    main()
