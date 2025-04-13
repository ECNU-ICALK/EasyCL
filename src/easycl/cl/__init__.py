"""
Continual Learning methods for LLaMA-Factory.
"""
from easycl.cl.ewc import EWC, EWCSeq2SeqTrainer, run_sft_ewc
from easycl.cl.lwf import LWF, LWFTrainer, run_sft_lwf
from easycl.cl.replay import ReplaySeq2SeqTrainer, run_sft_replay
from easycl.cl.lamol import LAMOLGenerator, LAMOLTrainer, run_sft_lamol
from easycl.cl.olora import OLoRA, OLoRATrainer, run_sft_olora
from easycl.cl.gem import  run_sft_gem
from easycl.cl.ilora import ILORA, ILORATrainer, run_sft_ilora
from easycl.cl.ssr import SSR, SSRTrainer, run_sft_ssr
from easycl.cl.pseudo_replay import PseudoReplay, PseudoReplayTrainer, run_sft_pseudo_replay
from easycl.cl.abscl import ABSCLTrainer, run_sft_abscl
from easycl.cl.dynamic_conpet import run_sft_dynamic_conpet
from easycl.cl.moe import run_sft_moelora
from easycl.cl.clmoe import run_sft_clitmoe

__all__ = [
    "CLMethod",
    "get_cl_trainer",
    "dispatch_cl_method",
    "prepare_cl_model",
    "run_cl_exp",
    "EWC",
    "EWCSeq2SeqTrainer",
    "run_sft_ewc",
    "LWF",
    "LWFTrainer",
    "run_sft_lwf",
    "ReplaySeq2SeqTrainer",
    "run_sft_replay",
    "LAMOLGenerator",
    "LAMOLTrainer",
    "run_sft_lamol",
    "OLoRA",
    "OLoRATrainer",
    "run_sft_olora",
    "run_sft_gem",
    "ILORA",
    "ILORATrainer",
    "run_sft_ilora",
    "SSR",
    "SSRTrainer",
    "run_sft_ssr",
    "PseudoReplay",
    "PseudoReplayTrainer",
    "run_sft_pseudo_replay",
    "ABSCLTrainer",
    "run_sft_abscl",
    "run_sft_dynamic_conpet",
    "run_sft_moelora",
    "run_sft_clitmoe"
]