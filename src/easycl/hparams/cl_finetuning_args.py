from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class CommonCLFinetuningArguments:
    """Base class for all continual learning finetuning arguments."""
    
    adapters_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save adapters (task-specific and shared)."}
    )
    previous_task_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the previous task model."},
    )
    current_task_id: Optional[str] = field(
        default=None,
        metadata={"help": "Current task identifier."},
    )
    prev_task_id: Optional[str] = field(
        default=None,
        metadata={"help": "Previous task identifier."},
    )
    previous_task_data: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the previous task's training data for computing Fisher information."},
    )
    cl_buffer_size: int = field(
        default=2000,
        metadata={"help": "Size of the experience replay buffer for general CL methods."},
    )

    def __post_init__(self):
        """Post initialization validation."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert the arguments to a dictionary."""
        return asdict(self)

@dataclass
class EWCArguments(CommonCLFinetuningArguments):
    """Arguments pertaining to the Elastic Weight Consolidation (EWC)."""
    
    use_ewc: bool = field(
        default=False,
        metadata={"help": "Whether or not to use Elastic Weight Consolidation (EWC)."},
    )
    ewc_lambda: float = field(
        default=10000,
        metadata={"help": "The importance coefficient for previous task(s) in EWC."},
    )
    ewc_num_samples: int = field(
        default=100,
        metadata={"help": "Number of samples to compute Fisher information matrix in EWC."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.use_ewc and not self.previous_task_data:
            logger.warning("EWC requires previous task data for computing Fisher information matrix.")

@dataclass
class LWFArguments(CommonCLFinetuningArguments):
    """Arguments pertaining to the Learning without Forgetting (LWF)."""
    
    use_lwf: bool = field(
        default=False,
        metadata={"help": "Whether or not to use Learning without Forgetting (LWF)."},
    )
    lwf_temperature: float = field(
        default=2.0,
        metadata={"help": "The temperature parameter for LWF distillation."},
    )
    lwf_alpha: float = field(
        default=0.5,
        metadata={"help": "The trade-off parameter between original loss and distillation loss in LWF."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.use_lwf and not self.previous_task_model:
            logger.warning("LWF requires previous task model for knowledge distillation.")

@dataclass
class ReplayArguments(CommonCLFinetuningArguments):
    """Arguments pertaining to the Experience Replay."""
    
    use_replay: bool = field(
        default=False,
        metadata={"help": "Whether or not to use Experience Replay."},
    )
    replay_ratio: float = field(
        default=0.3,
        metadata={"help": "Ratio of replay samples to use from each historical dataset."},
    )
    previous_task_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing previous tasks' datasets for replay."},
    )
    replay_task_list: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of dataset names to replay (e.g., 'amazon,yelp')."},
    )
    maxsamples_list: Optional[str] = field(
        default=None, 
        metadata={"help": "Comma-separated list of maximum samples for each replay dataset. If not provided, will use replay_ratio."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.use_replay and not self.previous_task_dataset:
            logger.warning("Experience Replay requires previous task dataset.")
        if self.replay_ratio <= 0 or self.replay_ratio >= 1:
            raise ValueError("`replay_ratio` must be between 0 and 1.")

@dataclass
class OLoRAArguments(CommonCLFinetuningArguments):
    """Arguments pertaining to O-LoRA."""
    
    use_olora: bool = field(
        default=False,
        metadata={"help": "Whether to use O-LoRA method."}
    )
    orthogonal_lambda: float = field(
        default=0.1,
        metadata={"help": "Weight for orthogonal regularization loss."}
    )
    l2_lambda: float = field(
        default=0.01, 
        metadata={"help": "Weight for L2 regularization on new LoRA parameters."}
    )
    olora_history_path: str = field(
        default="olora_history",
        metadata={"help": "Path to save/load O-LoRA history adapters."}
    )

@dataclass
class LAMOLArguments(CommonCLFinetuningArguments):
    """Arguments pertaining to LAMOL."""
    
    use_lamol: bool = field(
        default=False,
        metadata={"help": "Whether to use LAMOL method (pseudo-replay style) for continual learning"}
    )
    lamol_show_gen: bool = field(
        default=False,
        metadata={"help": "Prefix generated samples with 'This is a generated sample.'"}
    )
    lamol_num_samples_per_task: int = field(
        default=200,
        metadata={"help": "Number of LAMOL pseudo samples to generate per task."}
    )
    lamol_generation_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for generating LAMOL pseudo samples."}
    )
    lamol_samples_dir: str = field(
        default="lamol_samples",
        metadata={"help": "Directory for storing LAMOL pseudo samples."}
    )

@dataclass
class GEMArguments(CommonCLFinetuningArguments):
    """Arguments pertaining to Gradient Episodic Memory."""

    use_gem: bool = field(
        default=False,
        metadata={"help": "Whether to use Gradient Episodic Memory (GEM) for continual learning."}
    )
    gem_memory_strength: float = field(
        default=0.5,
        metadata={"help": "The strength parameter (penalty coefficient) for the gradient projection constraint in GEM."}
    )

@dataclass
class MOELoraArguments(CommonCLFinetuningArguments):
    """Arguments pertaining to MOELoRA method."""
    
    use_moe: bool = field(
        default=False,
        metadata={"help": "Whether to use MOELoRA for finetuning."}
    )
    
    expert_num: int = field(
        default=4,
        metadata={"help": "Number of experts in MOELoRA."}
    )
    
    task_embedding_dim: int = field(
        default=64,
        metadata={"help": "Dimension of task embedding."}
    )
    
    # 移除不需要的参数，保持简洁
    router_type: str = field(
        default="gate",
        metadata={"help": "Type of router: 'gate' or 'router'"}
    )
    
    router_capacity: int = field(
        default=1,
        metadata={"help": "Capacity of each expert in router mode"}
    )
    
    router_jitter_noise: float = field(
        default=0.0,
        metadata={"help": "Jitter noise for router during training"}
    )
    
    router_bias: bool = field(
        default=False,
        metadata={"help": "Whether to use bias in router"}
    )
    
    router_dtype: str = field(
        default="float32",
        metadata={"help": "Data type for router computation"}
    )


@dataclass
class clMoEArguments(CommonCLFinetuningArguments):
    """Arguments pertaining to cl-MoE method."""
    
    use_cl_moe: bool = field(
        default=False,
        metadata={"help": "Whether to use cl-MoE for finetuning."}
    )
    
    top_k_experts: int = field(
        default=8,
        metadata={"help": "The number of top experts to select during statistics calculation."}
    )

@dataclass
class DynamicConPETArguments(CommonCLFinetuningArguments):
    """Arguments pertaining to Dynamic ConPET method."""
    
    use_dynamic_conpet: bool = field(
        default=False,
        metadata={"help": "Whether to use Dynamic ConPET method."}
    )
    classification_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight coefficient for the dataset classification loss in Dynamic ConPET."}
    )

@dataclass
class SSRArguments(CommonCLFinetuningArguments):
    """Arguments pertaining to Self-Synthesized Rehearsal (SSR)."""
    
    use_ssr: bool = field(
        default=False,
        metadata={"help": "Whether to use Self-Synthesized Rehearsal (SSR) method."}
    )
    base_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the base model for ICL generation."}
    )
    num_shots: int = field(
        default=2,
        metadata={"help": "Number of examples to use in ICL generation."}
    )
    generation_temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for generating pseudo samples."}
    )
    n_clusters: int = field(
        default=20,
        metadata={"help": "Number of clusters for K-means clustering."}
    )
    pseudo_sample_memory: int = field(
        default=200,
        metadata={"help": "Number of pseudo samples to keep per task."}
    )
    pseudo_samples_dir: str = field(
        default="pseudo_samples",
        metadata={"help": "Directory for storing pseudo samples."}
    )

@dataclass
class ILORAArguments:
    """Arguments pertaining to I-LORA method."""
    
    use_ilora: bool = field(
        default=False,
        metadata={"help": "Whether to use I-LORA method."}
    )
    
    ema_alpha: float = field(
        default=0.25,
        metadata={"help": "EMA smoothing coefficient for I-LORA."}
    )
    
    consistency_weight: float = field(
        default=1.0,
        metadata={"help": "Weight coefficient for consistency loss in I-LORA."}
    )
    
    save_ema_adapter: bool = field(
        default=True,
        metadata={"help": "Whether to save EMA adapter in I-LORA."}
    )
    
    ema_adapter_path: str = field(
        default=None,
        metadata={"help": "Path to the EMA adapter, defaults to None for automatic path construction based on task ID."}
    )

    selective_update: bool = field(
        default=True,
        metadata={"help": "Whether to use selective update strategy in I-LORA."}
    )

    min_update_threshold: float = field(
        default=0.1,
        metadata={"help": "Minimum threshold for selective update in I-LORA."}
    )

    hidden_state_layers: List[int] = field(
        default_factory=lambda: [-1, -2, -3],
        metadata={"help": "Layers to compute hidden state consistency loss, -1 means last layer."}
    )



@dataclass
class ABSCLArguments(CommonCLFinetuningArguments):
    """Arguments pertaining to the ABSCL method."""
    
    use_abscl: bool = field(
        default=False,
        metadata={"help": "Whether to use the ABSCL continual learning method."},
    )
    abscl_orthogonal_lambda: float = field(
        default=0.1,
        metadata={"help": "Weight for the orthogonal constraint loss (lambda_1)."}
    )
    abscl_shared_l2_lambda: float = field(
        default=0.01,
        metadata={"help": "Weight for the L2 regularization on shared LoRA parameters (lambda_2)."}
    )
    abscl_stats_path: str = field(
        default="abscl_stats",
        metadata={"help": "Path to save/load ABSCL statistics (means, covariance)."}
    )

@dataclass
class PseudoReplayArguments(CommonCLFinetuningArguments):
    """Arguments pertaining to Pseudo Replay method."""
    
    use_pseudo_replay: bool = field(
        default=False,
        metadata={"help": "Whether to use Pseudo Replay method."}
    )
    base_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the base model for pseudo sample generation."}
    )
    num_samples_per_task: int = field(
        default=200,
        metadata={"help": "Number of pseudo samples to generate per task."}
    )
    generation_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for generating pseudo samples."}
    )
    pseudo_samples_dir: str = field(
        default="pseudo_samples",
        metadata={"help": "Directory for storing pseudo samples."}
    )
    num_shots: int = field(
        default=5,
        metadata={"help": "Number of examples to use in few-shot generation."}
    )

@dataclass
class CLFinetuningArguments(
    EWCArguments,
    LWFArguments,
    ReplayArguments,
    OLoRAArguments,
    LAMOLArguments,
    GEMArguments,
    MOELoraArguments,
    clMoEArguments,
    DynamicConPETArguments,
    SSRArguments,
    ABSCLArguments,
    PseudoReplayArguments,
    ILORAArguments
):
    """
    Combined arguments for all continual learning methods.
    This class inherits from all CL method argument classes.
    """

    def __post_init__(self):
        """Validate the arguments after initialization."""
        super().__post_init__()
        
        # Count active CL methods
        active_methods = sum([
            self.use_ewc,
            self.use_lwf,
            self.use_replay,
            self.use_olora,
            self.use_lamol,
            self.use_gem,
            self.use_cl_moe,
            self.use_dynamic_conpet,
            self.use_ssr,
            self.use_abscl,
            self.use_pseudo_replay
        ])
        
        if active_methods > 1:
            logger.warning("Multiple CL methods are enabled. This might lead to unexpected behavior.")
            
        # Validate method-specific requirements
        if self.use_ewc and not self.previous_task_data:
            raise ValueError("EWC requires previous task data for computing Fisher information matrix.")
            
        if self.use_lwf and not self.previous_task_model:
            raise ValueError("LWF requires previous task model for knowledge distillation.")
            
        if self.use_replay and not self.previous_task_dataset:
            raise ValueError("Experience Replay requires previous task dataset.")
            
        if self.use_ssr and not self.base_model_path:
            raise ValueError("SSR requires base model path for ICL generation.")
            
        if self.use_pseudo_replay and not self.base_model_path:
            raise ValueError("Pseudo Replay requires base model path for sample generation.") 