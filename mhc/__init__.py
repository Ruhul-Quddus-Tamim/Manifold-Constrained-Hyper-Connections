"""mHC: Manifold-Constrained Hyper-Connections implementation."""

from mhc.data import (
    CharTokenizer,
    CharLMTokenizer,
    CharLMDataset,
    CopyDataset,
    HuggingFaceCopyDataset,
    create_dataloaders,
    create_dataloaders_from_hf,
    create_lm_dataloaders,
    create_lm_dataloader_full,
    load_hf_dataset,
)
from mhc.models import HyperTransformer, build_model_config
from mhc.models_gimhc import GIHCModule, GIHyperTransformer, build_gimhc_config
from mhc.train_utils import autoregressive_generate, get_deepseek_step_scheduler, train

__all__ = [
    "CharTokenizer",
    "CharLMTokenizer",
    "CharLMDataset",
    "CopyDataset",
    "HuggingFaceCopyDataset",
    "create_dataloaders",
    "create_dataloaders_from_hf",
    "create_lm_dataloaders",
    "create_lm_dataloader_full",
    "load_hf_dataset",
    "HyperTransformer",
    "build_model_config",
    "GIHCModule",
    "GIHyperTransformer",
    "build_gimhc_config",
    "train",
    "autoregressive_generate",
    "get_deepseek_step_scheduler",
]
