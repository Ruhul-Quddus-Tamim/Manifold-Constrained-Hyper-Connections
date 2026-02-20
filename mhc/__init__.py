"""mHC: Manifold-Constrained Hyper-Connections implementation."""

from mhc.data import (
    CharTokenizer,
    CopyDataset,
    HuggingFaceCopyDataset,
    create_dataloaders,
    create_dataloaders_from_hf,
    load_hf_dataset,
)
from mhc.models import HyperTransformer, build_model_config
from mhc.train_utils import autoregressive_generate, evaluate, train

__all__ = [
    "CharTokenizer",
    "CopyDataset",
    "HuggingFaceCopyDataset",
    "create_dataloaders",
    "create_dataloaders_from_hf",
    "load_hf_dataset",
    "HyperTransformer",
    "build_model_config",
    "train",
    "evaluate",
    "autoregressive_generate",
]
