"""Dataset adapters. Importing this package registers every adapter."""

from .base import DatasetAdapter, Sample, get_adapter, register, registered
from .mklab import MKLabAdapter
from .pangaea_emed import PangaeaEMedAdapter
from .zenodo_s1 import ZenodoS1Adapter

__all__ = [
    "DatasetAdapter",
    "Sample",
    "get_adapter",
    "register",
    "registered",
    "MKLabAdapter",
    "PangaeaEMedAdapter",
    "ZenodoS1Adapter",
]
