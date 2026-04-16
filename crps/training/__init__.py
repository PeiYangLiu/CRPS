"""CRPS SFT Training Module.

Provides dataset handling and trainer for supervised fine-tuning
on CRPS-synthesized mathematical reasoning data.
"""

from crps.training.dataset import CRPSDataset, DataCollator
from crps.training.trainer import CRPSTrainer

__all__ = ["CRPSDataset", "DataCollator", "CRPSTrainer"]
