"""Backbone components required by inference."""

from .common import FrozenBatchNorm2d, freeze_batch_norm2d, get_activation
from .hgnetv2 import HGNetv2

__all__ = ["FrozenBatchNorm2d", "HGNetv2", "freeze_batch_norm2d", "get_activation"]
