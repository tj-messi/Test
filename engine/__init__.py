"""Inference runtime registrations."""

from .deim.postprocessor import PostProcessor
from .extre_module.tasks import OVDEIM_MG
from .obbdeim.postprocessor import OBBPostProcessor

__all__ = ["OBBPostProcessor", "OVDEIM_MG", "PostProcessor"]
