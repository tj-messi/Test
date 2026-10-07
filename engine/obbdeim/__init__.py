"""Oriented detection components required by inference."""

from .decoder import OBBDFINETransformer
from .postprocessor import OBBPostProcessor

__all__ = ["OBBDFINETransformer", "OBBPostProcessor"]
