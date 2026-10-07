"""Configuration and component registry for inference."""

from .workspace import GLOBAL_CONFIG, create, register
from .yaml_config import YAMLConfig

__all__ = ["GLOBAL_CONFIG", "YAMLConfig", "create", "register"]
