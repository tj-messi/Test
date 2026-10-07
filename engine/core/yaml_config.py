"""Minimal YAML loader for the released inference graphs."""

from pathlib import Path

import yaml

from .workspace import GLOBAL_CONFIG, create


class YAMLConfig:
    """Instantiate only the model and postprocessor declared by a YAML file."""

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        self._registry = {
            key: value.copy() if isinstance(value, dict) else value
            for key, value in GLOBAL_CONFIG.items()
        }
        self._registry.update(
            {
                key: value
                for key, value in self.config.items()
                if not isinstance(value, dict)
            }
        )
        for key, value in self.config.items():
            if isinstance(value, dict) and key in self._registry:
                self._registry[key].update(value)
        self._model = None
        self._postprocessor = None

    @property
    def model(self):
        if self._model is None:
            self._model = create(self.config["model"], self._registry)
        return self._model

    @property
    def postprocessor(self):
        if self._postprocessor is None:
            self._postprocessor = create(self.config["postprocessor"], self._registry)
        return self._postprocessor
