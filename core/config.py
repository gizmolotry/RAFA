""" Configuration loader for RAFA training. """

import os
import yaml

def load_config(path: str | None = None) -> dict:
    """
    Load RAFA configuration from a YAML file.
    """
    if path is None:
        path = os.environ.get("RAFA_CONFIG_PATH", "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)
