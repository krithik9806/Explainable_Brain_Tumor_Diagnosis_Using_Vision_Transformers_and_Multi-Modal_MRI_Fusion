"""
Config Loader Utility for Vision Transformer & MRI Fusion Pipeline.

Provides YAML loading, recursive base-config merging, schema validation,
and a convenient attribute-accessible ConfigDict dictionary object.
"""

import os
from pathlib import Path
from typing import Any, Dict, Union
import yaml


class ConfigDict(dict):
    """
    Dictionary subclass allowing attribute-style access (e.g., config.dataset.name)
    alongside traditional key indexing (e.g., config['dataset']['name']).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = ConfigDict(value) if isinstance(value, dict) else value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")


REQUIRED_SCHEMA: Dict[str, list] = {
    "dataset": [
        "name",
        "raw_data_path",
        "processed_data_path",
        "num_classes",
        "class_names",
        "input_channels",
        "image_size",
    ],
    "model": [
        "backbone",
        "pretrained",
        "input_channel_override",
    ],
    "training": [
        "batch_size",
        "num_epochs",
        "optimizer",
        "learning_rate",
        "lr_scheduler",
        "weight_decay",
        "loss_function",
    ],
    "augmentation": [
        "flip_prob",
        "rotation_degrees",
        "elastic_deformation",
    ],
    "splits": [
        "train",
        "val",
        "test",
        "random_seed",
        "patient_level",
    ],
    "checkpointing": [
        "save_dir",
        "save_every_n_epochs",
        "metric_to_monitor",
    ],
    "logging": [
        "wandb_project_name",
        "log_every_n_steps",
    ],
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dictionary `override` into dictionary `base`.
    Values in `override` take precedence over `base`.

    Args:
        base (Dict[str, Any]): Base dictionary with default values.
        override (Dict[str, Any]): Override dictionary with specific settings.

    Returns:
        Dict[str, Any]: Deep-merged dictionary.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_config(config: Dict[str, Any], config_path: str = "") -> None:
    """
    Validate that mandatory sections and parameters exist in the configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary to validate.
        config_path (str): File path for error context.

    Raises:
        ValueError: If a required top-level section or nested parameter is missing.
    """
    path_context = f" in '{config_path}'" if config_path else ""

    # Check top-level experiment_name
    if "experiment_name" not in config or not config["experiment_name"]:
        raise ValueError(f"Missing required field 'experiment_name'{path_context}.")

    # Check sections and sub-fields
    for section, fields in REQUIRED_SCHEMA.items():
        if section not in config:
            raise ValueError(f"Missing required top-level section '{section}'{path_context}.")
        if not isinstance(config[section], dict):
            raise ValueError(f"Section '{section}' must be a dictionary{path_context}.")

        for field in fields:
            if field not in config[section]:
                raise ValueError(
                    f"Missing required field '{field}' in section '{section}'{path_context}."
                )

    # Validate split sum
    splits = config["splits"]
    split_sum = splits["train"] + splits["val"] + splits["test"]
    if not (0.99 <= split_sum <= 1.01):
        raise ValueError(
            f"Train/Val/Test splits must sum to 1.0, got {split_sum:.3f}{path_context}."
        )


def load_config(config_path: Union[str, Path]) -> ConfigDict:
    """
    Load a YAML configuration file, merge base defaults if specified via `base_config`,
    validate mandatory schema fields, and return an attribute-accessible `ConfigDict`.

    Args:
        config_path (str or Path): Path to target YAML configuration file.

    Returns:
        ConfigDict: Validated and merged configuration dictionary object.

    Raises:
        FileNotFoundError: If `config_path` or `base_config` file does not exist.
        ValueError: If required configuration fields are missing or invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    # Resolve base_config inheritance if specified
    if "base_config" in raw_config and raw_config["base_config"]:
        base_rel_path = Path(raw_config["base_config"])

        # Determine path to base_config: check relative to current config file dir first, then relative to working dir
        if (config_path.parent / base_rel_path).exists():
            base_full_path = config_path.parent / base_rel_path
        elif base_rel_path.exists():
            base_full_path = base_rel_path
        else:
            raise FileNotFoundError(
                f"Base configuration file '{base_rel_path}' specified in '{config_path}' not found."
            )

        with open(base_full_path, "r", encoding="utf-8") as bf:
            base_config = yaml.safe_load(bf) or {}

        # Deep merge base_config with raw_config overrides
        merged_config = deep_merge(base_config, raw_config)
    else:
        merged_config = raw_config

    # Validate complete merged configuration
    validate_config(merged_config, str(config_path))

    # Return as attribute-accessible ConfigDict
    return ConfigDict(merged_config)
