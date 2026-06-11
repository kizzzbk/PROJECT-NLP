"""
Configuration Management
========================
Load, merge, and validate YAML configuration files with CLI override support.
"""

import os
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Config(dict):
    """
    A dictionary subclass that allows attribute-style access.
    
    Example:
        cfg = Config({'model': {'name': 'bilstm', 'hidden_dim': 256}})
        cfg.model.name  # => 'bilstm'
        cfg.model.hidden_dim  # => 256
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"Config has no attribute '{key}'. "
                f"Available keys: {list(self.keys())}"
            )

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def to_dict(self) -> dict:
        """Convert back to a regular nested dict (for serialization)."""
        result = {}
        for key, value in self.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def freeze(self) -> str:
        """Serialize config to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load a single YAML config file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Config object with attribute-style access.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    
    if raw is None:
        raw = {}
    
    return Config(raw)


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override dict into base dict.
    Override values take precedence.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_configs(
    base_path: Union[str, Path],
    *override_paths: Union[str, Path],
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    Load and merge multiple config files with priority ordering.
    
    Priority (lowest to highest):
        base.yaml < model_specific.yaml < cli_overrides
    
    Args:
        base_path: Path to the base config file.
        *override_paths: Additional config files to merge (in order).
        cli_overrides: Dict of CLI argument overrides (highest priority).
        
    Returns:
        Merged Config object.
        
    Example:
        config = merge_configs(
            'configs/base.yaml',
            'configs/model_bilstm.yaml',
            cli_overrides={'training.epochs': 50}
        )
    """
    merged = load_config(base_path).to_dict()
    
    for path in override_paths:
        override = load_config(path).to_dict()
        merged = _deep_merge(merged, override)
    
    # Apply CLI overrides (dot-notation keys)
    if cli_overrides:
        for key, value in cli_overrides.items():
            _set_nested(merged, key, value)
    
    return Config(merged)


def _set_nested(d: dict, dot_key: str, value: Any):
    """
    Set a value in a nested dict using dot notation.
    
    Example:
        _set_nested(d, 'training.epochs', 50)
        # => d['training']['epochs'] = 50
    """
    keys = dot_key.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def save_config(config: Config, save_path: Union[str, Path]):
    """Save a Config object to a YAML file (for experiment snapshots)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config.to_dict(),
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def get_project_root() -> Path:
    """Get the project root directory (where configs/ lives)."""
    # Walk up from this file until we find configs/
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").is_dir():
            return parent
    # Fallback to CWD
    return Path.cwd()
