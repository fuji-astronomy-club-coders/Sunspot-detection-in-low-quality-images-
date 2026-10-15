"""Load project configuration and resolve paths relative to the project root."""

from pathlib import Path
from pprint import pprint

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "src" / "config.yaml"


def load_config() -> dict:
    """Load and return the project configuration from config.yaml."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a YAML mapping")

    return config


def resolve_path(relative_path: str) -> Path:
    """プロジェクトルートを基準にパスを解決する。"""
    return (PROJECT_ROOT / relative_path).resolve()


config = load_config()


if __name__ == "__main__":
    config = load_config()
    for k, v in config:
        print(f"{k}:")
        pprint(v)
