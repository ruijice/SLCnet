from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8", newline="\n")
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(read_text(path))


def write_json(path: str | Path, data: Any) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    return target


def read_config(path: str | Path) -> dict[str, Any]:
    text = read_text(path)
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Install PyYAML or use JSON-compatible config files.") from exc
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return loaded

