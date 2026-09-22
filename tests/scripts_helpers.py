"""Run repository scripts in-process against a (temporary) config dict."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def run_script(name: str, config: dict, *extra: str) -> int:
    """Write ``config`` to a YAML file next to its manifests and call ``scripts/<name>.py``."""
    config_path = Path(config["paths"]["runs_dir"]).parent / "test_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    config_path.write_text(yaml.safe_dump(clean))
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.main(["--config", str(config_path), *extra])
