from __future__ import annotations

from pathlib import Path

import pytest

from cover_retrieval.utils.io import load_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def smoke_config(tmp_path: Path) -> dict:
    """Smoke config with every path redirected into a temporary directory."""
    overrides = {
        "paths": {
            "datacos_root": str(tmp_path / "da-tacos"),
            "manifests_dir": str(tmp_path / "manifests"),
            "cache_dir": str(tmp_path / "cache"),
            "runs_dir": str(tmp_path / "runs"),
            "reports_dir": str(tmp_path / "reports"),
        }
    }
    return load_config(REPO / "configs" / "smoke.yaml", overrides)
