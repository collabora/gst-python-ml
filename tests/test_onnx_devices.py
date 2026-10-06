import sys
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")

BASE_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIRECTORY / "plugins" / "python"))

from engine.onnx_engine import ONNXEngine, ort  # noqa: E402


def test_selects_coreml_execution_provider(monkeypatch):
    provider = "CoreMLExecutionProvider"
    monkeypatch.setattr(ort, "get_available_providers", lambda: [provider])
    engine = ONNXEngine()

    assert engine._provider_for("coreml") == provider
