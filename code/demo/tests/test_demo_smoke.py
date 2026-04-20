from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_compare_models():
    app_path = Path(__file__).resolve().parents[1] / "gradio_app.py"
    spec = spec_from_file_location("gradio_app", app_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.compare_models


def test_compare_models_returns_string() -> None:
    compare_models = _load_compare_models()
    result = compare_models("great movie", "dora", 8)
    assert isinstance(result, str)
    assert "Method: dora" in result
