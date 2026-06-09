import pytest


class TestResolveWeightsExplicit:
    def test_explicit_absolute(self, tmp_path):
        from src.models.api_server import resolve_weights_path

        w = tmp_path / "model.pt"
        w.write_text("x")
        result = resolve_weights_path(str(w), str(tmp_path))
        assert result == w

    def test_explicit_relative_in_project(self, tmp_path):
        from src.models.api_server import resolve_weights_path

        w = tmp_path / "model.pt"
        w.write_text("x")
        result = resolve_weights_path(str(w), str(tmp_path))
        assert result == w

    def test_weights_not_found_raises(self, tmp_path):
        from src.models.api_server import resolve_weights_path

        with pytest.raises(FileNotFoundError):
            resolve_weights_path("nonexistent.pt", str(tmp_path))

    def test_auto_discover_empty_dir(self, tmp_path):
        from src.models.api_server import resolve_weights_path

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            resolve_weights_path(None, str(empty))
