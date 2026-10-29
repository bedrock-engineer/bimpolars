import urllib.error
from pathlib import Path

import pytest

from bimpolars import data


def test_data_dir_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(data.DATA_DIR_ENV, str(tmp_path))
    assert data.data_dir() == tmp_path
    monkeypatch.delenv(data.DATA_DIR_ENV)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert data.data_dir() == tmp_path / "cache" / "bimpolars"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert data.data_dir() == Path.home() / ".cache" / "bimpolars"


def test_fetch_finds_a_local_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(data.DATA_DIR_ENV, str(tmp_path))
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "model.ifc").write_text("ISO-10303-21;")
    assert data.fetch("nested/model.ifc") == tmp_path / "nested" / "model.ifc"
    with pytest.raises(FileNotFoundError, match=data.DATA_DIR_ENV):
        data.fetch("missing.ifc", download=False)


def test_fetch_downloads_a_missing_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bucket = tmp_path / "bucket"
    (bucket / "sub").mkdir(parents=True)
    (bucket / "sub" / "a b.parquet").write_bytes(b"PAR1")
    monkeypatch.setattr(data, "DATA_URL", bucket.as_uri() + "/")
    monkeypatch.setenv(data.DATA_DIR_ENV, str(tmp_path / "local"))
    path = data.fetch("sub/a b.parquet")
    assert path == tmp_path / "local" / "sub" / "a b.parquet"
    assert path.read_bytes() == b"PAR1"
    assert not path.with_name(path.name + ".part").exists()
    with pytest.raises(urllib.error.URLError):
        data.fetch("sub/missing.parquet")
