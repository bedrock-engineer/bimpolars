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


def test_fetch_takes_the_url_in_the_bucket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bucket = tmp_path / "bucket"
    (bucket / "sub").mkdir(parents=True)
    (bucket / "sub" / "a b.parquet").write_bytes(b"PAR1")
    monkeypatch.setattr(data, "DATA_URL", bucket.as_uri() + "/")
    monkeypatch.setenv(data.DATA_DIR_ENV, str(tmp_path / "local"))
    assert data.bucket_key(f"{data.DATA_URL}sub/a%20b.parquet") == "sub/a b.parquet"
    assert data.bucket_key("sub/a b.parquet") == "sub/a b.parquet"
    path = data.fetch(f"{data.DATA_URL}sub/a%20b.parquet")
    assert path == tmp_path / "local" / "sub" / "a b.parquet"
    assert data.fetch(f"{data.DATA_URL}sub/a b.parquet") == path
    with pytest.raises(ValueError, match="not in the bucket"):
        data.fetch("https://example.com/sub/a%20b.parquet")
    # the real bucket answers over http and https alike
    monkeypatch.setattr(data, "DATA_URL", "http://r2.eu.bedrock.engineer/bimpolars/")
    assert data.bucket_key("https://r2.eu.bedrock.engineer/bimpolars/x/y.ifc") == "x/y.ifc"
    assert data.bucket_key("http://r2.eu.bedrock.engineer/bimpolars/x/y.ifc") == "x/y.ifc"
    # a key, from a URL or not, stays inside the data folder
    for outside in (
        "https://r2.eu.bedrock.engineer/bimpolars//etc/hostname",
        "https://r2.eu.bedrock.engineer/bimpolars/%2e%2e/%2e%2e/x",
        "http://r2.eu.bedrock.engineer/bimpolars/a/../../x",
        "/etc/hostname",
        "../x",
        r"..\x",
        r"C:\mirror\x\y.ifc",
        "",
    ):
        with pytest.raises(ValueError, match="inside the bucket"):
            data.bucket_key(outside)
    with pytest.raises(ValueError, match="inside the bucket"):
        data.fetch("/etc/hostname")


def test_a_failed_download_leaves_no_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    (bucket / "model.ifc").write_bytes(b"ISO-10303-21;")
    monkeypatch.setattr(data, "DATA_URL", bucket.as_uri() + "/")
    monkeypatch.setenv(data.DATA_DIR_ENV, str(tmp_path / "local"))

    def interrupted(source, destination) -> None:
        destination.write(b"ISO")
        raise ConnectionResetError("cut off")

    monkeypatch.setattr(data.shutil, "copyfileobj", interrupted)
    with pytest.raises(ConnectionResetError):
        data.fetch("model.ifc")
    assert list((tmp_path / "local").iterdir()) == []
