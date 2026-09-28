from repcount.data.download_model import download_model


def test_downloads_once_and_keeps_existing_file(tmp_path):
    source = tmp_path / "src.task"
    source.write_bytes(b"model-v1")
    dest = tmp_path / "models" / "pose.task"

    download_model(dest, source.as_uri())
    source.write_bytes(b"model-v2")
    download_model(dest, source.as_uri())
    assert dest.read_bytes() == b"model-v1"

    download_model(dest, source.as_uri(), overwrite=True)
    assert dest.read_bytes() == b"model-v2"
