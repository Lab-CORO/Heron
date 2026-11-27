import pytest


@pytest.fixture(autouse=True)
def isolated_site(tmp_path_factory, monkeypatch):
    """Every test sees the default site, whatever site.yaml the repository has."""
    path = tmp_path_factory.mktemp("site") / "site.yaml"
    path.write_text("{}\n")
    monkeypatch.setenv("HERON_SITE", str(path))
    return path
