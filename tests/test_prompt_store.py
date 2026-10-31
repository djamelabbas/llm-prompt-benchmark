
import pytest
import prompt_store


@pytest.fixture
def prompt_dir(tmp_path, monkeypatch):
    # Utiliser un dossier temporaire pour les tests
    monkeypatch.setattr(
        prompt_store, "PROMPT_DIR", tmp_path
    )

    (tmp_path / "agent_v1.txt").write_text(
        "Contexte: {context}\nQuestion: {input}",
        encoding="utf-8",
    )

    (tmp_path / "agent_v2.txt").write_text(
        "Reponds avec {context} et {input}",
        encoding="utf-8",
    )

    return tmp_path


def test_list_versions(prompt_dir):
    versions = prompt_store.list_versions(
        "agent_assurance"
    )

    assert list(versions) == ["v1", "v2"]


def test_create_new_version(prompt_dir):
    version, path = prompt_store.save_new_version(
        "agent_assurance",
        "Nouveau prompt: {context} - {input}",
    )

    assert version == "v3"
    assert path.exists()
    assert "Nouveau prompt" in path.read_text(
        encoding="utf-8"
    )


def test_missing_placeholder(prompt_dir):
    with pytest.raises(ValueError):
        prompt_store.save_new_version(
            "agent_assurance",
            "Prompt sans contexte: {input}",
        )


def test_existing_versions_unchanged(prompt_dir):
    original = (
        prompt_dir / "agent_v2.txt"
    ).read_text(encoding="utf-8")

    prompt_store.save_new_version(
        "agent_assurance",
        "Nouvelle version: {context} {input}",
    )

    assert (
        prompt_dir / "agent_v2.txt"
    ).read_text(encoding="utf-8") == original
