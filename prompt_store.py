
import re
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

PREFIXES = {
    "classification": "classification_",
    "agent_assurance": "agent_",
}


def list_versions(dataset: str) -> dict[str, Path]:
    """Lister les versions disponibles pour un dataset."""

    if dataset not in PREFIXES:
        raise ValueError(f"Dataset inconnu : {dataset}")

    prefix = PREFIXES[dataset]
    versions = {}

    for path in PROMPT_DIR.glob(f"{prefix}v*.txt"):
        version = path.stem.removeprefix(prefix)

        if re.fullmatch(r"v[1-9]\d*", version):
            versions[version] = path

    return dict(
        sorted(
            versions.items(),
            key=lambda item: int(item[0][1:]),
        )
    )


def save_new_version(
    dataset: str,
    content: str,
) -> tuple[str, Path]:
    """Enregistrer un nouveau prompt sans écraser l'existant."""

    versions = list_versions(dataset)

    if not versions:
        raise ValueError("Aucune version initiale disponible")

    content = content.strip()

    if not content:
        raise ValueError("Le prompt est vide")

    required = {"{input}"}

    if dataset == "agent_assurance":
        required.add("{context}")

    missing = [
        field for field in required
        if field not in content
    ]

    if missing:
        raise ValueError(
            f"Placeholders manquants : {missing}"
        )

    next_number = max(
        int(version[1:])
        for version in versions
    ) + 1

    new_version = f"v{next_number}"

    path = PROMPT_DIR / (
        f"{PREFIXES[dataset]}{new_version}.txt"
    )

    # Mode x : interdit d'écraser un fichier existant
    with path.open("x", encoding="utf-8") as file:
        file.write(content + "\n")

    return new_version, path
