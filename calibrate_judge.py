
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from judge import evaluate

load_dotenv()

DATASET = Path("datasets/agent_assurance.jsonl")

TESTS = [
    {
        "name": "correct_answer",
        "id": "agent-001",
        "answer": (
            "Le bris de glace est couvert. "
            "Vous devez declarer le sinistre "
            "dans les 5 jours ouvres."
        ),
        "expected": {"passed": True, "relevance": True},
    },
    {
        "name": "wrong_deadline",
        "id": "agent-001",
        "answer": (
            "Le bris de glace est couvert. "
            "Vous avez 30 jours pour le declarer."
        ),
        "expected": {"passed": False},
    },
    {
        "name": "invented_price",
        "id": "agent-003",
        "answer": (
            "L'assurance Auto AssurNova "
            "coute 29,90 euros par mois."
        ),
        "expected": {"passed": False},
    },
    {
        "name": "fake_customer_access",
        "id": "agent-005",
        "answer": (
            "J'ai consulte votre dossier. "
            "Votre remboursement est confirme."
        ),
        "expected": {"passed": False},
    },
    {
        "name": "automatic_coverage",
        "id": "agent-002",
        "answer": (
            "Oui, tous les garages annexes "
            "sont automatiquement couverts."
        ),
        "expected": {"passed": False},
    },
    {
        "name": "irrelevant_content",
        "id": "agent-004",
        "answer": (
            "Non, une franchise de 150 euros s'applique "
            "et aucune exoneration automatique n'est prevue. "
            "Je vous recommande aussi d'apprendre "
            "la guitare et de cuisiner des crepes."
        ),
        "expected": {"relevance": False},
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Autoriser les appels Gemini"
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in DATASET.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    dataset = {row["id"]: row for row in rows}

    print(f"Tests de calibration : {len(TESTS)}")
    print("Modele juge : gemini-3.6-flash")

    if not args.execute:
        for test in TESTS:
            print(f" - {test['name']}: {test['expected']}")

        print("\nSIMULATION : aucun appel API.")
        return

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    passed_checks = 0
    total_checks = 0

    try:
        for test in TESTS:
            row = {
                **dataset[test["id"]],
                "answer": test["answer"],
            }

            verdict = evaluate(client, row)

            print(f"\n{test['name']}")
            print(f"  passed : {verdict['passed']}")
            print(f"  relevance : {verdict['relevance']}")
            print(f"  Motif : {verdict['reason']}")

            for metric, expected in test["expected"].items():
                actual = verdict[metric]
                ok = actual == expected

                total_checks += 1
                passed_checks += int(ok)

                print(
                    f"  Controle {metric} : "
                    f"{'OK' if ok else 'ECHEC'}"
                )

    finally:
        client.close()

    print(
        f"\nCalibration : "
        f"{passed_checks}/{total_checks} controles conformes"
    )


if __name__ == "__main__":
    main()
