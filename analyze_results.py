
import argparse
import json
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["prompts", "models"],
    required=True,
)
parser.add_argument("--file", type=Path)
args = parser.parse_args()

# Charger les tarifs
pricing = json.loads(
    Path("pricing.json").read_text(encoding="utf-8")
)["models"]

# Selectionner le dernier benchmark du mode demande
paths = (
    [args.file]
    if args.file
    else sorted(
        Path("results").glob("benchmark_*.jsonl"),
        reverse=True,
    )
)

source = None
records = None

for path in paths:
    rows = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if rows and all(
        r.get("mode") == args.mode for r in rows
    ):
        source, records = path, rows
        break

if records is None:
    raise ValueError(
        f"Aucun benchmark trouve pour : {args.mode}"
    )

if any(r.get("error") for r in records):
    raise ValueError("Le benchmark contient des erreurs.")

if len({r["run_id"] for r in records}) != 1:
    raise ValueError("Plusieurs executions melangees.")

run_date = datetime.strptime(
    records[0]["run_id"][:8], "%Y%m%d"
).date()

# Controler que seule la dimension choisie varie
if args.mode == "prompts":
    if len({r["model_version"] for r in records}) != 1:
        raise ValueError("Plusieurs modeles detectes.")
else:
    if len({r["prompt_hash"] for r in records}) != 1:
        raise ValueError("Les prompts ne sont pas identiques.")

dimension = (
    "prompt_version"
    if args.mode == "prompts"
    else "model"
)

groups = defaultdict(list)
for record in records:
    groups[record[dimension]].append(record)

if len(groups) < 2:
    raise ValueError("Deux variantes minimum requises.")

# Verifier la comparabilite des jeux de test
reference_ids = None

for name, rows in groups.items():
    ids = [r["test_id"] for r in rows]

    if len(ids) != len(set(ids)):
        raise ValueError(f"Doublons pour {name}")

    if reference_ids is None:
        reference_ids = set(ids)
    elif set(ids) != reference_ids:
        raise ValueError(
            "Les variantes n'ont pas les memes tests."
        )


def calculate_cost(record):
    model = record["model_version"] or record["model"]

    if model not in pricing:
        raise ValueError(f"Tarif manquant : {model}")

    tariff = pricing[model]

    if (
        "valid_until" in tariff
        and run_date > date.fromisoformat(
            tariff["valid_until"]
        )
    ):
        raise ValueError(f"Tarif expire : {model}")

    required = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
    )

    if any(record.get(k) is None for k in required):
        raise ValueError("Metriques de tokens manquantes.")

    input_tokens = record["input_tokens"]
    output_tokens = record["output_tokens"]
    thinking_tokens = record.get("thinking_tokens") or 0

    if (
        input_tokens + output_tokens + thinking_tokens
        != record["total_tokens"]
    ):
        raise ValueError(
            "Comptage des tokens non couvert "
            "par le calcul de cout."
        )

    return (
        input_tokens * tariff["input"]
        + (output_tokens + thinking_tokens)
        * tariff["output"]
    ) / 1_000_000


summaries = {}

print(f"\nSource : {source.name}")
print(f"Mode : {args.mode}")
print("\n===== COMPARAISON =====")

for name, rows in sorted(groups.items()):
    cost = sum(calculate_cost(r) for r in rows)

    summary = {
        "accuracy": sum(r["correct"] for r in rows)
        / len(rows),
        "tokens": sum(r["total_tokens"] for r in rows),
        "latency": mean(r["latency_s"] for r in rows),
        "cost": cost,
    }

    summaries[name] = summary

    print(f"\n{name}")
    print(f"  Tests : {len(rows)}")
    print(f"  Qualite : {summary['accuracy']:.0%}")
    print(f"  Tokens : {summary['tokens']}")
    print(f"  Latence moyenne : {summary['latency']:.3f}s")
    print(f"  Cout estime : ${cost:.8f}")

if args.mode == "prompts" and {
    "v1", "v2"
}.issubset(summaries):
    v1, v2 = summaries["v1"], summaries["v2"]

    def reduction(a, b):
        return 100 * (a - b) / a if a else 0

    print("\n===== OPTIMISATION V2 vs V1 =====")
    print(
        f"Tokens : "
        f"{reduction(v1['tokens'], v2['tokens']):.1f}%"
    )
    print(
        f"Cout : "
        f"{reduction(v1['cost'], v2['cost']):.1f}%"
    )
    print(
        f"Latence : "
        f"{reduction(v1['latency'], v2['latency']):.1f}%"
    )
    print(
        "Non-regression :",
        "PASS" if v2["accuracy"] >= v1["accuracy"]
        else "FAIL",
    )
