
import argparse
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]


def calculate_cost(row, tariffs):
    model = row.get("model_version") or row["model"]

    if model not in tariffs:
        raise ValueError(f"Tarif manquant : {model}")

    tariff = tariffs[model]

    run_date = datetime.strptime(
        row["run_id"][:8], "%Y%m%d"
    ).date()

    if (
        tariff.get("valid_until")
        and run_date > date.fromisoformat(
            tariff["valid_until"]
        )
    ):
        raise ValueError(f"Tarif expire : {model}")

    inputs = row.get("input_tokens")
    outputs = row.get("output_tokens")
    thoughts = row.get("thinking_tokens") or 0
    total = row.get("total_tokens")

    if any(x is None for x in (inputs, outputs, total)):
        raise ValueError("Metriques de tokens manquantes")

    if inputs + outputs + thoughts != total:
        raise ValueError(
            "Comptage des tokens incomplet"
        )

    return (
        inputs * tariff["input"]
        + (outputs + thoughts) * tariff["output"]
    ) / 1_000_000


def report_judge_quality(groups):
    # Indexer les evaluations par reponse d'origine
    evaluations = {}

    for path in sorted(RESULTS.glob("judge_*.jsonl")):
        for item in load_jsonl(path):
            verdict = item.get("evaluation")

            if item.get("error") or not isinstance(verdict, dict):
                continue

            key = (
                item.get("source_run_id"),
                item.get("test_id"),
                item.get("model"),
                item.get("prompt_version"),
            )

            if all(key):
                # En cas de reevaluation, garder la plus recente
                evaluations[key] = verdict

    print("\n===== QUALITE LLM-AS-A-JUDGE =====")

    for variant, rows in sorted(groups.items()):
        assessed = []

        for row in rows:
            key = (
                row["run_id"],
                row["test_id"],
                row["model"],
                row["prompt_version"],
            )

            verdict = evaluations.get(key)

            if verdict is not None:
                assessed.append((row, verdict))

        n = len(assessed)
        total = len(rows)

        print(f"\n{variant}")
        print(f"  Reponses evaluees : {n}/{total}")

        if not n:
            continue

        passed = sum(
            verdict.get("passed") is True
            for _, verdict in assessed
        )

        relevant = [
            verdict["relevance"]
            for _, verdict in assessed
            if isinstance(verdict.get("relevance"), bool)
        ]

        print(
            f"  Conformite metier : "
            f"{passed}/{n} ({passed / n:.1%})"
        )

        if relevant:
            relevant_count = sum(relevant)
            print(
                f"  Pertinence : {relevant_count}/{len(relevant)} "
                f"({relevant_count / len(relevant):.1%})"
            )


        # Qualite globale : conformite metier ET pertinence
        complete = [
            verdict
            for _, verdict in assessed
            if isinstance(verdict.get("passed"), bool)
            and isinstance(verdict.get("relevance"), bool)
        ]

        if complete:
            acceptable = sum(
                verdict["passed"] and verdict["relevance"]
                for verdict in complete
            )

            print(
                f"  Reponses acceptables : "
                f"{acceptable}/{len(complete)} "
                f"({acceptable / len(complete):.1%})"
            )

        if len(complete) < total:
            print("  ATTENTION : controles qualite incomplets")


        issues = {}

        for row, verdict in assessed:
            if verdict.get("relevance") is False:
                test_id = row["test_id"]
                issues[test_id] = issues.get(test_id, 0) + 1

        if issues:
            print("  Cas a examiner :")
            for test_id, count in sorted(issues.items()):
                print(f"    {test_id} : {count} signalement(s)")

        if n < total:
            print("  ATTENTION : evaluation incomplete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()

    if args.file:
        manifest_path = args.file
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
    else:
        files = sorted(
            RESULTS.glob("experiment_*.json")
        )
        if not files:
            raise FileNotFoundError(
                "Aucune experience trouvee"
            )
        manifest_path = files[-1]

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    tariffs = json.loads(
        (ROOT / "pricing.json").read_text(
            encoding="utf-8"
        )
    )["models"]

    mode = manifest["mode"]

    if len(manifest["runs"]) != manifest["repeats"]:
        raise ValueError("Repetitions manquantes")

    groups = defaultdict(list)
    repetitions = defaultdict(lambda: defaultdict(list))

    expected_variants = None
    expected_cases = None

    for run in manifest["runs"]:
        path = Path(run["file"])

        if not path.is_absolute():
            path = ROOT / path

        rows = load_jsonl(path)

        if not rows:
            raise ValueError(f"Fichier vide : {path}")

        current_variants = set()

        for row in rows:
            if row.get("error") or not row.get("answer"):
                raise ValueError(
                    f"Appel invalide : {row['test_id']}"
                )

            if (
                row.get("mode") != mode
                or row.get("dataset") != manifest["dataset"]
            ):
                raise ValueError("Configuration incoherente")

            variant = (
                row["prompt_version"]
                if mode == "prompts"
                else row["model"]
            )

            row["variant"] = variant
            row["cost_usd"] = calculate_cost(
                row, tariffs
            )

            current_variants.add(variant)
            groups[variant].append(row)
            repetitions[run["repetition"]][variant].append(row)

        if (
            expected_variants is not None
            and current_variants != expected_variants
        ):
            raise ValueError("Variantes incoherentes")

        expected_variants = current_variants

        for variant in current_variants:
            subset = repetitions[run["repetition"]][variant]
            ids = [r["test_id"] for r in subset]

            if (
                len(ids) != manifest["limit"]
                or len(set(ids)) != len(ids)
            ):
                raise ValueError(
                    "Tests manquants ou dupliques"
                )

            if expected_cases is None:
                expected_cases = set(ids)
            elif set(ids) != expected_cases:
                raise ValueError(
                    "Scenarios non comparables"
                )

    print(f"\nExperience : {manifest_path.name}")
    print(f"Mode : {mode}")
    print(f"Repetitions : {len(manifest['runs'])}")

    print("\n===== ANALYSE PAR VARIANTE =====")

    summaries = {}

    for variant in sorted(groups):
        rows = groups[variant]
        tokens = [r["total_tokens"] for r in rows]
        latencies = [r["latency_s"] for r in rows]
        costs = [r["cost_usd"] for r in rows]

        run_tokens = [
            sum(r["total_tokens"] for r in repetitions[n][variant])
            for n in sorted(repetitions)
        ]

        summary = {
            "tokens": sum(tokens),
            "cost": sum(costs),
            "latency": mean(latencies),
        }

        summaries[variant] = summary

        print(f"\n{variant}")
        print(f"  Appels : {len(rows)}")
        print(f"  Tokens cumules : {sum(tokens)}")
        print(f"  Tokens moyens/appel : {mean(tokens):.1f}")
        print(f"  Tokens medians/appel : {median(tokens):.1f}")
        print(
            f"  Tokens par repetition : {run_tokens}"
        )
        print(
            f"  Ecart-type tokens/repetition : "
            f"{stdev(run_tokens) if len(run_tokens) > 1 else 0:.1f}"
        )
        print(
            f"  Latence moyenne : {mean(latencies):.3f}s"
        )
        print(
            f"  Latence mediane : {median(latencies):.3f}s"
        )
        print(f"  Cout total estime : ${sum(costs):.8f}")
        print(
            f"  Cout moyen/appel : ${mean(costs):.8f}"
        )

    if len(summaries) == 2:
        names = sorted(summaries)
        a, b = summaries[names[0]], summaries[names[1]]

        def change(before, after):
            return (
                (after / before - 1) * 100
                if before else 0
            )

        print(
            f"\n===== EVOLUTION "
            f"{names[1]} VS {names[0]} ====="
        )
        print(
            f"Tokens : {change(a['tokens'], b['tokens']):+.1f}%"
        )
        print(
            f"Cout : {change(a['cost'], b['cost']):+.1f}%"
        )
        print(
            f"Latence : {change(a['latency'], b['latency']):+.1f}%"
        )

    report_judge_quality(groups)


if __name__ == "__main__":
    main()
