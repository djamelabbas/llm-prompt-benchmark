
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmark import DATASETS, MODELS, run_benchmark

MAX_API_CALLS = 30


def main():
    parser = argparse.ArgumentParser(
        description="Repeated LLM Benchmark"
    )

    parser.add_argument(
        "--mode",
        choices=["prompts", "models"],
        default="prompts",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS),
        default="agent_assurance",
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        default=MODELS[0],
    )
    parser.add_argument("--prompt", default="v2")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)

    # Protection contre les appels involontaires
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Autoriser les appels API facturables",
    )

    args = parser.parse_args()

    dataset_config = DATASETS[args.dataset]

    if args.prompt not in dataset_config["prompts"]:
        parser.error("Version de prompt inconnue")

    available = sum(
        bool(line.strip())
        for line in dataset_config["file"].read_text(
            encoding="utf-8-sig"
        ).splitlines()
    )

    if not 1 <= args.limit <= available:
        parser.error("Nombre de cas invalide")

    if not 1 <= args.repeats <= 10:
        parser.error("Nombre de repetitions invalide")

    variants = (
        len(dataset_config["prompts"])
        if args.mode == "prompts"
        else len(MODELS)
    )

    total_calls = (
        args.limit * variants * args.repeats
    )

    if total_calls > MAX_API_CALLS:
        parser.error(
            f"Maximum {MAX_API_CALLS} appels API"
        )

    print("\n===== PLAN EXPERIMENTAL =====")
    print(f"Dataset : {args.dataset}")
    print(f"Mode : {args.mode}")
    print(f"Cas par repetition : {args.limit}")
    print(f"Variantes : {variants}")
    print(f"Repetitions : {args.repeats}")
    print(f"Appels API prevus : {total_calls}")

    if not args.execute:
        print("\nSIMULATION : aucun appel API.")
        return

    experiment_id = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    manifest = {
        "experiment_id": experiment_id,
        "dataset": args.dataset,
        "mode": args.mode,
        "model": args.model,
        "prompt": args.prompt,
        "limit": args.limit,
        "repeats": args.repeats,
        "runs": [],
    }

    output = (
        Path("results")
        / f"experiment_{experiment_id}.json"
    )

    for iteration in range(1, args.repeats + 1):
        print(
            f"\n===== REPETITION "
            f"{iteration}/{args.repeats} ====="
        )

        result_path = run_benchmark(
            mode=args.mode,
            dataset=args.dataset,
            model=args.model,
            prompt=args.prompt,
            limit=args.limit,
        )

        manifest["runs"].append({
            "repetition": iteration,
            "file": str(result_path),
        })

        # Sauvegarder apres chaque repetition
        output.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"\nExperience terminee : {output}")


if __name__ == "__main__":
    main()
