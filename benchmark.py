
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from prompt_store import list_versions

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langfuse import get_client, observe

load_dotenv()

MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
]

PROMPTS = {
    "v1": Path("prompts/classification_v1.txt"),
    "v2": Path("prompts/classification_v2.txt"),
}

DATASET = Path("datasets/classification.jsonl")


DATASETS = {
    "classification": {
        "file": DATASET,
        "prompts": list_versions("classification"),
    },
    "agent_assurance": {
        "file": Path("datasets/agent_assurance.jsonl"),
        "prompts": list_versions("agent_assurance"),
    },
}

CONFIG = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
        thinking_level="low"
    )
)



def generate_traced(client, model, prompt, record):
    langfuse = get_client()

    with langfuse.start_as_current_observation(
        as_type="generation",
        name="gemini-benchmark",
        model=model,
        input=prompt,
        metadata={
            "run_id": record["run_id"],
            "mode": record["mode"],
            "dataset": record["dataset"],
            "test_id": record["test_id"],
            "prompt_version": record["prompt_version"],
            "prompt_hash": record["prompt_hash"],
        },
    ) as generation:

        # Identifiants pour rattacher les futures evaluations
        record["langfuse_trace_id"] = generation.trace_id
        record["langfuse_observation_id"] = generation.id


        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=CONFIG,
        )

        usage = response.usage_metadata
        usage_details = {}

        if usage:
            if usage.prompt_token_count is not None:
                usage_details["input"] = usage.prompt_token_count

            if usage.candidates_token_count is not None:
                usage_details["output"] = usage.candidates_token_count

            if usage.thoughts_token_count is not None:
                usage_details["output_reasoning"] = usage.thoughts_token_count

            if usage.total_token_count is not None:
                usage_details["total"] = usage.total_token_count

        answer = (response.text or "").strip()

        generation.update(
            model=response.model_version or model,
            output=answer,
            usage_details=usage_details or None,
        )

        # Exact match uniquement pour les classifications
        if record["dataset"] == "classification":
            expected = record["expected"].strip()

            generation.score(
                name="exact_match",
                value=int(answer.upper() == expected.upper()),
                data_type="BOOLEAN",
                comment="Comparaison exacte avec la categorie attendue",
            )

        # Aucun score pour les réponses ouvertes à ce stade
        return response


@observe(
    name="benchmark-campaign",
    capture_input=False,
    capture_output=False,
)

def _run_benchmark(
    mode: str,
    model: str = MODELS[0],
    prompt: str = "v2",
    limit: int | None = None,
    dataset: str = "classification",
    versions: list[str] | None = None,
) -> Path:

    if mode not in ("prompts", "models"):
        raise ValueError("Mode invalide")

    if model not in MODELS:
        raise ValueError("Modele inconnu")

    if dataset not in DATASETS:
        raise ValueError(f"Dataset inconnu : {dataset}")

    dataset_config = DATASETS[dataset]
    prompt_files = list_versions(dataset)

    if mode == "prompts":
        chosen_versions = (
            versions if versions is not None
            else list(prompt_files)
        )

        if (
            len(chosen_versions) < 2
            or len(chosen_versions) != len(set(chosen_versions))
            or any(v not in prompt_files for v in chosen_versions)
        ):
            raise ValueError("Sélection de prompts invalide")
    else:
        if versions is not None:
            raise ValueError("Versions incompatibles avec le mode models")

    if prompt not in prompt_files:
        raise ValueError("Version de prompt inconnue")

    if limit is not None and limit < 1:
        raise ValueError("--limit doit etre positif")

    args = argparse.Namespace(
        mode=mode,
        model=model,
        prompt=prompt,
        limit=limit,
        dataset=dataset,
    )

    # Charger les donnees
    with dataset_config["file"].open(encoding="utf-8-sig") as f:
        tests = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if args.limit:
        tests = tests[:args.limit]

    if not tests:
        raise ValueError("Dataset vide")

    templates = {
        name: path.read_text(encoding="utf-8-sig")
        for name, path in prompt_files.items()
    }

    # Construire les configurations
    if args.mode == "prompts":
        variants = [
            (args.model, version)
            for version in chosen_versions
        ]
    else:
        variants = [
            (model, args.prompt)
            for model in MODELS
        ]

    for _, version in variants:
        if "{input}" not in templates[version]:
            raise ValueError(
                f"Placeholder manquant : {version}"
            )
        if dataset == "agent_assurance" and "{context}" not in templates[version]:
            raise ValueError(f"Placeholder context absent : {version}")


    required_fields = (
        {"id", "input", "expected"}
        if dataset == "classification"
        else {"id", "input", "context", "reference", "criteria"}
    )

    for test in tests:
        missing = required_fields - test.keys()
        if missing:
            raise ValueError(
                f"Champs absents pour {test.get('id')}: {missing}"
            )

    if len({test["id"] for test in tests}) != len(tests):
        raise ValueError("Identifiants de tests dupliques")


    # Preparer l'experience
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    get_client().update_current_span(
        metadata={
            "run_id": run_id,
            "benchmark_mode": mode,
            "test_cases": len(tests),
            "variants": len(variants),
        }
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        f"benchmark_{run_id}.jsonl"
    )

    print(f"\nMODE : {args.mode}")
    print(f"TESTS : {len(tests)}")
    print(f"VARIANTES : {len(variants)}")
    print(f"APPELS PREVUS : {len(tests) * len(variants)}")

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    records = []

    try:
        with output.open("w", encoding="utf-8") as f:
            for index, test in enumerate(tests):
                ordered = variants.copy()

                # Equilibrer l'ordre des appels
                if index % 2:
                    ordered.reverse()

                for model, version in ordered:
                    template = templates[version]

                    prompt = (
                        template
                        .replace("{context}", test.get("context", ""))
                        .replace("{input}", test["input"])
                    )

                    record = {
                        "run_id": run_id,
                        "mode": args.mode,
                        "test_id": test["id"],
                        "model": model,
                        "model_version": None,
                        "dataset": dataset,
                        "context": test.get("context"),
                        "reference": test.get("reference"),
                        "criteria": test.get("criteria", []),
                        "prompt_version": version,
                        "prompt_hash": hashlib.sha256(
                            template.encode("utf-8")
                        ).hexdigest()[:12],
                        "input": test["input"],
                        "expected": test.get("expected"),
                        "answer": None,
                        "correct": None,
                        "latency_s": None,
                        "input_tokens": None,
                        "output_tokens": None,
                        "thinking_tokens": None,
                        "total_tokens": None,
                        "error": None,
                    }

                    start = time.perf_counter()

                    try:
                        response = generate_traced(
                            client=client,
                            model=model,
                            prompt=prompt,
                            record=record
                        )

                        usage = response.usage_metadata
                        answer = (response.text or "").strip()

                        record["model_version"] = response.model_version
                        record["answer"] = answer

                        if dataset == "classification":
                            record["correct"] = (
                                answer.upper() == test["expected"].upper()
                            )

                        if usage:
                            record["input_tokens"] = (
                                usage.prompt_token_count
                            )
                            record["output_tokens"] = (
                                usage.candidates_token_count
                            )
                            record["thinking_tokens"] = (
                                usage.thoughts_token_count
                            )
                            record["total_tokens"] = (
                                usage.total_token_count
                            )

                        if not answer:
                            record["error"] = "Reponse vide"

                    except Exception as error:
                        record["error"] = (
                            f"{type(error).__name__}: {error}"
                        )

                    record["latency_s"] = round(
                        time.perf_counter() - start, 3
                    )

                    records.append(record)

                    f.write(
                        json.dumps(
                            record, ensure_ascii=False
                        ) + "\n"
                    )
                    f.flush()

                    print(
                        f"{test['id']} | {model} | "
                        f"{version} | "
                        f"OK: {record['correct']} | "
                        f"Tokens: {record['total_tokens']} | "
                        f"{record['latency_s']}s"
                    )

    finally:
        client.close()

    # Synthese comparative
    print("\n===== SYNTHESE =====")

    for model, version in variants:
        rows = [
            r for r in records
            if r["model"] == model
            and r["prompt_version"] == version
        ]

        successful = [
            r for r in rows if r["error"] is None
        ]

        correct = sum(r["correct"] is True for r in rows)

        tokens = sum(
            r["total_tokens"] or 0
            for r in successful
        )

        latency = (
            mean(r["latency_s"] for r in successful)
            if successful else 0
        )

        print(f"\n{model} | Prompt {version}")
        if dataset == "classification":
            print(f"  Exactitude : {correct}/{len(rows)}")
        else:
            print("  Qualite : non evaluee (LLM-as-a-Judge a venir)")
        print(f"  Erreurs : {len(rows) - len(successful)}")
        print(f"  Tokens total : {tokens}")
        print(f"  Latence moyenne : {latency:.3f}s")

    print(f"\nResultats : {output}")


    # Scoring global uniquement pour la classification
    if dataset == "classification":
        accuracy = (
            sum(r["correct"] is True for r in records)
            / len(records)
        )

        get_client().score_current_trace(
            name="campaign_accuracy",
            value=round(accuracy, 4),
            data_type="NUMERIC",
            comment="Proportion de reponses exactes dans la campagne",
        )

    return output


def run_benchmark(
    mode: str,
    model: str = MODELS[0],
    prompt: str = "v2",
    limit: int | None = None,
    dataset: str = "classification",
    versions: list[str] | None = None,
) -> Path:
    try:
        return _run_benchmark(
            mode=mode,
            model=model,
            prompt=prompt,
            limit=limit,
            dataset=dataset,
            versions=versions,
        )
    finally:
        get_client().flush()


def main():
    parser = argparse.ArgumentParser(
        description="LLM & Prompt Benchmark"
    )

    parser.add_argument(
        "--mode",
        choices=["prompts", "models"],
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=MODELS,
        default=MODELS[0],
    )

    parser.add_argument(
        "--prompt",
        default="v2",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dataset",
        choices=list(DATASETS),
        default="classification",
    )

    args = parser.parse_args()

    try:
        run_benchmark(
            mode=args.mode,
            model=args.model,
            prompt=args.prompt,
            limit=args.limit,
            dataset=args.dataset,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
