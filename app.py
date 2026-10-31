
import os
from benchmark import MODELS, DATASETS, run_benchmark
import json
from datetime import date, datetime
from pathlib import Path
from prompt_store import list_versions, save_new_version
from uuid import uuid4
from judge import run_judge

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="LLM & Prompt Optimization Lab",
    page_icon="🧪",
    layout="wide",
)

st.title("LLM & Prompt Optimization Lab")
st.caption(
    "Benchmark de modèles et optimisation de prompts "
    "— qualité, tokens, latence et coûts."
)


# --- Editeur de prompts ---

with st.expander("Editeur de prompts - Nouvelle version"):

    editor_dataset = st.selectbox(
        "Dataset",
        options=list(DATASETS),
        key="editor_dataset",
    )

    versions = list_versions(editor_dataset)

    source_version = st.selectbox(
        "Version source",
        options=list(versions),
        index=len(versions) - 1,
        key="editor_source_version",
    )

    source_content = versions[source_version].read_text(
        encoding="utf-8-sig"
    )

    edited_content = st.text_area(
        "Contenu du prompt",
        value=source_content,
        height=300,
        key=f"editor_{editor_dataset}_{source_version}",
    )

    if st.button(
        "Enregistrer comme nouvelle version",
        type="primary",
    ):
        try:
            if edited_content.strip() == source_content.strip():
                st.warning(
                    "Modifie le prompt avant de creer une version."
                )
            else:
                new_version, path = save_new_version(
                    editor_dataset,
                    edited_content,
                )

                st.session_state["saved_prompt"] = (
                    f"Version {new_version} creee : {path.name}"
                )

                st.rerun()

        except (ValueError, OSError) as error:
            st.error(str(error))

    if "saved_prompt" in st.session_state:
        st.success(st.session_state.pop("saved_prompt"))


def load_jsonl(path):
    with path.open(encoding="utf-8-sig") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def estimate_cost(row, pricing, run_date):
    if row.get("error"):
        return None

    model = row.get("model_version") or row["model"]
    tariff = pricing.get(model)

    if not tariff:
        return None

    expires = tariff.get("valid_until")
    if expires and run_date > date.fromisoformat(expires):
        return None

    inputs = row.get("input_tokens")
    outputs = row.get("output_tokens")
    thinking = row.get("thinking_tokens") or 0
    total = row.get("total_tokens")

    if any(value is None for value in [inputs, outputs, total]):
        return None

    if inputs + outputs + thinking != total:
        return None

    return (
        inputs * tariff["input"]
        + (outputs + thinking) * tariff["output"]
    ) / 1_000_000


# --- Lancement d'une nouvelle experience ---

MAX_API_CALLS = 12

with st.sidebar:
    st.subheader("Nouvelle experience")

    mode = st.radio(
        "Type de benchmark",
        ["prompts", "models"],
        format_func=lambda value: {
            "prompts": "Optimisation de prompts",
            "models": "Comparaison de modeles",
        }[value],
    )

    dataset = st.selectbox(
        "Jeu de tests",
        options=list(DATASETS),
        format_func=lambda value: {
            "classification": "Classification assurance",
            "agent_assurance": "Agent AssurNova",
        }[value],
    )

    prompt_files = list_versions(dataset)

    selected_versions = None

    model = MODELS[0]
    prompt = "v2"

    if mode == "prompts":
        model = st.selectbox(
            "Modele a utiliser",
            MODELS,
        )

        selected_versions = st.multiselect(
            "Versions à comparer",
            options=list(prompt_files),
            default=list(prompt_files)[:2],
            key=f"versions_{dataset}",
        )

        variants_count = len(selected_versions)

        if variants_count < 2:
            st.warning(
                "Sélectionne au moins deux versions."
            )

        st.caption(
            f"{variants_count} versions de prompt comparees."
        )
    else:
        prompt = st.selectbox(
            "Version du prompt",
            list(prompt_files),
            index=list(prompt_files).index("v2"),
        )
        variants_count = len(MODELS)
        st.caption(
            f"{variants_count} modeles compares."
        )

    # Limite des appels API facturables
    available_cases = sum(
        1
        for line in DATASETS[dataset]["file"].read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    )

    max_cases = min(
        available_cases,
        MAX_API_CALLS // max(variants_count, 1),
    )

    if max_cases < 1:
        st.error("Aucun test disponible.")
        st.stop()

    limit = st.number_input(
        "Nombre de cas a tester",
        min_value=1,
        max_value=max_cases,
        value=1,
        step=1,
    )

    api_calls = int(limit) * variants_count

    st.info(
        f"{api_calls} appels API Gemini prevus "
        f"(maximum autorise : {MAX_API_CALLS})."
    )

    confirmed = st.checkbox(
        "Je confirme les appels API facturables"
    )

    launch = st.button(
        "Lancer le benchmark",
        type="primary",
        use_container_width=True,
        disabled=(
            not confirmed
            or (mode == "prompts" and variants_count < 2)
            ),
        )

if launch:
    if not os.getenv("GEMINI_API_KEY"):
        st.sidebar.error(
            "Cle GEMINI_API_KEY introuvable."
        )
    else:
        try:
            with st.spinner(
                f"Benchmark en cours : {api_calls} appels API..."
            ):
                output = run_benchmark(
                    mode=mode,
                    model=model,
                    prompt=prompt,
                    limit=int(limit),
                    dataset=dataset,
                    versions=selected_versions,
                )

            # Selectionner automatiquement le nouveau resultat
            st.session_state["selected_run"] = output

            st.rerun()

        except Exception as error:
            st.sidebar.error(
                f"Erreur : {type(error).__name__}: {error}"
            )

# --- Selection de l'experience ---

files = sorted(
    Path("results").glob("benchmark_*.jsonl"),
    reverse=True,
)

if not files:
    st.info("Aucune experience disponible.")
    st.stop()


# Initialiser ou restaurer l'experience selectionnee
if (
    "selected_run" not in st.session_state
    or st.session_state["selected_run"] not in files
):
    st.session_state["selected_run"] = files[0]

selected = st.sidebar.selectbox(
    "Historique des experiences",
    files,
    format_func=lambda path: path.name,
    key="selected_run",
)

records = load_jsonl(selected)

if not records:
    st.error("Le fichier selectionne est vide.")
    st.stop()

# Compatibilite avec notre premier benchmark historique
mode = records[0].get("mode")
if not mode:
    mode = (
        "models"
        if len({r["model"] for r in records}) > 1
        else "prompts"
    )

run_date = datetime.strptime(
    records[0]["run_id"][:8], "%Y%m%d"
).date()

pricing = json.loads(
    Path("pricing.json").read_text(encoding="utf-8-sig")
)["models"]

for row in records:
    row["variant"] = (
        row["prompt_version"]
        if mode == "prompts"
        else row["model"]
    )
    row["cost_usd"] = estimate_cost(
        row, pricing, run_date
    )

df = pd.DataFrame(records)

if df.duplicated(["test_id", "variant"]).any():
    st.error("Doublons detectes dans cette experience.")
    st.stop()

groups = [
    set(group["test_id"])
    for _, group in df.groupby("variant")
]

if any(group != groups[0] for group in groups[1:]):
    st.warning(
        "Les variantes ne partagent pas les memes cas de test."
    )

st.sidebar.markdown("---")
st.sidebar.write(
    "**Mode :**",
    "Optimisation de prompts"
    if mode == "prompts"
    else "Comparaison de modeles",
)
st.sidebar.write("**Date :**", str(run_date))
st.sidebar.write("**Fichier :**", selected.name)

# --- KPI generaux ---

c1, c2, c3, c4 = st.columns(4)

c1.metric("Cas distincts", df["test_id"].nunique())
c2.metric("Variantes", df["variant"].nunique())
c3.metric("Appels enregistres", len(df))
c4.metric("Erreurs API", df["error"].notna().sum())

# --- Agregation par variante ---

stats = []

for variant, group in df.groupby("variant"):
    complete_tokens = group["total_tokens"].notna().all()
    complete_cost = group["cost_usd"].notna().all()

    tokens = (
        int(group["total_tokens"].sum())
        if complete_tokens else None
    )

    cost = (
        float(group["cost_usd"].sum())
        if complete_cost else None
    )

    accuracy = (
        float(group["correct"].astype(bool).mean())
        if group["correct"].notna().all()
        else None
    )

    stats.append({
        "variant": variant,
        "tests": len(group),
        "accuracy": accuracy,
        "tokens": tokens,
        "latency": group["latency_s"].mean(),
        "cost": cost,
        "cost_per_1000": (
            cost * 1000 / len(group)
            if cost is not None else None
        ),
    })

summary = pd.DataFrame(stats)

st.subheader("Comparaison des variantes")

display = summary[
    ["variant", "tests", "accuracy", "tokens", "latency", "cost"]
].copy()

display["accuracy"] = display["accuracy"].map(
    lambda value: (
        f"{value:.0%}"
        if pd.notna(value)
        else "Non évaluée"
    )
)
display["latency"] = display["latency"].map(
    lambda value: f"{value:.3f} s"
)
display["cost"] = display["cost"].map(
    lambda value: (
        f"${value:.8f}" if pd.notna(value) else "N/D"
    )
)

display.columns = [
    "Variante", "Tests", "Exactitude",
    "Tokens totaux", "Latence moyenne", "Cout estime",
]

st.dataframe(
    display,
    hide_index=True,
    use_container_width=True,
)

# --- Visualisations ---

left, right = st.columns(2)

with left:
    fig = px.bar(
        summary,
        x="variant",
        y="tokens",
        text="tokens",
        title="Consommation totale de tokens",
        labels={
            "variant": "Variante",
            "tokens": "Tokens",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.bar(
        summary,
        x="variant",
        y="cost_per_1000",
        text_auto=".4f",
        title="Cout estime pour 1 000 requetes similaires (USD)",
        labels={
            "variant": "Variante",
            "cost_per_1000": "USD",
        },
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Comparaison dynamique des prompts ---

if mode == "prompts":
    st.subheader("Bilan de l'optimisation")

    variants = summary["variant"].tolist()

    baseline = st.selectbox(
        "Version de reference (baseline)",
        options=variants,
        index=variants.index("v1") if "v1" in variants else 0,
        key=f"baseline_{selected.stem}",
    )

    reference = summary.set_index("variant").loc[baseline]


    # Index des evaluations LLM-as-a-Judge
    judge_index = {}

    for file in sorted(Path("results").glob("judge_*.jsonl")):
        for item in load_jsonl(file):
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
                judge_index[key] = verdict

    def get_quality(version):
        group = df[df["variant"] == version]

        if group.empty:
            return None

        # Classification : exact match
        if group["correct"].notna().all():
            passed = int(group["correct"].astype(bool).sum())
            return {
                "rate": passed / len(group),
                "label": f"{passed}/{len(group)} (Exact match)",
            }

        # Agent : conformite metier ET pertinence
        results = []

        for _, record in group.iterrows():
            key = (
                record["run_id"],
                record["test_id"],
                record["model"],
                record["prompt_version"],
            )

            verdict = judge_index.get(key)

            if (
                verdict is None
                or not isinstance(verdict.get("passed"), bool)
                or not isinstance(verdict.get("relevance"), bool)
            ):
                return None

            results.append(
                verdict["passed"] and verdict["relevance"]
            )

        passed = sum(results)

        return {
            "rate": passed / len(results),
            "label": f"{passed}/{len(results)} (LLM Judge)",
        }

    baseline_quality = get_quality(baseline)

    def relative_change(current, base):
        if pd.isna(current) or pd.isna(base) or base == 0:
            return "N/D"

        change = (current / base - 1) * 100
        return f"{change:+.1f} %"


    comparison_rows = []

    for _, row in summary.iterrows():
        version = row["variant"]
        candidate_quality = get_quality(version)

        if version == baseline:
            quality_status = "Reference"

        elif baseline_quality is None or candidate_quality is None:
            quality_status = "Non evaluee"

        elif candidate_quality["rate"] >= baseline_quality["rate"]:
            quality_status = "PASS (provisoire)"

        else:
            quality_status = "FAIL (provisoire)"

        if version == baseline:
            decision = "Baseline"
        elif quality_status == "Non evaluee":
            decision = "Evaluation requise"
        elif quality_status.startswith("FAIL"):
            decision = "Regression qualite"
        elif pd.isna(row["cost"]) or pd.isna(reference["cost"]):
            decision = "Cout indisponible"
        elif row["cost"] >= reference["cost"]:
            decision = "Surcout observe"
        else:
            decision = "Candidate a confirmer"

        comparison_rows.append({
            "Version": version,
            "Role": (
                "Baseline" if version == baseline else "Candidate"
            ),
            "Variation tokens": relative_change(
                row["tokens"], reference["tokens"]
            ),
            "Variation cout": relative_change(
                row["cost"], reference["cost"]
            ),
            "Variation latence": relative_change(
                row["latency"], reference["latency"]
            ),
            "Qualite": (
                candidate_quality["label"]
                if candidate_quality else "Non evaluee"
            ),
            "Non-regression": quality_status,
            "Decision indicative": decision,
        })

    st.dataframe(
        pd.DataFrame(comparison_rows),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Variations relatives a la baseline. "
        "Une valeur negative indique une reduction "
        "des tokens, du cout ou de la latence. "
        "Le controle exact match ne couvre pas "
        "les evaluations LLM-as-a-Judge."
    )


# --- Comparaison des reponses ---

st.subheader("Reponses cote a cote")

comparison = df.pivot(
    index="test_id",
    columns="variant",
    values="answer",
).reset_index()

comparison.columns.name = None

# Afficher la categorie attendue pour les classifications
if "expected" in df.columns and df["expected"].notna().any():
    expected_df = df[
        ["test_id", "expected"]
    ].drop_duplicates(subset=["test_id"])

    comparison = expected_df.merge(
        comparison,
        on="test_id",
        how="right",
    )

st.dataframe(
    comparison,
    hide_index=True,
    use_container_width=True,
)

# --- References et criteres du dataset ---

if (
    "reference" in df.columns
    and df["reference"].notna().any()
):
    st.subheader("References d'evaluation")

    reference_df = df[
        ["test_id", "input", "reference", "criteria"]
    ].drop_duplicates(subset=["test_id"])

    st.dataframe(
        reference_df,
        hide_index=True,
        use_container_width=True,
    )


# --- Lancement du LLM-as-a-Judge ---

if any(r.get("dataset") == "agent_assurance" for r in records):

    st.subheader("Lancer l'evaluation LLM-as-a-Judge")

    # Identifier les reponses deja evaluees
    evaluated_keys = set()

    for judge_file in sorted(
        Path("results").glob("judge_*.jsonl")
    ):
        for item in load_jsonl(judge_file):
            if (
                item.get("error") is None
                and isinstance(item.get("evaluation"), dict)
            ):
                evaluated_keys.add((
                    item.get("source_run_id"),
                    item.get("test_id"),
                    item.get("model"),
                    item.get("prompt_version"),
                ))

    # Conserver uniquement les reponses non evaluees
    pending = [
        row for row in records
        if (
            row["run_id"],
            row["test_id"],
            row["model"],
            row["prompt_version"],
        ) not in evaluated_keys
    ]

    completed = len(records) - len(pending)

    st.caption(
        f"{completed}/{len(records)} reponses deja evaluees."
    )

    if "judge_feedback" in st.session_state:
        st.success(st.session_state.pop("judge_feedback"))

    if not pending:
        st.success("Toutes les reponses sont evaluees.")

    elif any(
        row.get("error")
        or not row.get("answer")
        or not row.get("langfuse_trace_id")
        or not row.get("langfuse_observation_id")
        for row in pending
    ):
        st.error(
            "Certaines reponses sont invalides ou ne "
            "possedent pas leurs identifiants Langfuse."
        )

    else:
        judge_limit = st.number_input(
            "Nombre de reponses a evaluer",
            min_value=1,
            max_value=min(10, len(pending)),
            value=min(2, len(pending)),
            step=1,
            key=f"judge_limit_{selected.stem}",
        )

        st.info(
            f"{judge_limit} appels facturables "
            "a Gemini 3.6 Flash."
        )

        judge_confirmed = st.checkbox(
            "Je confirme les appels du modele juge",
            key=f"judge_confirm_{selected.stem}",
        )

        if st.button(
            "Evaluer les reponses",
            type="primary",
            disabled=not judge_confirmed,
            key=f"judge_button_{selected.stem}",
        ):

            # Fichier temporaire contenant les reponses
            # qui ne sont pas encore evaluees
            temp_file = Path("results") / (
                f".judge_pending_{uuid4().hex}.jsonl"
            )

            try:
                chosen = pending[:int(judge_limit)]

                temp_file.write_text(
                    "\n".join(
                        json.dumps(
                            row, ensure_ascii=False
                        )
                        for row in chosen
                    ) + "\n",
                    encoding="utf-8",
                )

                with st.spinner(
                    "Evaluation des reponses en cours..."
                ):
                    output = run_judge(
                        source_file=temp_file,
                        limit=int(judge_limit),
                    )

                results = load_jsonl(output)

                success_count = sum(
                    result.get("error") is None
                    and isinstance(
                        result.get("evaluation"), dict
                    )
                    for result in results
                )

                st.session_state["judge_feedback"] = (
                    f"{success_count}/{len(results)} "
                    "evaluations reussies."
                )

                st.rerun()

            except Exception as error:
                st.error(
                    f"Erreur d'evaluation : "
                    f"{type(error).__name__}: {error}"
                )

            finally:
                temp_file.unlink(missing_ok=True)


# --- Evaluations LLM-as-a-Judge ---

if any(r.get("dataset") == "agent_assurance" for r in records):
    st.subheader("Evaluations LLM-as-a-Judge")

    # Recuperer la derniere evaluation de chaque reponse
    evaluations = {}

    for file in sorted(Path("results").glob("judge_*.jsonl")):
        for item in load_jsonl(file):
            verdict = item.get("evaluation")

            if item.get("error") or not isinstance(verdict, dict):
                continue

            key = (
                item.get("source_run_id"),
                item.get("test_id"),
                item.get("model"),
                item.get("prompt_version"),
            )

            evaluations[key] = verdict

    def score_label(verdict, metric):
        if not verdict or verdict.get(metric) is None:
            return "Non evalue"
        return "PASS" if verdict[metric] else "FAIL"

    evaluation_rows = []

    for record in records:
        key = (
            record["run_id"],
            record["test_id"],
            record["model"],
            record["prompt_version"],
        )

        verdict = evaluations.get(key)

        criteria = (
            verdict.get("criteria_met", [])
            if verdict else []
        )

        evaluation_rows.append({
            "Cas": record["test_id"],
            "Variante": record["variant"],
            "Exactitude": score_label(
                verdict, "factual_accuracy"
            ),
            "Ancrage documentaire": score_label(
                verdict, "grounded"
            ),
            "Pertinence": score_label(
                verdict, "relevance"
            ),
            "Criteres respectes": (
                f"{sum(criteria)}/{len(criteria)}"
                if verdict else "Non evalue"
            ),
            "Verdict global": score_label(
                verdict, "passed"
            ),
            "Justification": (
                verdict.get("reason", "")
                if verdict else ""
            ),
        })

    evaluation_df = pd.DataFrame(evaluation_rows)

    st.dataframe(
        evaluation_df,
        hide_index=True,
        use_container_width=True,
    )

    evaluated = (
        evaluation_df["Verdict global"] != "Non evalue"
    ).sum()

    st.caption(
        f"{evaluated}/{len(evaluation_df)} reponses evaluees. "
        "Scores issus des fichiers JSONL locaux. "
        "Le jugement automatique reste indicatif."
    )


with st.expander("Metriques detaillees par appel"):
    columns = [
        "test_id", "variant", "answer", "correct",
        "input_tokens", "output_tokens",
        "thinking_tokens", "total_tokens",
        "latency_s", "cost_usd", "error",
    ]

    st.dataframe(
        df[columns],
        hide_index=True,
        use_container_width=True,
    )

st.download_button(
    "Exporter les resultats en CSV",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"{selected.stem}.csv",
    mime="text/csv",
)

if df["cost_usd"].isna().any():
    st.warning(
        "Certains couts n'ont pas pu etre calcules : "
        "verifier pricing.json et les metriques de tokens."
    )


# --- Analyse des experiences repetees ---

st.subheader("Analyse multi-repetitions")

manifests = sorted(
    Path("results").glob("experiment_*.json"),
    reverse=True,
)

if not manifests:
    st.info("Aucune experience repetee disponible.")

else:
    selected_manifest = st.selectbox(
        "Experience repetee",
        manifests,
        format_func=lambda path: path.name,
    )

    manifest = json.loads(
        selected_manifest.read_text(encoding="utf-8")
    )


    # Charger les evaluations des campagnes repetees
    judge_index = {}

    for judge_file in sorted(
        Path("results").glob("judge_*.jsonl")
    ):
        for item in load_jsonl(judge_file):
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
                judge_index[key] = verdict


    repeated_rows = []

    for run in manifest["runs"]:
        path = Path(run["file"])

        if not path.is_absolute():
            path = Path.cwd() / path

        for record in load_jsonl(path):
            run_date = datetime.strptime(
                record["run_id"][:8], "%Y%m%d"
            ).date()

            variant = (
                record["prompt_version"]
                if manifest["mode"] == "prompts"
                else record["model"]
            )

            judge_key = (
                record["run_id"],
                record["test_id"],
                record["model"],
                record["prompt_version"],
            )

            verdict = judge_index.get(judge_key)

            repeated_rows.append({
                "repetition": run["repetition"],
                "variant": variant,
                "test_id": record["test_id"],
                "input_tokens": record.get("input_tokens"),
                "output_tokens": record.get("output_tokens"),
                "thinking_tokens": (
                    record.get("thinking_tokens") or 0
                ),
                "total_tokens": record.get("total_tokens"),
                "latency_s": record.get("latency_s"),
                "cost_usd": estimate_cost(
                    record, pricing, run_date
                ),
                "judge_passed": (
                    int(verdict["passed"])
                    if verdict is not None
                    and isinstance(verdict.get("passed"), bool)
                    else None
                ),
                "judge_relevance": (
                    int(verdict["relevance"])
                    if verdict is not None
                    and isinstance(verdict.get("relevance"), bool)
                    else None
                ),
                "error": record.get("error"),
            })

    repeated_df = pd.DataFrame(repeated_rows)

    required = [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency_s",
        "cost_usd",
    ]

    if repeated_df.empty:
        st.warning("Aucune donnee trouvee.")

    elif (
        repeated_df["error"].notna().any()
        or repeated_df[required].isna().any().any()
    ):
        st.error(
            "Donnees incompletes : analyse impossible."
        )

    else:
        summary = repeated_df.groupby(
            "variant", as_index=False
        ).agg(
            appels=("test_id", "count"),
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            thinking_tokens=("thinking_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            latency_mean=("latency_s", "mean"),
            latency_median=("latency_s", "median"),
            cost_usd=("cost_usd", "sum"),
        )

        st.markdown("#### Synthese consolidee")

        display_summary = summary.copy()
        display_summary["latency_mean"] = (
            display_summary["latency_mean"].round(3)
        )
        display_summary["latency_median"] = (
            display_summary["latency_median"].round(3)
        )
        display_summary["cost_usd"] = (
            display_summary["cost_usd"].map(
                lambda value: f"${value:.8f}"
            )
        )

        st.dataframe(
            display_summary,
            hide_index=True,
            use_container_width=True,
        )


        # --- Qualite consolidee par variante ---

        st.markdown("#### Qualite LLM-as-a-Judge")

        quality = repeated_df.groupby(
            "variant", as_index=False
        ).agg(
            responses=("test_id", "count"),
            judged=("judge_passed", "count"),
            passed=("judge_passed", "sum"),
            relevance_judged=("judge_relevance", "count"),
            relevant=("judge_relevance", "sum"),
        )

        def format_score(success, assessed, total):
            if assessed == 0:
                return "Non evaluee"

            rate = success / assessed
            label = f"{int(success)}/{int(assessed)} ({rate:.1%})"

            if assessed < total:
                label += " - Evaluation partielle"

            return label

        quality["Conformite metier"] = quality.apply(
            lambda row: format_score(
                row["passed"],
                row["judged"],
                row["responses"],
            ),
            axis=1,
        )

        quality["Pertinence"] = quality.apply(
            lambda row: format_score(
                row["relevant"],
                row["relevance_judged"],
                row["responses"],
            ),
            axis=1,
        )

        quality_display = quality[
            ["variant", "responses", "Conformite metier", "Pertinence"]
        ].rename(
            columns={
                "variant": "Prompt",
                "responses": "Reponses",
            }
        )

        st.dataframe(
            quality_display,
            hide_index=True,
            use_container_width=True,
        )

        # Repartition de la consommation
        breakdown = summary.melt(
            id_vars="variant",
            value_vars=[
                "input_tokens",
                "output_tokens",
                "thinking_tokens",
            ],
            var_name="type",
            value_name="tokens",
        )

        fig_tokens = px.bar(
            breakdown,
            x="variant",
            y="tokens",
            color="type",
            barmode="stack",
            title="Repartition des tokens",
        )

        st.plotly_chart(
            fig_tokens,
            use_container_width=True,
        )

        # Variabilite des couts par repetition
        per_run = repeated_df.groupby(
            ["repetition", "variant"],
            as_index=False,
        ).agg(
            tokens=("total_tokens", "sum"),
            cost_usd=("cost_usd", "sum"),
        )

        fig_cost = px.line(
            per_run,
            x="repetition",
            y="cost_usd",
            color="variant",
            markers=True,
            title="Evolution du cout par repetition",
        )

        st.plotly_chart(
            fig_cost,
            use_container_width=True,
        )

        st.caption(
            f"{len(repeated_df)} appels analyses. "
            "Qualite non evaluee pour cette experience. "
            "Les couts sont des estimations."
        )


st.caption(
    "Couts estimes selon pricing.json. "
    "Exactitude mesuree par correspondance stricte. "
    "Ces resultats ne constituent pas une validation production."
)
