"""Prepare model-specific source/target datasets from BigBio corpora (Typer CLI).

For each selected dataset (MedMentions, EMEA, MEDLINE) and model name, creates
pickle files with token-inserted source/target sequences. Optionally includes
synthetic training data (SynthMM / SynthQUAERO JSON) if available.
"""

import json
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from datasets import load_dataset

from syncabel.parse_data import process_bigbio_dataset

app = typer.Typer(
    help="Preprocess BigBio datasets into model-specific train/dev/test pickles."
)

DEFAULT_MODELS = [
    "google/mt5-large",
    "facebook/bart-large",
    "GanjinZero/biobart-v2-large",
    "facebook/genre-linking-blink",
    "facebook/mbart-large-50",
]


def _load_json_if_exists(path: Path):
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _build_mappings(umls_parquet: Path):
    df = pl.read_parquet(umls_parquet)
    syn_df = df.with_columns(Syn=pl.col("Entity").str.split(" of type ").list.get(0))
    cui_to_syn = dict(
        syn_df.group_by("CUI").agg([pl.col("Entity").unique()]).iter_rows()
    )
    return syn_df, cui_to_syn


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _dump(obj, path: Path):
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=-1)


def _iter_selected(options: Optional[Iterable[str]], all_list: list[str]) -> list[str]:
    if not options:
        return all_list
    return [m for m in options if m in all_list]


def _process_single_dataset(
    name: str,
    hf_id: str,
    hf_config: str,
    synth_data,
    syn_df,
    cui_to_syn,
    model_names: list[str],
    start_entity: str,
    end_entity: str,
    start_tag: str,
    end_tag: str,
    out_root: Path,
):
    typer.echo(f"→ Loading dataset {hf_id}:{hf_config} ...")
    ds = load_dataset(hf_id, name=hf_config)
    data_folder = out_root / name
    _ensure_dir(data_folder)

    # Determine sentence tokenizer language: MedMentions (English), else French for QUAERO variants.
    language = "english" if name == "MedMentions" else "french"

    for model_name in model_names:
        typer.echo(f"  • Processing model {model_name}")
        if synth_data is not None:
            synth_src, synth_tgt = process_bigbio_dataset(
                synth_data,
                start_entity,
                end_entity,
                start_tag,
                end_tag,
                natural=True,
                CUI_to_Syn=cui_to_syn,
                Syn_to_annotation=syn_df,
                model_name=model_name,
                language=language,
            )
        else:
            synth_src, synth_tgt = [], []

        # Build splits dict only for existing keys to avoid static type checker complaints
        splits = {"train": ds["train"]}
        if "validation" in ds:
            splits["validation"] = ds["validation"]
        if "test" in ds:
            splits["test"] = ds["test"]
        processed = {}
        for split_name, split_data in splits.items():
            if not split_data:
                continue
            src, tgt = process_bigbio_dataset(
                split_data,
                start_entity,
                end_entity,
                start_tag,
                end_tag,
                natural=True,
                CUI_to_Syn=cui_to_syn,
                Syn_to_annotation=syn_df,
                model_name=model_name,
                language=language,
            )
            processed[split_name] = (src, tgt)

        # Write outputs
        if synth_src:
            _dump(synth_src, data_folder / f"synth_train_source_{model_name}.pkl")
            _dump(synth_tgt, data_folder / f"synth_train_target_{model_name}.pkl")
        for split_name, (src, tgt) in processed.items():
            _dump(src, data_folder / f"{split_name}_source_{model_name}.pkl")
            _dump(tgt, data_folder / f"{split_name}_target_{model_name}.pkl")


@app.command()
def run(
    datasets: list[str] = typer.Option(
        ["MedMentions", "EMEA", "MEDLINE"],
        help="Datasets to process (subset of MedMentions, EMEA, MEDLINE)",
    ),
    models: list[str] = typer.Option(
        None, help="Subset of model names; defaults to all supported if omitted"
    ),
    start_entity: str = typer.Option("[", help="Start entity marker"),
    end_entity: str = typer.Option("]", help="End entity marker"),
    start_tag: str = typer.Option("{", help="Start tag marker"),
    end_tag: str = typer.Option("}", help="End tag marker"),
    synth_mm_path: Path = typer.Option(
        Path("data/synthetic_data/SynthMM/SynthMM_bigbio.json"),
        help="Synthetic MM JSON",
    ),
    synth_quaero_path: Path = typer.Option(
        Path("data/synthetic_data/SynthQUAERO/SynthQUAERO_bigbio.json"),
        help="Synthetic QUAERO JSON",
    ),
    umls_mm_parquet: Path = typer.Option(
        Path("data/UMLS_processed/MM/all_disambiguated.parquet"), help="UMLS MM parquet"
    ),
    umls_quaero_parquet: Path = typer.Option(
        Path("data/UMLS_processed/QUAERO/all_disambiguated.parquet"),
        help="UMLS QUAERO parquet",
    ),
    out_root: Path = typer.Option(
        Path("data/final_data"), help="Root output directory"
    ),
) -> None:
    """Run preprocessing pipeline for selected datasets and models."""
    model_list = _iter_selected(models, DEFAULT_MODELS)
    typer.echo(f"Models: {model_list}")

    # Load UMLS mapping resources
    syn_mm_df, cui_to_syn_mm = _build_mappings(umls_mm_parquet)
    syn_quaero_df, cui_to_syn_quaero = _build_mappings(umls_quaero_parquet)

    # Synthetic data (optional)
    synth_mm = _load_json_if_exists(synth_mm_path)
    synth_quaero = _load_json_if_exists(synth_quaero_path)
    if synth_mm is None:
        typer.echo(
            "⚠️ SynthMM not found; skipping synthetic augmentation for MedMentions."
        )
    if synth_quaero is None:
        typer.echo(
            "⚠️ SynthQUAERO not found; skipping synthetic augmentation for QUAERO-based datasets."
        )

    # Dispatch per dataset
    if "MedMentions" in datasets:
        _process_single_dataset(
            "MedMentions",
            "bigbio/medmentions",
            "medmentions_st21pv_bigbio_kb",
            synth_mm,
            syn_mm_df,
            cui_to_syn_mm,
            model_list,
            start_entity,
            end_entity,
            start_tag,
            end_tag,
            out_root,
        )
    if "EMEA" in datasets:
        _process_single_dataset(
            "EMEA",
            "bigbio/quaero",
            "quaero_emea_bigbio_kb",
            synth_quaero,
            syn_quaero_df,
            cui_to_syn_quaero,
            model_list,
            start_entity,
            end_entity,
            start_tag,
            end_tag,
            out_root,
        )
    if "MEDLINE" in datasets:
        _process_single_dataset(
            "MEDLINE",
            "bigbio/quaero",
            "quaero_medline_bigbio_kb",
            synth_quaero,
            syn_quaero_df,
            cui_to_syn_quaero,
            model_list,
            start_entity,
            end_entity,
            start_tag,
            end_tag,
            out_root,
        )
    typer.echo("✅ Preprocessing complete.")


if __name__ == "__main__":  # pragma: no cover
    app()
