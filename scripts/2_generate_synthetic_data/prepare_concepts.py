"""Prepare concept user prompts for synthetic data generation.

Turns raw UMLS concept + definition parquet files into per-CUI user prompts stored
in chunked parquet files ready for LLM generation (used by generate_synth scripts).

Usage:
    python prepare_concepts.py --mm-path data/MM_2017_all.parquet --mm-def data/UMLS_MM/umls_def.parquet \
         --out-dir data/user_prompts_MM --chunk-size 2500

Outputs: sample_{i}.parquet each containing columns: CUI, user_prompt.
"""

from pathlib import Path

import polars as pl
import typer
from tqdm import tqdm

app = typer.Typer(
    help="Build per-CUI user prompts from concept mentions and definitions."
)


def _load_join(concepts_path: Path, defs_path: Path) -> pl.DataFrame:
    concepts = pl.read_parquet(concepts_path)
    defs = pl.read_parquet(defs_path)
    return concepts.join(defs, on="CUI", how="left")


def clean_natural(text: str) -> str:
    return (
        text.replace("\xa0", " ")
        .replace("{", "(")
        .replace("}", ")")
        .replace("[", "(")
        .replace("]", ")")
    )


def build_templates(df: pl.DataFrame) -> pl.DataFrame:
    # First extract and process all mentions
    processed_df = (
        df.with_columns(Syn=pl.col("Entity").str.split(" of type ").list.first())
        .group_by("CUI")
        .agg(
            pl.col("Syn").unique().alias("mentions"),
            pl.col("DEF").first().alias("definitions"),
            pl.col("Syn").n_unique().alias("mention_count"),
        )
    )

    # Define functions with explicit return types
    def format_definitions(defs: list[str]) -> str:  # type: ignore
        return "\n".join(f"* {i + 1}. {d}" for i, d in enumerate(defs)) + "\n"  # type: ignore

    def format_mentions(mentions: list[str]) -> str:
        return clean_natural("'" + "', '".join(mentions) + "'")

    # Process with explicit return types to avoid warnings
    return (
        processed_df.filter(pl.col("definitions").is_not_null())
        .with_columns(
            definitions_processed=pl.col("definitions")
            .map_elements(format_definitions, return_dtype=pl.String)
            .fill_null("No definition found\n"),
            mentions_processed=pl.col("mentions").map_elements(
                format_mentions, return_dtype=pl.String
            ),
        )
        .with_columns(
            user_prompt=pl.concat_str([
                pl.lit("- **CUI**:\n"),
                pl.col("CUI"),
                pl.lit("\n- **Definitions**:\n"),
                pl.col("definitions_processed"),
                pl.lit("- **Mentions**:\n"),
                pl.col("mentions_processed"),
                pl.lit("\n"),
            ])
        )
        .select(["CUI", "user_prompt"])
    )


def _write_chunks(df: pl.DataFrame, out_dir: Path, chunk_size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(0, len(df), chunk_size), desc=f"Writing {out_dir}"):
        chunk = df.slice(i, chunk_size)
        path = out_dir / f"sample_{i}.parquet"
        chunk.write_parquet(path)


@app.command()
def run(
    mm_path: Path = typer.Option(None, help="Path to MM concepts parquet"),
    mm_def: Path = typer.Option(None, help="Path to MM definitions parquet"),
    quaero_path: Path = typer.Option(None, help="Path to QUAERO concepts parquet"),
    quaero_def: Path = typer.Option(None, help="Path to QUAERO definitions parquet"),
    out_mm: Path = typer.Option(
        Path("data/user_prompts_MM"), help="Output dir for MM prompts"
    ),
    out_quaero: Path = typer.Option(
        Path("data/user_prompts_quaero"), help="Output dir for QUAERO prompts"
    ),
    shuffle: bool = typer.Option(True, help="Shuffle concepts (sample fraction=1)"),
    chunk_size: int = typer.Option(2500, help="Chunk size for output parquet files"),
) -> None:
    """Generate user prompts for one or both datasets depending on provided paths."""
    if not any([mm_path and mm_def, quaero_path and quaero_def]):
        raise typer.BadParameter(
            "Provide at least one dataset (MM or QUAERO) with its definition file."
        )

    if mm_path and mm_def:
        mm_joined = _load_join(mm_path, mm_def)
        # If only definitions file is desired (original logic sampled from defs directly)
        if "DEF" in mm_joined.columns:
            mm_filtered = mm_joined.filter(pl.col("DEF").is_not_null())
            if shuffle:
                mm_filtered = mm_filtered.sample(fraction=1)
            user_prompt_mm = build_templates(mm_filtered)
            _write_chunks(user_prompt_mm, out_mm, chunk_size)
            typer.echo(f"MM concepts written to {out_mm}")

    if quaero_path and quaero_def:
        q_joined = _load_join(quaero_path, quaero_def)
        if "DEF" in q_joined.columns:
            q_filtered = q_joined.filter(pl.col("DEF").is_not_null())
            if shuffle:
                q_filtered = q_filtered.sample(fraction=1)
            user_prompt_q = build_templates(q_filtered)
            _write_chunks(user_prompt_q, out_quaero, chunk_size)
            typer.echo(f"QUAERO concepts written to {out_quaero}")


if __name__ == "__main__":
    app()
