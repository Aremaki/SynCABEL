#!/usr/bin/env python
"""Prepare UMLS synonym/semantic parquet files for downstream datasets.

Two supported dataset modes:
  - MM      (MedMentions ST21-pv style selection / adds SEM_NAME_MM)
  - QUAERO  (Category-based selection used for QUAERO MEDLINE/EMEA)

It expects the extraction step to have produced the three parquet files:
  umls_codes.parquet
  umls_title_syn.parquet
  umls_semantic.parquet
inside a directory passed via --umls-dir (typically data/UMLS_processed/<dataset>/).

Outputs (by default written under data/UMLS_processed/<dataset>/): all_disambiguated.parquet, fr_disambiguated.parquet
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer

app = typer.Typer(help="Prepare UMLS data for specific downstream datasets.")

# ST21-pv style tree code prefixes (mapped to names) used for MedMentions filtering
MM_PREFIX_TO_NAME = {
    "A1.1.4": "Virus",
    "A1.1.2": "Bacterium",
    "A1.2": "Anatomical Structure",
    "A2.1.4.1": "Body System",
    "A1.4.2": "Body Substance",
    "A2.2": "Finding",
    "B2.3": "Injury or Poisoning",
    "B2.2.1": "Biologic Function",
    "B1.3.1": "Health Care Activity",
    "B1.3.2": "Research Activity",
    "A1.3.1": "Medical Device",
    "A2.1.5": "Spatial Concept",
    "A2.6.1": "Biomedical Occupation or Discipline",
    "A2.7": "Organization",
    "A2.9.1": "Professional or Occupational Group",
    "A2.9.2": "Population Group",
    "A1.4.1": "Chemical",
    "A1.4.3": "Food",
    "A2.4": "Intellectual Product",
    "A2.3.1": "Clinical Attribute",
    "A1.1.3": "Eukaryote",
}

# Category whitelist for QUAERO filtering
QUAERO_CATEGORIES = [
    "ANAT",
    "CHEM",
    "DEVI",
    "DISO",
    "GEOG",
    "LIVB",
    "OBJC",
    "PHEN",
    "PHYS",
    "PROC",
]


def _build_mm_semantic_expr():
    prefixes = list(MM_PREFIX_TO_NAME.items())
    first_pref, first_name = prefixes[0]
    expr = pl.when(pl.col("TREE_CODE").str.starts_with(first_pref)).then(
        pl.lit(first_name)
    )
    for pref, name in prefixes[1:]:
        expr = expr.when(pl.col("TREE_CODE").str.starts_with(pref)).then(pl.lit(name))
    return expr.otherwise(None).alias("SEM_NAME_MM")


def _clean_syn(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("Syn")
        .str.replace_all("\xa0", " ", literal=True)
        .str.replace_all(r"\s*\(NOS\)\s*$", "")
        .str.replace_all(r",\sNOS\s*$", "")
        .str.replace_all(r"\sNOS\s*$", "")
        .str.replace_all(r"\s*\(SAI\)\s*$", "")
        .str.replace_all(r",\sSAI\s*$", "")
        .str.replace_all(r"\sSAI\s*$", "")
    ).with_columns(
        pl.when(
            pl.col("Syn").str.slice(0, 1).str.to_lowercase()
            == pl.col("Syn").str.slice(0, 1)
        )
        .then(
            pl.col("Syn").str.slice(0, 1).str.to_uppercase()
            + pl.col("Syn").str.slice(1)
        )
        .otherwise(pl.col("Syn"))
        .alias("Syn")
    )


def _disambiguate(
    df: pl.DataFrame, extra_sem_name: bool = False
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Add disambiguation columns
    if extra_sem_name and "SEM_NAME_MM" in df.columns:
        base_cols = ["Syn", "SEM_NAME", "SEM_NAME_MM", "CUI"]
    else:
        base_cols = ["Syn", "SEM_NAME", "CUI"]
    df = df.with_columns(
        Entity=pl.concat_str(
            [pl.col("Syn"), pl.lit("of type"), pl.col("SEM_NAME")], separator=" "
        ),
        Entity_full=pl.concat_str(base_cols, separator=" "),
    )
    # Language specific subset
    df_fr = (
        df.filter(pl.col("lang") == "fr")
        .select([
            "CUI",
            "SEM_NAME",
            "CATEGORY",
            "Syn",
            "Entity",
            "Entity_full",
            "is_main",
        ])
        .unique()
    )
    df_all = df.select([
        "CUI",
        "SEM_NAME",
        "CATEGORY",
        "Syn",
        "Entity",
        "Entity_full",
        "is_main",
    ]).unique()
    return df_all, df_fr


def _filter_non_ambiguous(df: pl.DataFrame) -> pl.DataFrame:
    # Follow original multi-stage ambiguity reduction logic
    n_cui = (
        df.group_by("Syn")
        .agg(pl.col("CUI").n_unique().alias("n_CUI"))
        .join(df, on="Syn")
    )
    one_cui = (
        n_cui.filter(pl.col("n_CUI") == 1)
        .with_columns(Entity=pl.col("Syn"))
        .select(["CUI", "Entity", "SEM_NAME", "CATEGORY", "is_main"])
    )
    more = n_cui.filter(pl.col("n_CUI") > 1).drop("n_CUI")
    n_entity = (
        more.group_by("Entity")
        .agg(pl.col("CUI").n_unique().alias("n_CUI"))
        .join(more, on="Entity")
    )
    one_cui_type = n_entity.filter(pl.col("n_CUI") == 1).select([
        "CUI",
        "Entity",
        "SEM_NAME",
        "CATEGORY",
        "is_main",
    ])
    more2 = n_entity.filter(pl.col("n_CUI") > 1).drop("n_CUI")
    n_entity_full = (
        more2.group_by("Entity_full")
        .agg(pl.col("CUI").n_unique().alias("n_CUI"))
        .join(more2, on="Entity_full")
    )
    one_cui_full = (
        n_entity_full.filter(pl.col("n_CUI") == 1)
        .with_columns(Entity=pl.col("Entity_full"))
        .select(["CUI", "Entity", "SEM_NAME", "CATEGORY", "is_main"])
    )
    return pl.concat([one_cui_type, one_cui, one_cui_full])


def _explode_language_frames(base: pl.DataFrame, is_mm: bool) -> pl.DataFrame:
    # Split language columns and explode synonyms / titles
    fr = (
        base.select([
            "CUI",
            "SEM_NAME",
            *(["SEM_NAME_MM"] if is_mm else []),
            "CATEGORY",
            "UMLS_Title_fr",
            "UMLS_alias_fr",
        ])
        .with_columns(pl.lit("fr").alias("lang"))
        .explode("UMLS_Title_fr")
        .explode("UMLS_alias_fr")
    )

    en = base.select([
        "CUI",
        "SEM_NAME",
        *(["SEM_NAME_MM"] if is_mm else []),
        "CATEGORY",
        "UMLS_Title_main",
        "UMLS_Title_en",
        "UMLS_alias_en",
    ]).with_columns(pl.lit("en").alias("lang"))
    en = en.explode("UMLS_Title_main").explode("UMLS_Title_en").explode("UMLS_alias_en")

    # Build unified rows for each type source (mark main True appropriately)
    parts = []
    if is_mm:
        sem_extra_cols = ["SEM_NAME_MM"]
    else:
        sem_extra_cols = []

    def _mk(df: pl.DataFrame, col: str, is_main: bool) -> pl.DataFrame:
        return (
            df.select([
                "CUI",
                "lang",
                "SEM_NAME",
                *sem_extra_cols,
                "CATEGORY",
                pl.col(col).alias("Syn"),
            ])
            .filter((pl.col("Syn") != "") & (pl.col("Syn").is_not_null()))
            .with_columns(is_main=pl.lit(is_main))
        )

    # Titles and aliases
    if "UMLS_Title_main" in en.columns:  # main (English main title)
        parts.append(_mk(en, "UMLS_Title_main", is_main=True if not is_mm else True))
    if "UMLS_Title_fr" in fr.columns:
        parts.append(_mk(fr, "UMLS_Title_fr", is_main=False if not is_mm else False))
    if "UMLS_Title_en" in en.columns:
        parts.append(_mk(en, "UMLS_Title_en", is_main=False))
    if "UMLS_alias_fr" in fr.columns:
        parts.append(_mk(fr, "UMLS_alias_fr", is_main=False))
    if "UMLS_alias_en" in en.columns:
        parts.append(_mk(en, "UMLS_alias_en", is_main=False))

    return pl.concat(parts)


def _prepare_mm(
    codes: pl.DataFrame, titles: pl.DataFrame, semantic: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Filter semantic rows by tree code prefixes
    semantic_filtered = semantic.filter(
        pl.any_horizontal(*[
            pl.col("TREE_CODE").str.starts_with(p) for p in MM_PREFIX_TO_NAME
        ])
    ).with_columns(_build_mm_semantic_expr())
    base = codes.join(semantic_filtered, on="CUI").join(titles, on="CUI", how="left")
    exploded = _explode_language_frames(base, is_mm=True)
    exploded = _clean_syn(exploded).unique()
    all_df, fr_df = _disambiguate(exploded, extra_sem_name=True)
    return _filter_non_ambiguous(all_df), _filter_non_ambiguous(fr_df)


def _prepare_quaero(
    codes: pl.DataFrame, titles: pl.DataFrame, semantic: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    semantic_filtered = semantic.filter(pl.col("CATEGORY").is_in(QUAERO_CATEGORIES))
    base = codes.join(semantic_filtered, on="CUI").join(titles, on="CUI", how="left")
    exploded = _explode_language_frames(base, is_mm=False)
    exploded = _clean_syn(exploded).unique()
    all_df, fr_df = _disambiguate(exploded)
    return _filter_non_ambiguous(all_df), _filter_non_ambiguous(fr_df)


@app.command()
def prepare(
    dataset: str = typer.Option(..., help="Target dataset style to prepare."),
    umls_dir: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        help="Directory containing extracted UMLS parquet files.",
    ),
) -> None:
    codes_path = umls_dir / "umls_codes.parquet"
    titles_path = umls_dir / "umls_title_syn.parquet"
    semantic_path = umls_dir / "umls_semantic.parquet"
    for p in (codes_path, titles_path, semantic_path):
        if not p.exists():
            raise typer.BadParameter(f"Missing required file: {p}")
    codes = pl.read_parquet(codes_path)
    titles = pl.read_parquet(titles_path)
    semantic = pl.read_parquet(semantic_path)

    if dataset == "MM":
        all_df, fr_df = _prepare_mm(codes, titles, semantic)
    else:
        all_df, fr_df = _prepare_quaero(codes, titles, semantic)

    all_path = umls_dir / "all_disambiguated.parquet"
    fr_path = umls_dir / "fr_disambiguated.parquet"
    all_df.write_parquet(all_path)
    fr_df.write_parquet(fr_path)
    typer.echo(f"Wrote {all_path} and {fr_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
