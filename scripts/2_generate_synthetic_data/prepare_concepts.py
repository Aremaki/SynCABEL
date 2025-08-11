import polars as pl
from tqdm import tqdm

# Load UMLS MM
umls_MM = pl.read_parquet("data/MM_2017_all.parquet")
umls_quaero = pl.read_parquet("data/QUAERO_2017_all.parquet")
umls_def_MM = pl.read_parquet("data/UMLS_MM/umls_def.parquet")
umls_def_quaero = pl.read_parquet("data/UMLS_QUAERO/umls_def.parquet")

umls_MM = umls_MM.join(umls_def_MM, on="CUI", how="left")
umls_quaero = umls_quaero.join(umls_def_quaero, on="CUI", how="left")
all_concepts_MM = umls_def_MM.sample(fraction=1).filter(pl.col("DEF").is_not_null())
all_concepts_quaero = umls_def_quaero.sample(fraction=1).filter(
    pl.col("DEF").is_not_null()
)


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
    def format_definitions(defs: list[str] | None) -> str:
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


# Process in chunks if needed
def process_in_chunks(df: pl.DataFrame, chunk_size: int = 100000) -> pl.DataFrame:
    chunks = []
    for i in tqdm(range(0, len(df), chunk_size), desc="Processing chunks"):
        chunk = df.slice(i, chunk_size)
        chunks.append(build_templates(chunk))
    return pl.concat(chunks)


user_prompt_MM = build_templates(all_concepts_MM)
user_prompt_quaero = build_templates(all_concepts_quaero)


chunk_size = 2500
for i in tqdm(range(0, len(user_prompt_MM), chunk_size), desc="Processing chunks"):
    chunk = user_prompt_MM.slice(i, chunk_size)
    chunk.write_parquet("data/user_prompts_MM/sample_{i}.parquet")
for i in tqdm(range(0, len(user_prompt_quaero), chunk_size), desc="Processing chunks"):
    chunk = user_prompt_quaero.slice(i, chunk_size)
    chunk.write_parquet("data/user_prompts_quaero/sample_{i}.parquet")
