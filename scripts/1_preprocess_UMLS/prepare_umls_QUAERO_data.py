import polars as pl

# Prepare UMLS for QUAERO
umls_df = pl.read_parquet("data/UMLS/umls_codes.parquet")
umls_title_syn_df = pl.read_parquet("data/UMLS/umls_title_syn.parquet")
umls_semantic_df = pl.read_parquet("data/UMLS/umls_semantic.parquet")
umls_semantic_df = umls_semantic_df.filter(
    pl.col("CATEGORY").is_in([
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
    ])
)
legal_umls_token = umls_df.join(umls_semantic_df, on="CUI").join(
    umls_title_syn_df, on="CUI", how="left"
)
legal_umls_token_fr = legal_umls_token.select([
    "CUI",
    "SEM_NAME",
    "CATEGORY",
    "UMLS_Title_fr",
    "UMLS_alias_fr",
]).with_columns(pl.lit("fr").alias("lang"))
legal_umls_token_en = legal_umls_token.select([
    "CUI",
    "SEM_NAME",
    "CATEGORY",
    "UMLS_Title_main",
    "UMLS_Title_en",
    "UMLS_alias_en",
]).with_columns(pl.lit("en").alias("lang"))
legal_umls_token_fr = legal_umls_token_fr.explode("UMLS_Title_fr").explode(
    "UMLS_alias_fr"
)
legal_umls_token_en = (
    legal_umls_token_en.explode("UMLS_Title_main")
    .explode("UMLS_Title_en")
    .explode("UMLS_alias_en")
)
legal_umls_token_UMLS_Title_main = (
    legal_umls_token_en.select([
        "CUI",
        "lang",
        "SEM_NAME",
        "CATEGORY",
        pl.col("UMLS_Title_main").alias("Syn"),
    ])
    .filter((pl.col("Syn") != "") & (pl.col("Syn").is_not_null()))
    .with_columns(is_main=pl.lit(False))
)
legal_umls_token_UMLS_Title_fr = (
    legal_umls_token_fr.select([
        "CUI",
        "lang",
        "SEM_NAME",
        "CATEGORY",
        pl.col("UMLS_Title_fr").alias("Syn"),
    ])
    .filter((pl.col("Syn") != "") & (pl.col("Syn").is_not_null()))
    .with_columns(is_main=pl.lit(True))
)
legal_umls_token_UMLS_Title_en = (
    legal_umls_token_en.select([
        "CUI",
        "lang",
        "SEM_NAME",
        "CATEGORY",
        pl.col("UMLS_Title_en").alias("Syn"),
    ])
    .filter((pl.col("Syn") != "") & (pl.col("Syn").is_not_null()))
    .with_columns(is_main=pl.lit(False))
)
legal_umls_token_UMLS_alias_fr = (
    legal_umls_token_fr.select([
        "CUI",
        "lang",
        "SEM_NAME",
        "CATEGORY",
        pl.col("UMLS_alias_fr").alias("Syn"),
    ])
    .filter((pl.col("Syn") != "") & (pl.col("Syn").is_not_null()))
    .with_columns(is_main=pl.lit(False))
)
legal_umls_token_UMLS_alias_en = (
    legal_umls_token_en.select([
        "CUI",
        "lang",
        "SEM_NAME",
        "CATEGORY",
        pl.col("UMLS_alias_en").alias("Syn"),
    ])
    .filter((pl.col("Syn") != "") & (pl.col("Syn").is_not_null()))
    .with_columns(is_main=pl.lit(False))
)
legal_umls_token = pl.concat([
    legal_umls_token_UMLS_Title_main,
    legal_umls_token_UMLS_Title_fr,
    legal_umls_token_UMLS_Title_en,
    legal_umls_token_UMLS_alias_fr,
    legal_umls_token_UMLS_alias_en,
])

# Clean the Synonyms
legal_umls_token = legal_umls_token.with_columns(
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
    .then(pl.col("Syn").str.slice(0, 1).str.to_uppercase() + pl.col("Syn").str.slice(1))
    .otherwise(pl.col("Syn"))
    .alias("Syn")
)
legal_umls_token = legal_umls_token.unique()

# Disambiguate the Synonyms
legal_umls_token = legal_umls_token.with_columns(
    Entity=pl.concat_str(
        [
            pl.col("Syn"),
            pl.lit("of type"),
            pl.col("SEM_NAME"),
        ],
        separator=" ",
        ignore_nulls=False,
    ).str.replace_all("\xa0", " ", literal=True)
)
legal_umls_token = legal_umls_token.with_columns(
    Entity_full=pl.concat_str(
        [
            pl.col("Syn"),
            pl.lit("of type"),
            pl.col("SEM_NAME"),
            pl.col("CUI"),
        ],
        separator=" ",
        ignore_nulls=False,
    ).str.replace_all("\xa0", " ", literal=True)
)
legal_umls_token_fr = (
    legal_umls_token.filter(pl.col("lang") == "fr")
    .select(["CUI", "SEM_NAME", "CATEGORY", "Syn", "Entity", "Entity_full", "is_main"])
    .unique()
)
legal_umls_token_all = legal_umls_token.select([
    "CUI",
    "SEM_NAME",
    "CATEGORY",
    "Syn",
    "Entity",
    "Entity_full",
    "is_main",
]).unique()


def filter_only_non_amibuguous_syn(legal_umls_token):
    legal_umls_token_n_cui = (
        legal_umls_token.group_by("Syn")
        .agg(pl.col("CUI").n_unique().alias("n_CUI"))
        .join(legal_umls_token, on="Syn")
    )
    legal_umls_token_one_cui = legal_umls_token_n_cui.filter(pl.col("n_CUI") == 1)
    legal_umls_token_one_cui = legal_umls_token_one_cui.with_columns(
        Entity=pl.col("Syn")
    ).select(["CUI", "Entity", "SEM_NAME", "CATEGORY", "is_main"])
    legal_umls_token_cuis = legal_umls_token_n_cui.filter(pl.col("n_CUI") > 1).drop(
        "n_CUI"
    )
    legal_umls_token_n_entity = (
        legal_umls_token_cuis.group_by("Entity")
        .agg(pl.col("CUI").n_unique().alias("n_CUI"))
        .join(legal_umls_token_cuis, on="Entity")
    )
    legal_umls_token_one_cui_type = legal_umls_token_n_entity.filter(
        pl.col("n_CUI") == 1
    ).select(["CUI", "Entity", "SEM_NAME", "CATEGORY", "is_main"])
    legal_umls_token_cuis = legal_umls_token_n_entity.filter(pl.col("n_CUI") > 1).drop(
        "n_CUI"
    )
    legal_umls_token_n_entity = (
        legal_umls_token_cuis.group_by("Entity_full")
        .agg(pl.col("CUI").n_unique().alias("n_CUI"))
        .join(legal_umls_token_cuis, on="Entity_full")
    )
    legal_umls_token_one_cui_type_code = legal_umls_token_n_entity.filter(
        pl.col("n_CUI") == 1
    )
    legal_umls_token_one_cui_type_code = (
        legal_umls_token_one_cui_type_code.with_columns(
            Entity=pl.col("Entity_full")
        ).select(["CUI", "Entity", "SEM_NAME", "CATEGORY", "is_main"])
    )
    legal_umls_token_limited = pl.concat([
        legal_umls_token_one_cui_type,
        legal_umls_token_one_cui,
        legal_umls_token_one_cui_type_code,
    ])
    return legal_umls_token_limited


legal_umls_token_all_filtered = filter_only_non_amibuguous_syn(legal_umls_token_all)
legal_umls_token_fr_filtered = filter_only_non_amibuguous_syn(legal_umls_token_fr)

# Write to Parquet file
legal_umls_token_all_filtered.write_parquet("data/quaero_2014_all.parquet")
legal_umls_token_fr_filtered.write_parquet("data/quaero_2014_fr.parquet")
