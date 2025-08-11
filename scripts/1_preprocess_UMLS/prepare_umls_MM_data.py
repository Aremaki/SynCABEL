import polars as pl

# Prepare UMLS for MM
umls_df = pl.read_parquet("data/UMLS/umls_codes.parquet")
umls_title_syn_df = pl.read_parquet("data/UMLS/umls_title_syn.parquet")
umls_semantic_df = pl.read_parquet("data/UMLS/umls_semantic.parquet")

# Filter for specific ST21-pv
umls_semantic_df = umls_semantic_df.filter(
    pl.col("TREE_CODE").str.starts_with("A1.1.4")  # Virus
    | pl.col("TREE_CODE").str.starts_with("A1.1.2")  # Bacterium
    | pl.col("TREE_CODE").str.starts_with("A1.2")  # Anatomical Structure
    | pl.col("TREE_CODE").str.starts_with("A2.1.4.1")  # Body System
    | pl.col("TREE_CODE").str.starts_with("A1.4.2")  # Body Substance
    | pl.col("TREE_CODE").str.starts_with("A2.2")  # Finding
    | pl.col("TREE_CODE").str.starts_with("B2.3")  # Injury or Poisoning
    | pl.col("TREE_CODE").str.starts_with("B2.2.1")  # Biologic Function
    | pl.col("TREE_CODE").str.starts_with("B1.3.1")  # Health Care Activity
    | pl.col("TREE_CODE").str.starts_with("B1.3.2")  # Research Activity
    | pl.col("TREE_CODE").str.starts_with("A1.3.1")  # Medical Device
    | pl.col("TREE_CODE").str.starts_with("A2.1.5")  # Spatial Concept
    | pl.col("TREE_CODE").str.starts_with(
        "A2.6.1"
    )  # Biomedical Occupation or Discipline
    | pl.col("TREE_CODE").str.starts_with("A2.7")  # Organization
    | pl.col("TREE_CODE").str.starts_with(
        "A2.9.1"
    )  # Professional or Occupational Group
    | pl.col("TREE_CODE").str.starts_with("A2.9.2")  # Population Group
    | pl.col("TREE_CODE").str.starts_with("A1.4.1")  # Chemical
    | pl.col("TREE_CODE").str.starts_with("A1.4.3")  # Food
    | pl.col("TREE_CODE").str.starts_with("A2.4")  # Intellectual Product
    | pl.col("TREE_CODE").str.starts_with("A2.3.1")  # Clinical Attribute
    | pl.col("TREE_CODE").str.starts_with("A1.1.3")  # Eukaryote
)


# Mapping from TREE_CODE prefix to semantic name
prefix_to_name = {
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
# Build the conditional expression
semantic_expr = pl.when(
    pl.col("TREE_CODE").str.starts_with(list(prefix_to_name.keys())[0])
).then(pl.lit(prefix_to_name[list(prefix_to_name.keys())[0]]))

for prefix, name in list(prefix_to_name.items())[1:]:
    semantic_expr = semantic_expr.when(
        pl.col("TREE_CODE").str.starts_with(prefix)
    ).then(pl.lit(name))

semantic_expr = semantic_expr.otherwise(None)

# Add SEMANTIC_NAME column
umls_semantic_df = umls_semantic_df.with_columns(semantic_expr.alias("SEM_NAME_MM"))

# Join the DataFrames
legal_umls_token = umls_df.join(umls_semantic_df, on="CUI").join(
    umls_title_syn_df, on="CUI", how="left"
)

# Select and rename columns
legal_umls_token_fr = legal_umls_token.select([
    "CUI",
    "SEM_NAME",
    "SEM_NAME_MM",
    "UMLS_Title_fr",
    "UMLS_alias_fr",
    "CATEGORY",
    "GROUP",
]).with_columns(pl.lit("fr").alias("lang"))
legal_umls_token_en = legal_umls_token.select([
    "CUI",
    "SEM_NAME",
    "SEM_NAME_MM",
    "GROUP",
    "UMLS_Title_main",
    "UMLS_Title_en",
    "UMLS_alias_en",
    "CATEGORY",
]).with_columns(pl.lit("en").alias("lang"))

# Explode the lists in the DataFrame
legal_umls_token_fr = legal_umls_token_fr.explode("UMLS_Title_fr").explode(
    "UMLS_alias_fr"
)
legal_umls_token_en = (
    legal_umls_token_en.explode("UMLS_Title_main")
    .explode("UMLS_Title_en")
    .explode("UMLS_alias_en")
)

# Create separate DataFrames for each title and alias
legal_umls_token_UMLS_Title_main = (
    legal_umls_token_en.select([
        "CUI",
        "lang",
        "SEM_NAME",
        "SEM_NAME_MM",
        "GROUP",
        "CATEGORY",
        pl.col("UMLS_Title_main").alias("Syn"),
    ])
    .filter((pl.col("Syn") != "") & (pl.col("Syn").is_not_null()))
    .with_columns(is_main=pl.lit(True))
)
legal_umls_token_UMLS_Title_fr = (
    legal_umls_token_fr.select([
        "CUI",
        "lang",
        "SEM_NAME",
        "SEM_NAME_MM",
        "GROUP",
        "CATEGORY",
        pl.col("UMLS_Title_fr").alias("Syn"),
    ])
    .filter((pl.col("Syn") != "") & (pl.col("Syn").is_not_null()))
    .with_columns(is_main=pl.lit(False))
)
legal_umls_token_UMLS_Title_en = (
    legal_umls_token_en.select([
        "CUI",
        "lang",
        "SEM_NAME",
        "SEM_NAME_MM",
        "GROUP",
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
        "SEM_NAME_MM",
        "GROUP",
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
        "SEM_NAME_MM",
        "GROUP",
        "CATEGORY",
        pl.col("UMLS_alias_en").alias("Syn"),
    ])
    .filter((pl.col("Syn") != "") & (pl.col("Syn").is_not_null()))
    .with_columns(is_main=pl.lit(False))
)
# Concatenate all the DataFrames
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
legal_umls_token_all_filtered.write_parquet("data/MM_2017_all.parquet")
legal_umls_token_fr_filtered.write_parquet("data/MM_2017_fr.parquet")
