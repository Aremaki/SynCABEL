import pickle

import polars as pl
import spacy
from datasets import load_dataset
from tqdm import tqdm
from unidecode import unidecode

# Load scispacy
nlp = spacy.load("en_core_sci_sm")
# Add the abbreviation pipe to the spacy pipeline.
nlp.add_pipe("abbreviation_detector")
nlp.add_pipe(
    "scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"}
)
linker = nlp.get_pipe("scispacy_linker")

# Load Data
all_data = {}
all_data["MM"] = load_dataset(
    "bigbio/medmentions", name="medmentions_st21pv_bigbio_kb", split="test"
)
all_data["MEDLINE"] = load_dataset(
    "bigbio/quaero", name="quaero_medline_bigbio_kb", split="test"
)
all_data["EMEA"] = load_dataset(
    "bigbio/quaero", name="quaero_emea_bigbio_kb", split="test"
)
sem_types = {
    "T005": "Virus",
    "T007": "Bacterium",
    "T017": "Anatomical Structure",
    "T022": "Body System",
    "T031": "Body Substance",
    "T033": "Finding",
    "T037": "Injury or Poisoning",
    "T038": "Biologic Function",
    "T058": "Health Care Activity",
    "T062": "Research Activity",
    "T074": "Medical Device",
    "T082": "Spatial Concept",
    "T091": "Biomedical Occupation or Discipline",
    "T092": "Organization",
    "T097": "Professional or Occupational Group",
    "T098": "Population Group",
    "T103": "Chemical",
    "T168": "Food",
    "T170": "Intellectual Product",
    "T201": "Clinical Attribute",
    "T204": "Eukaryote",
}
MM_umls_df = pl.read_parquet("../data/legal_umls_token_2017_short_syn_all_main.parquet")
MM_CUI_to_type = dict(
    MM_umls_df.group_by("CUI").agg([pl.col("GROUP").unique()]).iter_rows()
)
for dataset, data in all_data.items():
    results = {
        "doc_id": [],
        "mention": [],
        "category": [],
        "semtype": [],
        "CUI_gold": [],
        "matched_mention": [],
        "CUI": [],
        "Score": [],
    }
    page_id = 0
    for page in tqdm(data):
        for entity in page["entities"]:
            if not entity["normalized"]:
                print("⚠️ No CUI found")
                continue
            cui = entity["normalized"][0]["db_id"]
            if dataset == "MM":
                sem_type = sem_types[entity["type"]]
                category = MM_CUI_to_type[cui][0]
            else:
                category = entity["type"]
                sem_type = entity["type"]
            if not entity["text"]:
                continue
            mention = unidecode(entity["text"][0])
            pred = linker.candidate_generator([mention], 1)[0][0]  # type: ignore
            results["doc_id"].append(page_id)
            results["mention"].append(mention)
            results["category"].append(category)
            results["semtype"].append(sem_type)
            results["CUI_gold"].append(cui)
            results["matched_mention"].append(pred.aliases[0])
            results["CUI"].append(pred.concept_id)
            results["Score"].append(pred.similarities[0])
        page_id += 1

    all_results = pl.DataFrame(results).with_columns(
        success=(pl.col("CUI_gold").eq_missing(pl.col("CUI"))),
        fail=(pl.col("CUI_gold").ne_missing(pl.col("CUI"))),
        multi_word_mention=pl.col("mention").str.split(" ").list.len() > 1,
        direct_match=pl.col("mention").str.to_lowercase()
        == pl.col("matched_mention").str.to_lowercase(),
    )
    with open(f"data/results/scispacy/{dataset}_all_results.pkl", "wb") as file:
        pickle.dump(all_results, file)
