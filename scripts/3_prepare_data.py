import json
import pickle

import polars as pl
from datasets import load_dataset

from syncabel.parse_data import process_bigbio_dataset

# Load Synthetic data
with open("data/bigbio_datasets/SynthMM.json", "rb") as json_file:
    synthMM = json.load(json_file)
with open("data/bigbio_datasets/SynthQUAERO.json", "rb") as json_file:
    synthQUAERO = json.load(json_file)

# Load human-annotated data
MM_data = load_dataset("bigbio/medmentions", name="medmentions_st21pv_bigbio_kb")
MEDLINE_data = load_dataset("bigbio/quaero", name="quaero_medline_bigbio_kb")
EMEA_data = load_dataset("bigbio/quaero", name="quaero_emea_bigbio_kb")


# Load UMLS data
umls_MM_2017 = pl.read_parquet("data/MM_2017_all.parquet")
umls_quaero_2014 = pl.read_parquet("data/QUAERO_2014_all.parquet")

# Create synonym to annotation mappings
Syn_to_annotation_MM = umls_MM_2017.with_columns(
    Syn=pl.col("Entity").str.split(" of type ").list.get(0)
)
CUI_to_Syn_MM = dict(
    Syn_to_annotation_MM.group_by("CUI").agg([pl.col("Entity").unique()]).iter_rows()
)
Syn_to_annotation_quaero = umls_quaero_2014.with_columns(
    Syn=pl.col("Entity").str.split(" of type ").list.get(0)
)
CUI_to_Syn_quaero = dict(
    Syn_to_annotation_quaero.group_by("CUI")
    .agg([pl.col("Entity").unique()])
    .iter_rows()
)

# Process MedMentions dataset
dataset_name = "MedMentions"
start_entity = "["
end_entity = "]"
start_tag = "{"
end_tag = "}"
model_names = [
    "mt5-large",
    "bart-large",
    "biobart-v2-large",
    "bart-genre",
    "mbart-large-50",
]
for model_name in model_names:
    synth_train_source, synth_train_target = process_bigbio_dataset(
        synthMM,
        start_entity,
        end_entity,
        start_tag,
        end_tag,
        natural=True,
        CUI_to_Syn=CUI_to_Syn_MM,
        Syn_to_annotation=Syn_to_annotation_MM,
        model_name=model_name,
    )
    train_source, train_target = process_bigbio_dataset(
        MM_data["train"],
        start_entity,
        end_entity,
        start_tag,
        end_tag,
        natural=True,
        CUI_to_Syn=CUI_to_Syn_MM,
        Syn_to_annotation=Syn_to_annotation_MM,
        model_name=model_name,
    )
    dev_source, dev_target = process_bigbio_dataset(
        MM_data["validation"],
        start_entity,
        end_entity,
        start_tag,
        end_tag,
        natural=True,
        CUI_to_Syn=CUI_to_Syn_MM,
        Syn_to_annotation=Syn_to_annotation_MM,
        model_name=model_name,
    )
    test_source, test_target = process_bigbio_dataset(
        MM_data["test"],
        start_entity,
        end_entity,
        start_tag,
        end_tag,
        natural=True,
        CUI_to_Syn=CUI_to_Syn_MM,
        model_name=model_name,
    )

    data_folder = f"data/preprocessed_dataset/{dataset_name}"
    with open(data_folder + f"/synth_train_source_{model_name}.pkl", "wb") as file:
        pickle.dump(synth_train_source, file, protocol=-1)
    with open(data_folder + f"/synth_train_target_{model_name}.pkl", "wb") as file:
        pickle.dump(synth_train_target, file, protocol=-1)
    with open(data_folder + f"/train_source_{model_name}.pkl", "wb") as file:
        pickle.dump(train_source, file, protocol=-1)
    with open(data_folder + f"/train_target_{model_name}.pkl", "wb") as file:
        pickle.dump(train_target, file, protocol=-1)
    with open(data_folder + f"/dev_source_{model_name}.pkl", "wb") as file:
        pickle.dump(dev_source, file, protocol=-1)
    with open(data_folder + f"/dev_target_{model_name}.pkl", "wb") as file:
        pickle.dump(dev_target, file, protocol=-1)
    with open(data_folder + f"/test_source_{model_name}.pkl", "wb") as file:
        pickle.dump(test_source, file, protocol=-1)
    with open(data_folder + f"/test_target_{model_name}.pkl", "wb") as file:
        pickle.dump(test_target, file, protocol=-1)


# Process EMEA dataset
dataset_name = "EMEA"
start_entity = "["
end_entity = "]"
start_tag = "{"
end_tag = "}"
model_names = [
    "mt5-large",
    "bart-large",
    "biobart-v2-large",
    "bart-genre",
    "mbart-large-50",
]
for model_name in model_names:
    synth_train_source, synth_train_target = process_bigbio_dataset(
        synthQUAERO,
        start_entity,
        end_entity,
        start_tag,
        end_tag,
        natural=True,
        CUI_to_Syn=CUI_to_Syn_quaero,
        Syn_to_annotation=Syn_to_annotation_quaero,
        model_name=model_name,
    )
    train_source, train_target = process_bigbio_dataset(
        EMEA_data["train"],
        start_entity,
        end_entity,
        start_tag,
        end_tag,
        natural=True,
        CUI_to_Syn=CUI_to_Syn_quaero,
        Syn_to_annotation=Syn_to_annotation_quaero,
        model_name=model_name,
    )
    dev_source, dev_target = process_bigbio_dataset(
        EMEA_data["validation"],
        start_entity,
        end_entity,
        start_tag,
        end_tag,
        natural=True,
        CUI_to_Syn=CUI_to_Syn_quaero,
        Syn_to_annotation=Syn_to_annotation_quaero,
        model_name=model_name,
    )
    test_source, test_target = process_bigbio_dataset(
        EMEA_data["test"],
        start_entity,
        end_entity,
        start_tag,
        end_tag,
        natural=True,
        CUI_to_Syn=CUI_to_Syn_quaero,
        model_name=model_name,
    )

    data_folder = f"data/preprocessed_dataset/{dataset_name}"
    with open(data_folder + f"/synth_train_source_{model_name}.pkl", "wb") as file:
        pickle.dump(synth_train_source, file, protocol=-1)
    with open(data_folder + f"/synth_train_target_{model_name}.pkl", "wb") as file:
        pickle.dump(synth_train_target, file, protocol=-1)
    with open(data_folder + f"/train_source_{model_name}.pkl", "wb") as file:
        pickle.dump(train_source, file, protocol=-1)
    with open(data_folder + f"/train_target_{model_name}.pkl", "wb") as file:
        pickle.dump(train_target, file, protocol=-1)
    with open(data_folder + f"/dev_source_{model_name}.pkl", "wb") as file:
        pickle.dump(dev_source, file, protocol=-1)
    with open(data_folder + f"/dev_target_{model_name}.pkl", "wb") as file:
        pickle.dump(dev_target, file, protocol=-1)
    with open(data_folder + f"/test_source_{model_name}.pkl", "wb") as file:
        pickle.dump(test_source, file, protocol=-1)
    with open(data_folder + f"/test_target_{model_name}.pkl", "wb") as file:
        pickle.dump(test_target, file, protocol=-1)

# Process MEDLINE dataset
dataset_name = "MEDLINE"
start_entity = "["
end_entity = "]"
start_tag = "{"
end_tag = "}"
model_names = [
    "mt5-large",
    "bart-large",
    "biobart-v2-large",
    "bart-genre",
    "mbart-large-50",
]
for model_name in model_names:
    synth_train_source, synth_train_target = process_bigbio_dataset(
        synthQUAERO,
        start_entity,
        end_entity,
        start_tag,
        end_tag,
        natural=True,
        CUI_to_Syn=CUI_to_Syn_quaero,
        Syn_to_annotation=Syn_to_annotation_quaero,
        model_name=model_name,
    )
    train_source, train_target = process_bigbio_dataset(
        MEDLINE_data["train"],
        start_entity,
        end_entity,
        start_tag,
        end_tag,
        natural=True,
        CUI_to_Syn=CUI_to_Syn_quaero,
        Syn_to_annotation=Syn_to_annotation_quaero,
        model_name=model_name,
    )
    dev_source, dev_target = process_bigbio_dataset(
        MEDLINE_data["validation"],
        start_entity,
        end_entity,
        start_tag,
        end_tag,
        natural=True,
        CUI_to_Syn=CUI_to_Syn_quaero,
        Syn_to_annotation=Syn_to_annotation_quaero,
        model_name=model_name,
    )
    test_source, test_target = process_bigbio_dataset(
        MEDLINE_data["test"],
        start_entity,
        end_entity,
        start_tag,
        end_tag,
        natural=True,
        CUI_to_Syn=CUI_to_Syn_quaero,
        model_name=model_name,
    )

    data_folder = f"data/preprocessed_dataset/{dataset_name}"
    with open(data_folder + f"/synth_train_source_{model_name}.pkl", "wb") as file:
        pickle.dump(synth_train_source, file, protocol=-1)
    with open(data_folder + f"/synth_train_target_{model_name}.pkl", "wb") as file:
        pickle.dump(synth_train_target, file, protocol=-1)
    with open(data_folder + f"/train_source_{model_name}.pkl", "wb") as file:
        pickle.dump(train_source, file, protocol=-1)
    with open(data_folder + f"/train_target_{model_name}.pkl", "wb") as file:
        pickle.dump(train_target, file, protocol=-1)
    with open(data_folder + f"/dev_source_{model_name}.pkl", "wb") as file:
        pickle.dump(dev_source, file, protocol=-1)
    with open(data_folder + f"/dev_target_{model_name}.pkl", "wb") as file:
        pickle.dump(dev_target, file, protocol=-1)
    with open(data_folder + f"/test_source_{model_name}.pkl", "wb") as file:
        pickle.dump(test_source, file, protocol=-1)
    with open(data_folder + f"/test_target_{model_name}.pkl", "wb") as file:
        pickle.dump(test_target, file, protocol=-1)
