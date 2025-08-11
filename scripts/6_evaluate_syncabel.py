import pickle
from collections import defaultdict

import nltk.data
import pandas as pd
import polars as pl
import torch
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer

from syncabel.utils import get_entity_spans_finalize  # type: ignore

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
start_entity, start_tag, end_tag = "[", "]{", "}"


# Define a function to compute precision, recall, and F1-score
def _compute_metrics(df, filter_condition=None):
    if filter_condition is not None:
        df = df.filter(filter_condition)

    tp = df["success"].sum()
    fn = df["fail"].sum()

    def recall(tp, fn):
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    recall = recall(tp, fn)  # type: ignore

    return {
        "Accuracy": recall,
        "Success": int(tp),
        "Fail": int(fn),
        "Support": int(fn) + int(tp),
    }


def compute_metrics(result):
    # Compute metrics for different conditions
    metrics_overall = _compute_metrics(result)
    metrics_unseen_mention_gold = _compute_metrics(result, ~result["seen_mention_gold"])
    metrics_seen_mention_gold = _compute_metrics(result, result["seen_mention_gold"])
    metrics_seen_concept = _compute_metrics(result, result["seen_concept_gold"])
    metrics_unseen_concept = _compute_metrics(result, ~result["seen_concept_gold"])
    metrics_top_100_concept = _compute_metrics(result, result["top_100_concept_gold"])

    # Store results
    metrics_results = {
        "Unseen Mention": metrics_unseen_mention_gold,
        "Seen Mention": metrics_seen_mention_gold,
        "Unseen Concept": metrics_unseen_concept,
        "Seen Concept": metrics_seen_concept,
        "Top 100 concept": metrics_top_100_concept,
        "Overall": metrics_overall,
    }
    # Convert results into a DataFrame
    return pl.DataFrame(
        pd.DataFrame.from_dict(metrics_results, orient="index").reset_index(
            names="Type"
        )
    )


def strict_evaluation(pred_df: pl.DataFrame, gold_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute strict precision, recall, and F1 for each entity type and overall.

    Args:
        pred_df: DataFrame with columns ['doc_id', 'start_offset', 'length', 'pred_type', 'mention']
        gold_df: DataFrame with columns ['doc_id', 'start_offset', 'length', 'gold_type', 'mention_gold']

    Returns:
        A Polars DataFrame with evaluation metrics per type and overall
    """
    # Create string keys for comparison (hashable)
    pred_keys = pred_df.select([
        pl.format(
            "{}|{}|{}|{}",
            pl.col("doc_id"),
            pl.col("start_offset"),
            pl.col("length"),
            pl.col("CUI"),
        ).alias("key"),
        pl.col("semtype").alias("type"),
    ])

    gold_keys = gold_df.select([
        pl.format(
            "{}|{}|{}|{}",
            pl.col("doc_id"),
            pl.col("start_offset"),
            pl.col("length"),
            pl.col("CUI_gold"),
        ).alias("key"),
        pl.col("semtype_gold").alias("type"),
    ])

    # Convert to Python sets for fast lookups
    pred_set = set(pred_keys["key"].to_list())
    gold_set = set(gold_keys["key"].to_list())

    # Get type information
    pred_types = dict(zip(pred_keys["key"].to_list(), pred_keys["type"].to_list()))
    gold_types = dict(zip(gold_keys["key"].to_list(), gold_keys["type"].to_list()))

    # Initialize counters
    success = defaultdict(int)
    fails = defaultdict(int)

    # Count success
    for pred_key in pred_set:
        entity_type = pred_types[pred_key]
        if pred_key in gold_set:
            success[entity_type] += 1

    # Count fails (entities in gold but not in pred)
    for gold_key in gold_set:
        entity_type = gold_types[gold_key]
        if gold_key not in pred_set:
            fails[entity_type] += 1

    # Collect all entity types
    all_types = set(success.keys()).union(set(fails.keys()))

    # Calculate metrics per entity type
    metrics = []
    for entity_type in sorted(all_types):
        tp = success.get(entity_type, 0)
        fn = fails.get(entity_type, 0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        metrics.append({
            "Type": entity_type,
            "Accuracy": recall,
            "Success": tp,
            "Fail": fn,
            "Support": tp + fn,  # Number of gold entities for this type
        })

    return pl.DataFrame(metrics)


def aggregate_result(guess_entities, gold_entities, train_entities):
    if guess_entities:
        pred_df = pl.DataFrame(
            guess_entities,
            strict=False,
            orient="row",
            schema=[
                "doc_id",
                "start_offset",
                "length",
                "CUI",
                "mention",
                "entity",
                "semtype",
            ],
        )
    else:
        pred_df = pl.DataFrame({
            "doc_id": pl.Series([], dtype=pl.Utf8),
            "start_offset": pl.Series([], dtype=pl.Int64),
            "length": pl.Series([], dtype=pl.Int64),
            "mention": pl.Series([], dtype=pl.Utf8),
            "CUI": pl.Series([], dtype=pl.Utf8),
            "entity": pl.Series([], dtype=pl.Utf8),
            "semtype": pl.Series([], dtype=pl.Utf8),
        })
    gold_df = pl.DataFrame(
        gold_entities,
        strict=False,
        orient="row",
        schema=[
            "doc_id",
            "start_offset",
            "length",
            "CUI_gold",
            "mention_gold",
            "entity_gold",
            "semtype_gold",
        ],
    )
    train_df = pl.DataFrame(
        train_entities,
        strict=False,
        orient="row",
        schema=[
            "doc_id",
            "start_offset",
            "length",
            "train_CUI",
            "train_mention",
            "train_entity",
            "train_semtype",
        ],
    )
    seen_mention = train_df["train_mention"].unique()
    train_df["train_entity"].unique()
    seen_cui = train_df["train_CUI"].unique()
    top_100_cui = (
        train_df.group_by("train_CUI")
        .len()
        .sort("len", descending=True)["train_CUI"][:100]
    )
    result = (
        pred_df.join(
            gold_df, on=["doc_id", "start_offset", "length"], how="full", coalesce=True
        )
        .filter(pl.col("mention_gold").is_not_null())
        .unique()
    )
    all_results = result.with_columns(
        success=(pl.col("CUI").eq_missing(pl.col("CUI_gold"))),
        fail=(pl.col("CUI").ne_missing(pl.col("CUI_gold"))),
        multi_word_mention_gold=pl.col("mention_gold").str.split(" ").list.len() > 1,
        multi_word_mention=pl.col("mention").str.split(" ").list.len() > 1,
        direct_match_gold=pl.col("mention_gold").str.to_lowercase()
        == pl.col("entity_gold").str.to_lowercase(),
        direct_match=pl.col("mention").str.to_lowercase()
        == pl.col("entity").str.to_lowercase(),
        seen_mention_gold=pl.col("mention_gold").is_in(seen_mention),
        seen_mention=pl.col("mention").is_in(seen_mention),
        seen_concept_gold=pl.col("CUI_gold").is_in(seen_cui),
        seen_concept=pl.col("CUI").is_in(seen_cui),
        top_100_concept_gold=pl.col("CUI_gold").is_in(top_100_cui),
        top_100_concept=pl.col("CUI").is_in(top_100_cui),
    )
    return pred_df, gold_df, train_df, all_results


def get_entity_spans(
    sources,
    labels,
    start_entity,
    start_tag,
    end_tag,
    entity_to_CUI=None,
    CUI_to_Syn=None,
    CUI_to_type=None,
):
    result = get_entity_spans_finalize(
        sources, labels, start_entity, start_tag, end_tag
    )
    result = [
        (f"id_{k}",) + tuple(x) for k, e in zip(range(len(result)), result) for x in e
    ]
    enriched_result = []
    if entity_to_CUI:
        for id_doc, start_offset, length, entity_syn, mention in result:
            try:
                cui = entity_to_CUI[entity_syn.lstrip()][0]
                if CUI_to_type:
                    cui_type = CUI_to_type[cui][0]
                    enriched_result.append((
                        id_doc,
                        start_offset,
                        length,
                        cui,
                        mention,
                        entity_syn.lstrip(),
                        cui_type,
                    ))
                else:
                    enriched_result.append((
                        id_doc,
                        start_offset,
                        length,
                        cui,
                        mention,
                        entity_syn.lstrip(),
                    ))
            except:  # noqa: E722
                pass
    if CUI_to_Syn:
        for id_doc, start_offset, length, cui, mention in result:
            syns = CUI_to_Syn[cui.lstrip()]
            if mention in syns:
                syn = mention
            else:
                syn = syns[0]
            if CUI_to_type:
                cui_type = CUI_to_type[cui.lstrip()][0]
                enriched_result.append((
                    id_doc,
                    start_offset,
                    length,
                    cui.lstrip(),
                    mention,
                    syn,
                    cui_type,
                ))
            else:
                enriched_result.append((
                    id_doc,
                    start_offset,
                    length,
                    cui.lstrip(),
                    mention,
                    syn,
                ))
    return enriched_result


def custom_sentence_tokenize(text, nlp):
    # Split the text on newline
    parts = text.splitlines(keepends=True)
    sentences = []
    for part in parts:
        # Tokenize each part separately with Punkt
        sents = nlp.tokenize(part)
        sentences.extend(sents)
    # Filter out any empty sentences (if any)
    return [s for s in sentences if s]


# Preprocessing function
def get_pointer_end(target_passage, source, start_entity, start_tag, end_tag):
    i = 0
    j = 0
    len_s_ent = len(start_entity)
    len_s_tag = len(start_tag)
    len_e_tag = len(end_tag)
    while i < len(target_passage):
        if target_passage[i : i + len_s_ent] == start_entity:
            i += len_s_ent
        elif target_passage[i : i + len_s_tag] == start_tag:
            i += len_s_tag
            while target_passage[i : i + len_e_tag] != end_tag:
                i += 1
            i += len_e_tag
        elif target_passage[i] == source[j]:
            i += 1
            j += 1
        elif target_passage[i] in [" ", "\n"]:
            i += 1
        elif source[i] in [" ", "\n"]:
            j += 1
        else:
            print(target_passage[:])
            print(source[:])
            raise RuntimeError("Source and Target misaligned")
    return j


# Preprocessing function
def load_data(
    source_data,
    target_data,
    nlp,
    tokenizer,
    start_entity,
    start_tag,
    end_tag,
    max_length=512,
):
    data = {"source": [], "target": []}
    for source, target in tqdm(
        zip(source_data, target_data), total=len(source_data), desc="Processing Data"
    ):
        tokens = 0
        start_source = 0
        end_source = 0
        start_target = 0
        end_target = 0
        target_sentences = custom_sentence_tokenize(target, nlp)
        for sent in target_sentences:
            sent_end = 0
            sent_tokens = len(tokenizer.encode(sent))
            tokens += sent_tokens
            if (
                start_target != end_target
                and tokens > max_length
                and target[start_target:end_target].count(start_entity)
                == target[start_target:end_target].count(end_tag)
                and target[start_target:end_target].count(start_tag)
                == target[start_target:end_target].count(end_tag)
            ):
                tokens = sent_tokens
                data["target"].append(target[start_target:end_target].rstrip())
                end_source = start_source + get_pointer_end(
                    target[start_target:end_target],
                    source[start_source:],
                    start_entity,
                    start_tag,
                    end_tag,
                )
                data["source"].append(source[start_source:end_source].rstrip())
                start_target = end_target
                start_source = end_source
            while sent_end < len(sent):
                if target[end_target] == sent[sent_end]:
                    end_target += 1
                    sent_end += 1
                elif sent[sent_end] in [" ", "\n"]:
                    sent_end += 1
                elif target[end_target] in [" ", "\n"]:
                    end_target += 1
        if tokens > 50:
            data["target"].append(target[start_target:].rstrip())
            data["source"].append(source[start_source:].rstrip())
        elif data["target"]:
            data["target"][-1] = (data["target"][-1] + target[start_target:]).rstrip()
            data["source"][-1] = (data["source"][-1] + source[start_source:]).rstrip()

    return data


def flatten_result_table(df: pl.DataFrame, metadata: dict) -> pl.DataFrame:
    # Add metadata columns to each row of the result table
    meta_df = pl.DataFrame([{**metadata, **row} for row in df.to_dicts()])
    return meta_df


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


model_names = [
    "mbart-large-50",
    "biobart-v2-large",
    "mt5-large",
    "bart-large",
    "bart-genre",
]

model_paths = {
    "MM": [
        "main",
        "main_augmented_v3_llm70B",
        "main_wiki_augmented_v3_llm70B",
        "main_wiki_augmented_llm70B",
        "main_augmented_all_llm70B",
    ],
    "medline": [
        "main",
        "main_augmented_v3_llm70B",
        "main_wiki_augmented_v3_llm70B",
        "main_wiki_augmented_llm70B",
        "main_augmented_all_llm70B",
    ],
    "emea": [
        "main",
        "main_augmented_v3_llm70B",
        "main_wiki_augmented_v3_llm70B",
        "main_wiki_augmented_v2_llm70B",
        "main_augmented_all_llm70B",
    ],
}
dataset_paths = {
    "medline": "quaero_medline_entity_main_2014",
    "emea": "quaero_emea_entity_main_2014",
    "MM": "medmentions_st21pv_2017_tokenized_entity_main",
}

max_lengths = {
    "medline": [256],
    "emea": [64, 128, 256],
    "MM": [64, 128, 256],
}

quaero_umls_df = pl.read_parquet(
    "../data/quaero_legal_umls_token_2014_short_syn_all_main.parquet"
)
quaero_umls_df = quaero_umls_df.with_columns(
    pl.col("Entity")
    .str.replace_all("\xa0", " ", literal=True)
    .str.replace_all("{", "(", literal=True)
    .str.replace_all("}", ")", literal=True)
    .str.replace_all("[", "(", literal=True)
    .str.replace_all("]", ")", literal=True)
)
quaero_entity_to_CUI = dict(
    quaero_umls_df.group_by("Entity").agg([pl.col("CUI").unique()]).iter_rows()
)
quaero_CUI_to_Syn = dict(
    quaero_umls_df.group_by("CUI").agg([pl.col("Entity").unique()]).iter_rows()
)
quaero_CUI_to_type = dict(
    quaero_umls_df.group_by("CUI").agg([pl.col("CATEGORY").unique()]).iter_rows()
)

MM_umls_df = pl.read_parquet("../data/legal_umls_token_2017_short_syn_all_main.parquet")
MM_umls_df = MM_umls_df.with_columns(
    pl.col("Entity")
    .str.replace_all("\xa0", " ", literal=True)
    .str.replace_all("{", "(", literal=True)
    .str.replace_all("}", ")", literal=True)
    .str.replace_all("[", "(", literal=True)
    .str.replace_all("]", ")", literal=True)
)
MM_entity_to_CUI = dict(
    MM_umls_df.group_by("Entity").agg([pl.col("CUI").unique()]).iter_rows()
)
MM_CUI_to_Syn = dict(
    MM_umls_df.group_by("CUI").agg([pl.col("Entity").unique()]).iter_rows()
)
MM_CUI_to_type = dict(
    MM_umls_df.group_by("CUI").agg([pl.col("CATEGORY").unique()]).iter_rows()
)

umls_dfs = {
    "medline": (quaero_entity_to_CUI, quaero_CUI_to_Syn, quaero_CUI_to_type),
    "emea": (quaero_entity_to_CUI, quaero_CUI_to_Syn, quaero_CUI_to_type),
    "MM": (MM_entity_to_CUI, MM_CUI_to_Syn, MM_CUI_to_type),
}

all_results_dfs = []
for model_name in model_names:
    print("#" * 10 + f"  {model_name}  " + "#" * 10)
    for dataset in model_paths.keys():
        if dataset == "MM":
            nlp = nltk.data.load("tokenizers/punkt/english.pickle")
        else:
            nlp = nltk.data.load("tokenizers/punkt/french.pickle")
        print("#" * 10 + f"  {dataset}  " + "#" * 10)
        # Load data
        entity_to_CUI, CUI_to_Syn, CUI_to_type = umls_dfs[dataset]
        data_folder = f"/data/{dataset_paths[dataset]}"
        train_source_data = load_pickle(f"{data_folder}/train_source_{model_name}.pkl")
        train_target_data = load_pickle(f"{data_folder}/train_target_{model_name}.pkl")
        dev_source_data = load_pickle(f"{data_folder}/dev_source_{model_name}.pkl")
        dev_target_data = load_pickle(f"{data_folder}/dev_target_{model_name}.pkl")
        test_source_data = load_pickle(f"{data_folder}/test_source_{model_name}.pkl")
        test_target_data = load_pickle(f"{data_folder}/test_target_{model_name}.pkl")
        for model_path in model_paths[dataset]:
            print("#" * 10 + f"  {model_path}  " + "#" * 10)
            if len(model_path.split("_")) >= 2:
                if model_path.split("_")[2] == "all":
                    data_augmentation = "LLM augmented complete"
                elif model_path.split("_")[1] == "augmented":
                    data_augmentation = "LLM augmented ideal"
                elif model_path.split("_")[1] == "wiki":
                    if model_path.split("_")[3] == "v3":
                        data_augmentation = "LLM and Wikipedia augmented V2"
                    else:
                        data_augmentation = "LLM and Wikipedia augmented"
                else:
                    data_augmentation = "No augmentation"
            else:
                data_augmentation = "No augmentation"
            print(data_augmentation)
            for max_length in max_lengths[dataset]:
                print("#" * 10 + f"  {max_length}  " + "#" * 10)
                root_path = (
                    f"/models/{dataset}_NED_{model_path}_{max_length}/{model_name}"
                )
                for training_stop in ["/model_last", "/model_best"]:
                    full_path = root_path + training_stop
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            full_path, add_prefix_space=False
                        )
                    except:  # noqa: E722
                        continue
                    print("#" * 10 + f"  {training_stop}  " + "#" * 10)
                    # Load result
                    test_data = load_data(
                        test_source_data,
                        test_target_data,
                        nlp,
                        tokenizer,
                        start_entity,
                        start_tag,
                        end_tag,
                        max_length=max_length,
                    )
                    train_data = load_data(
                        train_source_data,
                        train_target_data,
                        nlp,
                        tokenizer,
                        start_entity,
                        start_tag,
                        end_tag,
                        max_length=max_length,
                    )
                    validation_data = load_data(
                        dev_source_data,
                        dev_target_data,
                        nlp,
                        tokenizer,
                        start_entity,
                        start_tag,
                        end_tag,
                        max_length=max_length,
                    )
                    # Create dataset
                    validation_dataset = Dataset.from_dict(validation_data)
                    split = int(len(validation_dataset) * 0.5)
                    train_data = concatenate_datasets([
                        Dataset.from_dict(train_data),
                        validation_dataset.select(range(split)),
                    ]).to_dict()

                    for n_beams in [1, 2, 5, 10]:
                        for constraint in [
                            "No Constraint",
                            "Constraint",
                            "Constraint Typed",
                        ]:
                            if constraint == "No Constraint":
                                pred_test_path = f"{full_path}/pred_test_no_constraint_{n_beams}_beams.pkl"
                            elif constraint == "Constraint":
                                pred_test_path = f"{full_path}/pred_test_constraint_{n_beams}_beams.pkl"
                            else:
                                pred_test_path = f"{full_path}/pred_test_constraint_{n_beams}_beams_typed.pkl"
                            try:
                                with open(pred_test_path, "rb") as file:
                                    pred_test = pickle.load(file)
                                if len(pred_test[0]) == 1:
                                    print("#" * 10 + "  WARNING  " + "#" * 10)
                                    print(pred_test_path)
                                    print(
                                        "#" * 10 + "  NOT INFERED PROPRELY  " + "#" * 10
                                    )
                            except:  # noqa: E722
                                print("#" * 10 + "  WARNING  " + "#" * 10)
                                print("#" * 10 + "  NOT AVAILABLE  " + "#" * 10)
                                continue

                            print(
                                "#" * 10
                                + f"  {constraint} {n_beams} BEAMS  "
                                + "#" * 10
                            )
                            guess_entities = get_entity_spans(
                                test_data["source"],
                                pred_test,
                                "[",
                                "]{",
                                "}",
                                entity_to_CUI,
                                CUI_to_Syn=None,
                                CUI_to_type=CUI_to_type,
                            )
                            gold_entities = get_entity_spans(
                                test_data["source"],
                                test_data["target"],
                                "[",
                                "]{",
                                "}",
                                entity_to_CUI=None,
                                CUI_to_Syn=CUI_to_Syn,
                                CUI_to_type=CUI_to_type,
                            )
                            train_entities = get_entity_spans(
                                train_data["source"],  # type: ignore
                                train_data["target"],  # type: ignore
                                "[",
                                "]{",
                                "}",
                                entity_to_CUI,
                                CUI_to_Syn=None,
                                CUI_to_type=CUI_to_type,
                            )
                            pred_df, gold_df, train_df, all_results = aggregate_result(
                                guess_entities, gold_entities, train_entities
                            )
                            results_df = strict_evaluation(pred_df, gold_df)
                            final_df = compute_metrics(all_results)
                            final_all_df = pl.concat([results_df, final_df])
                            metadata = {
                                "model": model_name,
                                "dataset": dataset,
                                "max_length": max_length,
                                "n_beams": n_beams,
                                "constrained_inference": constraint,
                                "data_augmentation": data_augmentation,
                                "training_stop": training_stop.split("_")[-1],
                            }
                            all_results_dfs.append(
                                flatten_result_table(final_all_df, metadata)
                            )
                            with pl.Config(tbl_rows=100):
                                print(final_all_df)


all_results_df = pl.concat(all_results_dfs).to_pandas()
all_results_df.to_pickle("data/results/gen_results.pkl")
