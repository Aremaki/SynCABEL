import argparse
import os
import pickle

import nltk.data
import polars as pl
import torch
from tqdm import tqdm
from transformers import GenerationConfig  # type: ignore

from syncabel.guided_inference import get_prefix_allowed_tokens_fn
from syncabel.models import MT5_GENRE, Bart_GENRE, MBart_GENRE
from syncabel.trie import Trie

# Load the English Punkt tokenizer once
nlp = nltk.data.load("tokenizers/punkt/english.pickle")


def custom_sentence_tokenize(text):
    # Split the text on newline
    parts = text.splitlines(keepends=True)
    sentences = []
    for part in parts:
        # Tokenize each part separately with Punkt
        sents = nlp.tokenize(part)  # type: ignore
        sentences.extend(sents)
    # Filter out any empty sentences (if any)
    return [s for s in sentences if s]


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def get_pointer_end(target_passage, source, start_entity, end_entity):
    i = 0
    j = 0
    len_s_entity = len(start_entity)
    len_e_entity = len(end_entity)
    while i < len(target_passage):
        if target_passage[i : i + len_s_entity] == start_entity:
            i += len_s_entity
            while target_passage[i : i + len_e_entity] != end_entity:
                i += 1
            i += len_e_entity
        elif target_passage[i] == source[j]:
            i += 1
            j += 1
        else:
            print(target_passage[:])
            print(source[:])
            raise RuntimeError("Source and Target misaligned")
    return j


def get_pointer_end_ner(target_passage, target_ner, start_entity, end_entity):
    i = 0
    j = 0
    len_s_entity = len(start_entity)
    len_e_entity = len(end_entity)
    while i < len(target_passage):
        if target_passage[i : i + len_s_entity] == start_entity:
            i += len_s_entity
            while target_passage[i : i + len_e_entity] != end_entity:
                i += 1
            i += len_e_entity
        elif target_ner[j : j + len_s_entity] == start_entity:
            j += len_s_entity
            while target_ner[j : j + len_e_entity] != end_entity:
                j += 1
            j += len_e_entity
        elif target_passage[i] == target_ner[j]:
            i += 1
            j += 1
        else:
            print(target_passage[:])
            print(target_ner[:])
            raise RuntimeError("target_ner and Target misaligned")
    return j


def load_data(
    source_data,
    target_data,
    target_data_ner,
    tokenizer,
    start_mention,
    end_mention,
    start_entity,
    end_entity,
    max_length=512,
):
    data = {"source": [], "target": [], "target_ner": []}
    for source, target, target_ner in tqdm(
        zip(source_data, target_data, target_data_ner),
        total=len(source_data),
        desc="Processing Data",
    ):
        tokens = 0
        start_source = 0
        end_source = 0
        start_target = 0
        end_target = 0
        start_target_ner = 0
        end_target_ner = 0
        target_sentences = custom_sentence_tokenize(target)
        for sent in target_sentences:
            sent_end = 0
            sent_tokens = len(tokenizer.encode(sent))
            tokens += sent_tokens
            if (
                start_target != end_target
                and tokens > max_length
                and target[start_target:end_target].count(start_mention)
                == target[start_target:end_target].count(end_mention)
                and target[start_target:end_target].count(start_entity)
                == target[start_target:end_target].count(end_entity)
            ):
                tokens = sent_tokens
                data["target"].append(target[start_target:end_target].rstrip())
                end_source = start_source + get_pointer_end(
                    target[start_target:end_target],
                    source[start_source:],
                    start_entity,
                    end_entity,
                )
                data["source"].append(source[start_source:end_source].rstrip())
                end_target_ner = start_target_ner + get_pointer_end_ner(
                    target[start_target:end_target],
                    target_ner[start_target_ner:],
                    start_entity,
                    end_entity,
                )
                data["target_ner"].append(
                    target_ner[start_target_ner:end_target_ner].rstrip()
                )
                start_target = end_target
                start_source = end_source
                start_target_ner = end_target_ner
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
            data["target_ner"].append(target_ner[start_target_ner:].rstrip())
        elif data["target"]:
            data["target"][-1] = (data["target"][-1] + target[start_target:]).rstrip()
            data["source"][-1] = (data["source"][-1] + source[start_source:]).rstrip()
            data["target_ner"][-1] = (
                data["target_ner"][-1] + target_ner[start_target_ner:]
            ).rstrip()

    return data


def main(model_name, model_path, max_length, num_beams, best):
    # Set device
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    start_mention, end_mention, start_entity, end_entity = "[", "]", "{", "}"
    root_path = f"/models/{model_path}_{max_length}/{model_name}"
    if best:
        full_path = root_path + "/model_best"
    else:
        full_path = root_path + "/model_last"

    if "mt5" in model_name:
        model = MT5_GENRE.from_pretrained(full_path).eval().to(device)  # type: ignore
        model.generation_config = GenerationConfig(
            decoder_start_token_id=0,
            eos_token_id=1,
            forced_eos_token_id=1,
            pad_token_id=0,
        )
    elif "mbart" in model_name:
        model = MBart_GENRE.from_pretrained(full_path).eval().to(device)  # type: ignore
        model.generation_config = GenerationConfig(
            bos_token_id=0,
            decoder_start_token_id=2,
            eos_token_id=2,
            forced_eos_token_id=2,
            pad_token_id=1,
        )
    else:
        model = Bart_GENRE.from_pretrained(full_path).eval().to(device)  # type: ignore
        model.generation_config = GenerationConfig(
            bos_token_id=0,
            decoder_start_token_id=2,
            eos_token_id=2,
            forced_eos_token_id=2,
            pad_token_id=1,
        )

    # Load data
    data_folder = "/data/preprocessed_dataset"
    test_source_data = load_pickle(f"{data_folder}/test_source_{model_name}.pkl")
    test_target_data = load_pickle(f"{data_folder}/test_target_{model_name}.pkl")

    ner_data_folder = "/data/preprocessed_dataset_ner"
    test_target_data_ner = load_pickle(
        f"{ner_data_folder}/test_target_{model_name}.pkl"
    )

    # Load and preprocess data
    tokenizer = model.tokenizer
    test_data = load_data(
        test_source_data,
        test_target_data,
        test_target_data_ner,
        tokenizer,
        start_mention,
        end_mention,
        start_entity,
        end_entity,
        max_length=max_length,
    )

    # Load candidate Trie
    trie_path = f"{data_folder}/trie_legal_tokens_typed_{model_name}.pkl"
    if os.path.exists(trie_path):  # Check if the file exists
        with open(trie_path, "rb") as file:
            trie_legal_tokens = pickle.load(file)
    else:
        # Compute candidate Trie
        start_idx = 1 if "bart" in model_name else 0
        if "medmentions" in data_folder:
            group_cat = "SEM_NAME_MM"
            legal_umls_token = pl.read_parquet(
                os.environ["WORK"]
                + "/GENRE/data/legal_umls_token_2017_short_syn_all_main.parquet"
            )
        else:
            group_cat = "CATEGORY"
            legal_umls_token = pl.read_parquet(
                os.environ["WORK"]
                + "/GENRE/data/quaero_legal_umls_token_2014_short_syn_all_main.parquet"
            )
        legal_umls_token = legal_umls_token.with_columns(
            pl.col("Entity")
            .str.replace_all("\xa0", " ", literal=True)
            .str.replace_all("{", "(", literal=True)
            .str.replace_all("}", ")", literal=True)
            .str.replace_all("[", "(", literal=True)
            .str.replace_all("]", ")", literal=True)
        )
        trie_legal_tokens = {}
        for category in legal_umls_token[group_cat].unique().to_list():
            print(f"processing {category}")
            cat_legal_umls_token = legal_umls_token.filter(
                pl.col(group_cat) == category
            )
            trie_legal_tokens[category] = Trie([
                model.tokenizer.encode(f"{start_entity}{entity}{end_entity}")[  # type: ignore
                    start_idx:-1
                ]
                for entity in cat_legal_umls_token["Entity"].to_list()
            ])

        # Save it
        with open(trie_path, "wb") as file:
            pickle.dump(trie_legal_tokens, file, protocol=-1)

    # Perform inference with constraint
    output_sentences = []
    for target_ner, source in tqdm(
        zip(test_data["target_ner"], test_data["source"]), desc="Processing Test Data"
    ):
        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
            model,
            [target_ner],
            candidates_trie=trie_legal_tokens,
        )
        output_sentence = model.sample(
            [source],
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_return_sequences=1,
        )
        output_sentences.append(output_sentence)

    # Save results
    output_path = f"{full_path}/pred_test_constraint_{num_beams}_beams_typed.pkl"
    with open(output_path, "wb") as file:
        pickle.dump(output_sentences, file, protocol=-1)

    print("Inference completed and results saved.")


if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="A script for inference seq2seq model")
    parser.add_argument("--model-name", type=str, required=True, help="The model name")
    parser.add_argument("--model-path", type=str, required=True, help="The model name")
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="The max number of token per sequence",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="The number of beams",
    )
    parser.add_argument(
        "--best", default=False, action="store_true", help="Use best if True else last"
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    # Pass the parsed argument to the main function
    main(args.model_name, args.model_path, args.max_length, args.num_beams, args.best)
