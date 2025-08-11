import re

import nltk
import nltk.data
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def clean_natural(text):
    return (
        text.replace("\xa0", " ")
        .replace("{", "(")
        .replace("}", ")")
        .replace("[", "(")
        .replace("]", ")")
    )


def parse_text(
    data,
    start_entity,
    end_entity,
    start_tag,
    end_tag,
    nlp,
    CUI_to_Syn=None,
    Syn_to_annotation=None,
    natural=False,
    tokenizer=None,
    corrected_cui=None,
):
    targets_tokens = []
    sources_tokens = []

    for idx, passage in enumerate(data["passages"]):
        passage_text = passage["text"][0]
        start_offset_passage = passage["offsets"][0][0]
        end_offset_passage = passage["offsets"][0][1]
        if idx > 0:
            passage_text = "\n" + passage_text

        # Apply natural cleaning if needed
        if natural:
            passage_text = clean_natural(passage_text)

        # Tokenize the passage text
        source_tokens = tokenizer.tokenize(passage_text)  # type: ignore

        stack = []  # Stack to track currently open entities
        target_offsets = []
        source_offsets = []

        # Sort by (start_offset, -end_offset) to ensure inner entities come first
        sorted_entities = sorted(
            data["entities"], key=lambda e: (e["offsets"][0][0], -e["offsets"][0][1])
        )

        prev_end_offset = -1  # Track previous entity's end
        prev_start_offset = 100  # Track previous entity's start
        prev_normalized_id = ""  # Track previous entity's CUI

        # if not sorted_entities:
        #     print(f"⚠️ No entity found")
        #     continue

        for entity in sorted_entities:
            if not entity["normalized"]:
                print("⚠️ No CUI found")
                continue

            entity_id = entity["id"]
            start_offset = entity["offsets"][0][0]
            end_offset = entity["offsets"][0][1]

            if start_offset > end_offset_passage or start_offset < start_offset_passage:
                continue

            end_offset -= start_offset_passage
            start_offset -= start_offset_passage

            # Check for overlapping entities
            if (
                start_offset < prev_end_offset
                and start_offset > prev_start_offset
                and end_offset > prev_end_offset
            ):
                print(
                    f"⚠️ Warning: Overlapping but not nested entity detected: {entity['text'][0]}"
                )
                continue  # Ignore overlapping but not nested entity

            if start_offset == prev_start_offset and end_offset == prev_end_offset:
                print(f"⚠️ Warning: Duplicated entity detected: {entity['text'][0]}")
                continue  # Ignore Duplicated

            normalized_id = entity["normalized"][0]["db_id"]

            if corrected_cui and normalized_id in corrected_cui.keys():
                print(
                    f"✅ Convert: No synonym found for the code {normalized_id} : {entity['text'][0]} it has been converted to {corrected_cui[normalized_id]}"
                )
                normalized_id = corrected_cui[normalized_id]

            if (
                start_offset >= prev_start_offset
                and end_offset <= prev_end_offset
                and normalized_id == prev_normalized_id
            ):
                print(
                    f"⚠️ Warning: Overlapping and nested entities with same CUI: {entity['text'][0]}"
                )
                continue  # Ignore overlapping with same CUI

            possible_syns = CUI_to_Syn.get(normalized_id)  # type: ignore
            if possible_syns is not None:
                if Syn_to_annotation is not None:
                    lev_similarities = []
                    text = entity["text"][0]
                    for syn in possible_syns:
                        lev_similarities.append(nltk.edit_distance(text, syn))
                    best_syn = possible_syns[np.argmin(lev_similarities)]
                    annotation = best_syn
                    # annotation = Syn_to_annotation.filter(
                    #     (pl.col("CUI") == normalized_id) & (pl.col("Syn") == best_syn)
                    # )[0, 1]
                    if natural:
                        annotation = clean_natural(annotation)
                else:
                    annotation = normalized_id
            else:
                print(
                    f"⚠️ Warning: No synonym found for the code {normalized_id} : {entity['text'][0]}"
                )
                continue

            while stack:
                first_id, first_start, first_end, first_tag = stack.pop(0)
                target_offsets.append((
                    first_end,
                    first_id,
                    first_start,
                    f"{end_entity}{start_tag}{first_tag}{end_tag}",
                ))
                source_offsets.append((
                    first_end,
                    first_id,
                    first_start,
                    f"{end_entity}",
                ))

            if idx > 0:
                start_token = len(
                    tokenizer.tokenize(passage_text[: start_offset + 1].rstrip())  # type: ignore
                )
                end_token = len(
                    tokenizer.tokenize(passage_text[: end_offset + 1].rstrip())  # type: ignore
                )
            else:
                start_token = len(
                    tokenizer.tokenize(passage_text[:start_offset].rstrip())  # type: ignore
                )
                end_token = len(tokenizer.tokenize(passage_text[:end_offset].rstrip()))  # type: ignore
            if start_token == end_token:
                start_token -= 1
            target_offsets.append((start_token, entity_id, -1, f"{start_entity}"))
            source_offsets.append((start_token, entity_id, -1, f"{start_entity}"))
            stack.append((entity_id, start_token, end_token, annotation))

            prev_end_offset = max(prev_end_offset, end_offset)
            prev_start_offset = min(prev_start_offset, start_offset)
            prev_normalized_id = normalized_id

        # Close remaining open entities
        while stack:
            first_id, first_start, first_end, first_tag = stack.pop(0)
            target_offsets.append((
                first_end,
                first_id,
                first_start,
                f"{end_entity}{start_tag}{first_tag}{end_tag}",
            ))
            source_offsets.append((first_end, first_id, first_start, f"{end_entity}"))

        # Sort offsets in reverse order to avoid index shifting issues
        target_offsets.sort(
            reverse=True, key=lambda x: (x[0], -x[2], [-ord(c) for c in x[1]])
        )
        source_offsets.sort(
            reverse=True, key=lambda x: (x[0], -x[2], [-ord(c) for c in x[1]])
        )

        # Apply offsets to tokens instead of characters
        target_tokens = source_tokens.copy()
        for index, _, _, insert_text in target_offsets:
            for token in reversed(tokenizer.tokenize(insert_text)):  # type: ignore
                # Insert the separator or synonym token at the right place
                target_tokens.insert(index, token)

        for index, _, _, insert_text in source_offsets:
            for token in reversed(tokenizer.tokenize(insert_text)):  # type: ignore
                # Insert the separator or synonym token at the right place
                source_tokens.insert(index, token)
        # Detokenize back to text
        targets_tokens.append(target_tokens)
        sources_tokens.append(source_tokens)

    # Join all processed passages into one text
    sources_tokens = tokenizer.convert_tokens_to_string([  # type: ignore
        token for source_tokens in sources_tokens for token in source_tokens
    ])
    targets_tokens = tokenizer.convert_tokens_to_string([  # type: ignore
        token for target_tokens in targets_tokens for token in target_tokens
    ])
    return sources_tokens, targets_tokens


def process_bigbio_dataset(
    bigbio_dataset,
    start_entity,
    end_entity,
    start_tag,
    end_tag,
    CUI_to_Syn=None,
    Syn_to_annotation=None,
    natural=False,
    model_name=None,
    corrected_cui=None,
):
    # Load sentenczier
    nlp = nltk.data.load("tokenizers/punkt/english.pickle")

    # Load tokenizer
    root_path = "/models"
    model_path = root_path + "/" + model_name  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=False)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [start_entity, end_entity, start_tag, end_tag]},
        replace_additional_special_tokens=False,
    )
    target_data = []
    source_data = []
    for page in tqdm(bigbio_dataset, total=len(bigbio_dataset)):
        source_texts, target_texts = parse_text(
            page,
            start_entity,
            end_entity,
            start_tag,
            end_tag,
            nlp,
            CUI_to_Syn,
            Syn_to_annotation,
            natural,
            tokenizer,
            corrected_cui,
        )
        target_data.append(target_texts)
        source_data.append(source_texts)
    return source_data, target_data


def load_data(source_data, target_data, nlp, tokenizer, max_length=512):
    """
    Load and preprocess source and target data, ensuring that the target text is split into passages
    while maintaining token limit constraints and balanced entity markers.

    Args:
        source_data (list of str): List of source texts.
        target_data (list of str): List of target texts with annotated entities.
        nlp (nltk.tokenize): NLTK tokenizer with a `.tokenize()` method (e.g., nltk.sent_tokenize).
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for computing token lengths.
        max_length (int, optional): Maximum allowed token length for each target passage. Default is 512.

    Returns:
        dict: A dictionary containing:
            - "source" (list of str): Source passages with entity annotations removed.
            - "target" (list of str): Target passages containing entity annotations.
    """
    data = {"source": [], "target": []}
    for _, target in zip(source_data, target_data):
        tokens = 0
        target_passage = ""
        target_doc = nlp.tokenize(target)
        for sent in target_doc:
            sent_tokens = len(tokenizer.encode(sent))
            tokens += sent_tokens
            if (
                tokens < max_length
                or (target_passage.count("<s_e>") != target_passage.count("<e_e>"))
                or (target_passage.count("<s_m>") != target_passage.count("<e_e>"))
            ):
                target_passage += sent + " "
            else:
                target_passage = target_passage[:-1]
                data["target"].append(target_passage)
                source_passage = re.sub(r"<s_m>\s|\s<s_e>.*?<e_e>", "", target_passage)
                data["source"].append(source_passage)
                target_passage = sent + " "
                tokens = sent_tokens
        if tokens > 20:
            target_passage = target_passage[:-1]
            data["target"].append(target_passage)
            source_passage = re.sub(r"<s_m>\s|\s<s_e>.*?<e_e>", "", target_passage)
            data["source"].append(source_passage)
        elif tokens < 20:
            target_passage = data["target"][-1] + " " + target_passage[:-1]
            data["target"][-1] = target_passage
            source_passage = re.sub(r"<s_m>\s|\s<s_e>.*?<e_e>", "", target_passage)
            data["source"][-1] = source_passage

    return data
