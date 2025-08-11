# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Dict, List

import torch

from syncabel.trie import Trie


def remove_types_and_store(text):
    # Find all bracketed content
    sem_types = re.findall(r"\{(.*?)\}", text)
    sem_types = [sem_type.strip() for sem_type in sem_types]
    # Remove bracketed content along with the brackets
    cleaned_text = re.sub(r"\{.*?\}", "", text)

    return cleaned_text, sem_types


def get_prefix_allowed_tokens_fn(
    model,
    sentences: List[str],
    start_mention_token="[",
    end_mention_token="]",
    start_entity_token="{",
    end_entity_token="}",
    candidates_trie: Dict[str, Trie] = None,  # type: ignore
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: model.tokenizer.encode(x),
        lambda x: model.tokenizer.decode(torch.tensor(x)),
        model.tokenizer.bos_token_id,
        model.tokenizer.pad_token_id,
        model.tokenizer.eos_token_id,
        sentences,
        start_mention_token,
        end_mention_token,
        start_entity_token,
        end_entity_token,
        candidates_trie,
        model.name_or_path,
    )


def _get_end_to_end_prefix_allowed_tokens_fn(
    encode_fn,
    decode_fn,
    bos_token_id,
    pad_token_id,
    eos_token_id,
    sentences: List[str],
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    candidates_trie: Dict[str, Trie] = None,  # type: ignore
    model_name: str = "",
):
    is_bart = "bart" in model_name
    codes = {
        n: encode_fn(f" {c}")[2 if is_bart else 0]
        for n, c in zip(
            (
                "start_mention_token",
                "end_mention_token",
                "start_entity_token",
                "end_entity_token",
            ),
            (
                start_mention_token,
                end_mention_token,
                start_entity_token,
                end_entity_token,
            ),
        )
    }
    codes["EOS"] = eos_token_id
    codes["BOS"] = bos_token_id
    codes["PAD"] = pad_token_id

    sent_origs = []
    sent_sem_types = []
    for sent in sentences:
        sent_source, sem_types = remove_types_and_store(sent)
        sent_sem_types.append(sem_types)
        if is_bart:
            sent_origs.append([2] + encode_fn(sent_source))
        else:
            sent_origs.append([0] + encode_fn(sent_source))

    def prefix_allowed_tokens_fn(batch_id, sent):
        sent = sent.tolist()
        status, entity_tok_count = get_status(sent, codes)
        sent_orig = sent_origs[batch_id]
        if sent[-1] == codes["end_mention_token"]:
            return [codes["start_entity_token"]]

        trie_out = []
        if status == "e":
            trie_out = get_trie_entity(sent, sent_orig, entity_tok_count)
            if trie_out == codes["EOS"]:
                trie_out = get_trie_outside(sent, sent_orig)
        elif status == "o":
            trie_out = get_trie_outside(sent, sent_orig)
        else:
            raise RuntimeError

        return trie_out

    def get_status(sent, codes):
        start_entity_tok_count = sum(e == codes["start_entity_token"] for e in sent)
        end_entity_tok_count = sum(e == codes["end_entity_token"] for e in sent)
        if start_entity_tok_count > end_entity_tok_count:
            return "e", end_entity_tok_count
        return "o", end_entity_tok_count

    def get_trie_outside(sent, sent_orig):
        pointer_end = get_pointer_end(sent, sent_orig)

        if pointer_end:
            return [sent_orig[pointer_end]]
        else:
            # print("OUT")
            # print(decode_fn(sent))
            return []

    def get_pointer_end(sent, sent_orig):
        i = 0
        j = 0
        while i < len(sent):
            if sent[i] == sent_orig[j]:
                i += 1
                j += 1
            elif (
                sent[i] == codes["start_mention_token"]
                or sent[i] == codes["end_mention_token"]
            ):
                i += 1
            elif sent[i] == codes["start_entity_token"]:
                i += 1
                if i == len(sent):
                    return j
                while sent[i] != codes["end_entity_token"]:
                    i += 1
                    if i == len(sent):
                        return None
                i += 1
            else:
                return None

        return j if j != len(sent_orig) else None

    def get_pointer_mention(sent):
        pointer_end = -1
        for i, e in enumerate(sent):
            if e == codes["end_mention_token"]:
                pointer_end = i
        return pointer_end

    def get_trie_entity(sent, sent_orig, entity_tok_count):
        pointer_end = get_pointer_mention(sent)
        if entity_tok_count < len(sem_types):  # type: ignore
            return candidates_trie[
                sem_types[entity_tok_count]  # type: ignore
            ].get(sent[pointer_end + 1 :])
        else:
            # print("ENTITY")
            # print(sent)
            # print(decode_fn(sent))
            return []

    return prefix_allowed_tokens_fn
