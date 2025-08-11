# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import re
from collections import defaultdict


def chunk_it(seq, num):
    assert num > 0
    chunk_len = len(seq) // num
    chunks = [seq[i * chunk_len : i * chunk_len + chunk_len] for i in range(num)]

    diff = len(seq) - chunk_len * num
    for i in range(diff):
        chunks[i].append(seq[chunk_len * num + i])

    return chunks


def get_entity_spans_pre_processing(sentences):
    return [
        (
            f" {sent} ".replace("\xa0", " ")
            .replace("{", "(")
            .replace("}", ")")
            .replace("[", "(")
            .replace("]", ")")
        )
        for sent in sentences
    ]


def get_entity_spans_post_processing(sentences):
    outputs = []
    for sent in sentences:
        sent = re.sub(r"{.*?", "{ ", sent)
        sent = re.sub(r"}.*?", "} ", sent)
        sent = re.sub(r"\].*?", "] ", sent)
        sent = re.sub(r"\[.*?", "[ ", sent)
        sent = re.sub(r"\s{2,}", " ", sent)
        sent = re.sub(r"\. \. \} \[ (.*?) \]", r". } [ \1 ] .", sent)
        sent = re.sub(r"\, \} \[ (.*?) \]", r" } [ \1 ] ,", sent)
        sent = re.sub(r"\; \} \[ (.*?) \]", r" } [ \1 ] ;", sent)
        sent = sent.replace("{ ", "{").replace(" } [ ", "}[").replace(" ]", "]")
        outputs.append(sent)

    return outputs


def get_entity_spans_finalize(
    input_sentences, output_sentences, start_entity, start_tag, end_tag
):
    return_outputs = []
    nested = False
    doc_id = 0
    for input_, output_ in zip(input_sentences, output_sentences):
        doc_id += 1
        entities = []
        status = "out"
        i = 0
        j = 0
        s_ent = len(start_entity)
        s_tag = len(start_tag)
        e_tag = len(end_tag)
        while (j < len(output_) and i < len(input_)) or status != "out":
            if status == "out":
                if output_[j : j + s_ent] == start_entity:
                    nested = 1
                    entities.append([i, 0, "", ""])
                    j += s_ent
                    status = "entity"
                elif j < len(output_) and i < len(input_):
                    if input_[i] == output_[j]:
                        i += 1
                        j += 1
                    elif output_[j] in [" ", "\n"]:
                        j += 1
                    elif input_[i] in [" ", "\n", "⩾"]:
                        i += 1
                    else:
                        # print(input_[i:])
                        # print(output_[j:])
                        # print(doc_id)
                        # print("output is misaligned")
                        break

            elif status == "entity":
                if output_[j : j + s_tag] == start_tag:
                    j += s_tag
                    status = "tag"
                # NESTED
                elif output_[j : j + s_ent] == start_entity:
                    nested += 1
                    entities.append([i, 0, "", ""])
                    j += s_ent
                elif j < len(output_) and i < len(input_):
                    if input_[i] == output_[j]:
                        for k in range(nested):
                            entities[-(k + 1)][3] += input_[i]
                            entities[-(k + 1)][1] += 1
                        i += 1
                        j += 1
                    elif output_[j] == " ":
                        j += 1
                    elif input_[i] == " ":
                        for k in range(nested):
                            entities[-(k + 1)][3] += input_[i]
                            entities[-(k + 1)][1] += 1
                        i += 1
                    else:
                        entities.pop()
                        # print("mention is misaligned")
                        break
                else:
                    status = "out"

            elif status == "tag":
                if output_[j : j + e_tag] == end_tag:
                    if nested > 1:
                        entities = (
                            entities[:-nested] + entities[-1:] + entities[-nested:-1]
                        )
                        nested -= 1
                        status = "entity"
                    else:
                        status = "out"
                    j += e_tag
                elif j < len(output_):
                    entities[-1][2] += output_[j]
                    j += 1
                else:
                    entities.pop()
                    # print("entity is misaligned")
                    break

        return_outputs.append(sorted(entities))

    return return_outputs


def strong_tp(guess_entities, gold_entities):
    return len(gold_entities.intersection(guess_entities))


def weak_tp(guess_entities, gold_entities):
    tp = 0
    for pred in guess_entities:
        for gold in gold_entities:
            if (
                pred[0] == gold[0]
                and (
                    gold[1] <= pred[1] <= gold[1] + gold[2]
                    or gold[1] <= pred[1] + pred[2] <= gold[1] + gold[2]
                )
                and pred[3] == gold[3]
            ):
                tp += 1

    return tp


def get_micro_precision(guess_entities, gold_entities, mode="strong"):
    guess_entities = set(guess_entities)
    gold_entities = set(gold_entities)

    if mode == "strong":
        return (
            (strong_tp(guess_entities, gold_entities) / len(guess_entities))
            if len(guess_entities)
            else 0
        )
    elif mode == "weak":
        return (
            (weak_tp(guess_entities, gold_entities) / len(guess_entities))
            if len(guess_entities)
            else 0
        )


def get_micro_recall(guess_entities, gold_entities, mode="strong"):
    guess_entities = set(guess_entities)
    gold_entities = set(gold_entities)

    if mode == "strong":
        return (
            (strong_tp(guess_entities, gold_entities) / len(gold_entities))
            if len(gold_entities)
            else 0
        )
    elif mode == "weak":
        return (
            (weak_tp(guess_entities, gold_entities) / len(gold_entities))
            if len(gold_entities)
            else 0
        )


def get_micro_f1(guess_entities, gold_entities, mode="strong"):
    precision = get_micro_precision(guess_entities, gold_entities, mode)
    recall = get_micro_recall(guess_entities, gold_entities, mode)
    return (
        (2 * (precision * recall) / (precision + recall)) if precision + recall else 0  # type: ignore
    )


def get_doc_level_guess_gold_entities(guess_entities, gold_entities):
    new_guess_entities = defaultdict(list)
    for e in guess_entities:
        new_guess_entities[e[0]].append(e)

    new_gold_entities = defaultdict(list)
    for e in gold_entities:
        new_gold_entities[e[0]].append(e)

    return new_guess_entities, new_gold_entities


def get_macro_precision(guess_entities, gold_entities, mode="strong"):
    guess_entities, gold_entities = get_doc_level_guess_gold_entities(
        guess_entities, gold_entities
    )
    all_scores = [
        get_micro_precision(guess_entities[k], gold_entities[k], mode)
        for k in guess_entities
    ]
    return (sum(all_scores) / len(all_scores)) if len(all_scores) else 0  # type: ignore


def get_macro_recall(guess_entities, gold_entities, mode="strong"):
    guess_entities, gold_entities = get_doc_level_guess_gold_entities(
        guess_entities, gold_entities
    )
    all_scores = [
        get_micro_recall(guess_entities[k], gold_entities[k], mode)
        for k in guess_entities
    ]
    return (sum(all_scores) / len(all_scores)) if len(all_scores) else 0  # type: ignore


def get_macro_f1(guess_entities, gold_entities, mode="strong"):
    guess_entities, gold_entities = get_doc_level_guess_gold_entities(
        guess_entities, gold_entities
    )
    all_scores = [
        get_micro_f1(guess_entities[k], gold_entities[k], mode) for k in guess_entities
    ]
    return (sum(all_scores) / len(all_scores)) if len(all_scores) else 0
