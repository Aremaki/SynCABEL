"""
Core models for SynCABEL
"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List

from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    LlamaForCausalLM,
    MBartForConditionalGeneration,
    MT5ForConditionalGeneration,
)

from syncabel.utils import chunk_it

logger = logging.getLogger(__name__)


def skip_undesired_tokens(outputs, tokenizer):
    if any("tag" in token for token in tokenizer.all_special_tokens):
        tokens_to_remove = tokenizer.all_special_tokens[:-3]
    elif any("{" in token for token in tokenizer.all_special_tokens):
        tokens_to_remove = tokenizer.all_special_tokens[:-4]
    else:
        tokens_to_remove = tokenizer.all_special_tokens
    cleaned_outputs = []
    for sequence in outputs:
        for token in tokens_to_remove:
            sequence = sequence.replace(token, "")  # Remove unwanted special tokens
        cleaned_outputs.append(sequence.strip())
    return cleaned_outputs


class _GENREHubInterface:
    def sample(
        self,
        sentences: List[str],
        num_beams: int = 5,
        num_return_sequences=5,
        text_to_id: Dict[str, str] = None,  # type: ignore
        marginalize: bool = False,
        **kwargs,
    ) -> List[str]:
        input_args = {
            k: v.to(self.device)  # type: ignore
            for k, v in self.tokenizer.batch_encode_plus(  # type: ignore
                sentences, padding="longest", return_tensors="pt"
            ).items()
        }

        outputs = self.generate(  # type: ignore
            **input_args,
            min_length=0,
            max_length=1024,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs,
        )
        decoded_sequences = self.tokenizer.batch_decode(  # type: ignore
            outputs.sequences,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        cleaned_output_sequences = skip_undesired_tokens(
            decoded_sequences,
            self.tokenizer,  # type: ignore
        )

        if num_return_sequences == 1:
            if len(cleaned_output_sequences) == 1:
                return cleaned_output_sequences[0]
            else:
                return cleaned_output_sequences
        else:
            outputs = chunk_it(
                [
                    {
                        "text": text,
                        "score": score,
                    }
                    for text, score in zip(
                        cleaned_output_sequences,
                        outputs.sequences_scores,
                    )
                ],
                len(sentences),
            )

        return outputs  # type: ignore

    def encode(self, sentence):
        return self.tokenizer.encode(sentence, return_tensors="pt")[0]  # type: ignore


class BARTGENREHubInterface(_GENREHubInterface, BartForConditionalGeneration):
    pass


class MBARTGENREHubInterface(_GENREHubInterface, MBartForConditionalGeneration):
    pass


class MT5GENREHubInterface(_GENREHubInterface, MT5ForConditionalGeneration):
    pass


class LlamaGENREHubInterface(_GENREHubInterface, LlamaForCausalLM):
    pass


class MBart_GENRE(MBartForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        model = MBARTGENREHubInterface.from_pretrained(model_name_or_path)
        model.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, add_prefix_space=False
        )
        return model


class Bart_GENRE(BartForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        model = BARTGENREHubInterface.from_pretrained(model_name_or_path)
        model.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, add_prefix_space=False
        )
        return model


class MT5_GENRE(MT5ForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        model = MT5GENREHubInterface.from_pretrained(model_name_or_path)
        model.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, add_prefix_space=False
        )
        return model


class Llama_GENRE(LlamaForCausalLM):
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        model = LlamaGENREHubInterface.from_pretrained(model_name_or_path)
        model.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return model
