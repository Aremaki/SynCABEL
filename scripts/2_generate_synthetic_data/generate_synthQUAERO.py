import argparse
import json
import re
import time

import polars as pl
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,  # type: ignore
)

system_prompt_v4_french = """
# Format d'input
- **CUI**:
Un identifiant unique du concept
- **Définitions**:
Une ou plusieurs définitions du concept
- **Mentions**:
Une liste de mentions (peut inclure des termes techniques ou artificiels)

# Tâches
1. **Sélection et génération de mentions** :
   - Évaluer d'abord quelles mentions existantes semblent les plus naturelles dans un contexte professionnel biomédical (clinique ou recherche).
   - Conserver uniquement les mentions les plus fluides et scientifiquement précises.
   - Générer ensuite des mentions supplémentaires de haute qualité qui :
       * Ajoutent des variations naturelles utilisées par les chercheurs.
       * Couvrent différentes conventions (acronymes, noms complets, noms fonctionnels).
       * Complètent les mentions existantes.
   - Le total doit être de **5 mentions** (mélange de mentions conservées et nouvelles).
   - **À éviter** :
       - Les virgules (`,`), parenthèses.
       - Les termes en MAJUSCULES ou avec underscores (ex. `ALPHA_TOCOPHEROL`).
       - Les noms chimiques complexes.
       - Les termes non français ou dans une autre langue.

2. **Phrases contextuelles** :
   Pour chacune des 5 mentions finales :
   - Rédiger une phrase claire et scientifiquement précise (15-30 mots) avec la mention entre `[crochets]`.
   - Chaque phrase doit :
       - Inclure le concept dans `[crochets]`.
       - Commencer par `exemple : `.
       - Avoir un **contexte distinct** (ex. mécanismes différents, populations variées, systèmes biologiques spécifiques).

# Format de la réponse
## Mentions finales
1. Mention 1
2. Mention 2
...
5. Mention 5

## Phrases contextuelles
exemple : "Phrase avec [mention 1] dans son contexte..."
exemple : "Phrase avec [mention 2] dans son contexte..."
...
exemple : "Phrase avec [mention 5] dans son contexte..."

# Exemples
## Exemple 1
- **CUI**:
C0002844
- **Définitions**:
* 1. A type of hormone that promotes the development and maintenance of male sex characteristics.
* 2. A class of sex hormones associated with the development and maintenance of the secondary male sex characteristics, sperm induction, and sexual differentiation. In addition to increasing virility and libido, they also increase nitrogen and water retention and stimulate skeletal growth. (MeSH)
* 3. a kind of male sex hormone
* 4. Compounds that interact with ANDROGEN RECEPTORS in target tissues to bring about the effects similar to those of TESTOSTERONE. Depending on the target tissues, androgenic effects can be on SEX DIFFERENTIATION; male reproductive organs, SPERMATOGENESIS; secondary male SEX CHARACTERISTICS; LIBIDO; development of muscle mass, strength, and power.
* 5. compounds that interact with androgen receptors in target tissues to bring about the effects similar to those of testosterone; depending on the target tissues, androgenic effects can be on sexual differentiation, male reproductive organs, spermtogenesis, secondary male sex characteristics, libido, development of muscle mass, strength, and power.
- **Mentions**:
'Compounds, Androgenic', 'Androgen preparation', 'Androgen Receptor Agonists', 'Androgen product', 'Therapeutic Androgen', 'Agonistes des récepteurs androgéniques', 'Agonistes des récepteurs aux androgènes', 'Androgens (medication)', 'Agonistes des récepteurs des androgènes', 'Androgènes', 'Androgenic drug', 'Composés androgéniques', 'Agonists, Androgen Receptor', 'Androgen', 'Androgens', 'Androgenic preparation', 'Androgenic preparation (product)', 'ANDROGENIC PREPARATIONS', 'Androgenic preparation (substance)', 'Receptor Agonists, Androgen', 'Androgenic Agents', 'Agents, Androgenic', 'Androgenic Compounds', 'A05 ANDROGENIC PREPARATIONS', 'ANDROGENS', 'Androgen (substance)'

## Mentions finales
1. Androgènes
2. Hormones androgènes
3. Agents androgéniques
4. Agonistes des récepteurs androgéniques
5. Traitement androgénique

## Phrases contextuelles
exemple : Au cours de la puberté, les [androgènes] sont essentiels pour l’apparition des caractères sexuels secondaires chez l’homme.
exemple : Les [hormones androgènes] interviennent dans la régulation du développement musculaire et de la densité osseuse.
exemple : Les [agents androgéniques] sont parfois prescrits pour traiter certains types d’hypogonadisme masculin.
exemple : Les [agonistes des récepteurs androgéniques] sont utilisés dans la recherche sur les troubles liés au développement sexuel.
exemple : Un [traitement androgénique] peut être envisagé dans la prise en charge de certaines formes d’insuffisance hormonale masculine.

## Exemple 2
- **CUI**:
C0257535
- **Définitions**:
* 1. A 38-kDa mitogen-activated protein kinase that is abundantly expressed in a broad variety of cell types. It is involved in the regulation of cellular stress responses as well as the control of proliferation and survival of many cell types. The kinase activity of the enzyme is inhibited by the pyridinyl-imidazole compound SB 203580.
- **Mentions**:
'Mitogen-Activated Protein Kinase 14', 'Stress activated protein kinase 2a', 'Mitogen-Activated Protein Kinase 14 (Chemical/Ingredient)', 'P38alphaMAPK', 'MAPK14 Mitogen Activated Protein Kinase', 'Protéine Mxi2', 'MAPK14', 'CSBP, Kinase', 'Protéine-2 d'intéraction avec MAX', 'Mxi2 Protein', 'MAX Interacting Protein 2', 'Kinase CSBP', 'Protéine-kinase MAPK14 activée par les facteurs mitogènes', 'Protéine-kinase-2a activée par le stress', 'SAPK2a', 'Protéine de liaison au CSAID', 'Mitogen Activated Protein Kinase 14', 'P38alpha Mitogen Activated Protein Kinase', 'Stress-activated protein kinase 2a', 'P38 mapk', 'MAX-Interacting Protein 2', 'Kinase map p38', 'Protéine-kinase p38alpha activée par les facteurs mitogènes', 'P38alpha MAP Kinase', 'MAP Kinase, p38alpha', 'MAPK14 Mitogen-Activated Protein Kinase', 'P38alpha Mitogen-Activated Protein Kinase', 'CSAID-Binding Protein', 'P38 map kinase', 'Cytokine Suppressive Anti inflammatory Drug Binding Protein', 'MAP-kinase p38alpha', 'MITOGEN ACTIVATED PROTEIN KINASE 14', 'Cytokine Suppressive Anti-inflammatory Drug Binding Protein', 'CSAID Binding Protein'

## Mentions finales
1. MAPK14
2. p38 alpha
3. kinase p38 alpha
4. protéine Mxi2
5. SAPK2a

## Phrases contextuelle
exemple : Les analyses transcriptomiques montrent une expression accrue de [MAPK14] dans les tissus affectés par l'inflammation chronique, suggérant son rôle dans la régulation des réponses immunitaires.
exemple : L'inhibition de [p38 alpha] réduit la production de cytokines pro-inflammatoires dans les modèles cellulaires exposés à un stress oxydatif aigu.
exemple : La régulation de la prolifération cellulaire en réponse à des agents chimiothérapeutiques fait intervenir la [kinase p38 alpha] dans certains types de tumeurs solides.
exemple : L'isoforme [protéine Mxi2] est associée à des cellules rénales et intervient spécifiquement dans les processus signalétiques liés à l'apoptose.
exemple : Les chercheurs ont montré que [SAPK2a] module la réponse des cellules endothéliales lors d'une exposition à des facteurs de croissance angiogéniques.
"""


def process_sentence(sentence, document_id, sentence_num, cui):
    # Initialize the output dictionary for this sentence
    output = {
        "id": str(document_id),
        "document_id": f"{document_id}_{sentence_num}",
        "passages": [
            {
                "id": f"{document_id}_{sentence_num}__text",
                "type": "abstract",
                "text": [sentence + "\n"],
                "offsets": [[0, len(sentence)]],
            }
        ],
        "entities": [],
        "events": [],
        "coreferences": [],
        "relations": [],
    }

    # Find all entities in square brackets
    entities = re.findall(r"\[(.*?)\]", sentence)

    # For each entity, find all occurrences in the sentence
    for entity in set(entities):  # Using set() to avoid duplicates
        # Escape special regex characters in the entity
        escaped_entity = re.escape(entity)
        # Find all matches of this entity in the sentence
        matches = re.finditer(escaped_entity, sentence)

        for match_num, match in enumerate(matches, 1):
            entity_id = f"{document_id}_{sentence_num}_T{match_num}"
            entity_data = {
                "id": entity_id,
                "type": "LLM_generated",
                "text": [entity],
                "offsets": [[match.start(), match.end()]],
                "normalized": [
                    {
                        "db_name": "UMLS",
                        "db_id": cui,  # Using the provided CUI
                    }
                ],
            }
            output["entities"].append(entity_data)

    return output


def load_model_and_tokenizer(model_path):
    torch.backends.cuda.enable_flash_sdp(True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Set a padding token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
    )
    return model, tokenizer


def apply_chat_template(tokenizer, batch_user_prompts, system_prompt):
    batch_chat = []
    for user_prompt in batch_user_prompts:
        batch_chat.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
    prompt = tokenizer.apply_chat_template(batch_chat, tokenize=False)
    return prompt


def generate_batches(
    model,
    tokenizer,
    user_prompts_df: pl.DataFrame,
    max_new_tokens=512,
    system_prompt=system_prompt_v4_french,
    batch_size=6,
    max_retries=5,
    pattern=r"\[([^]]+)\]",
):
    user_prompts = user_prompts_df["user_prompt"].to_list()
    cui_codes = user_prompts_df["CUI"].to_list()
    all_outputs = []
    timing_data = []

    for batch_start in tqdm(
        range(0, len(user_prompts), batch_size), desc="Generating in batches"
    ):
        batch_user_prompts = user_prompts[batch_start : batch_start + batch_size]
        batch_cui_codes = cui_codes[batch_start : batch_start + batch_size]

        # Apply chat template and prepare batch prompts
        batch_inputs = apply_chat_template(
            tokenizer,
            batch_user_prompts,
            system_prompt,
        )

        # Tokenize as batch
        inputs = tokenizer(batch_inputs, padding="longest", return_tensors="pt")
        inputs = {key: val.cuda() for key, val in inputs.items()}
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        success_mask = [False] * len(batch_user_prompts)
        batch_final_text = [None] * len(batch_user_prompts)

        for _ in range(1, max_retries + 1):
            if all(success_mask):
                break

            with torch.inference_mode():
                gen_start_time = time.time()
                output_tokens = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                gen_time = time.time() - gen_start_time

            # Decode and post-process outputs
            decoded_outputs = tokenizer.batch_decode(
                output_tokens, skip_special_tokens=True
            )

            print(f"""Processed Batch {batch_start}: {gen_time:.2f} sec/batch""")

            for i, (_, decoded_output) in enumerate(
                zip(batch_cui_codes, decoded_outputs)
            ):
                if success_mask[i]:
                    continue

                decoded_text = decoded_output.split("assistant\n\n")[-1].strip()
                examples = [
                    line.split("exemple : ")[1]
                    for line in decoded_text.split("\n")
                    if "exemple : " in line and bool(re.search(pattern, line))
                ]

                if isinstance(examples, list) and len(examples) >= 5:
                    success_mask[i] = True
                    batch_final_text[i] = "\n".join(examples[:5])  # type: ignore
                else:
                    batch_final_text[i] = f"FAIL !!\n\n{decoded_text}"  # type: ignore

                # Timing and metrics
                input_len = len(inputs["input_ids"][i])
                output_len = len(output_tokens[i])
                new_tokens = output_len - input_len
                tokens_per_second = new_tokens / gen_time
                timing_data.append({
                    "total_new_tokens": new_tokens,
                    "tokens_per_second": tokens_per_second,
                    "time_per_cui": gen_time / len(batch_user_prompts),
                })

        # Collect all outputs
        for cui, out_text in zip(batch_cui_codes, batch_final_text):
            all_outputs.append((cui, out_text))

    # Summary
    if timing_data:
        avg_tps = sum(t["tokens_per_second"] for t in timing_data) / len(timing_data)
        avg_sec_per_cui = sum(t["time_per_cui"] for t in timing_data) / len(timing_data)
        total_time = sum(t["time_per_cui"] for t in timing_data)
        total_tokens = sum(t["total_new_tokens"] for t in timing_data)

        print("\n=== Summary Statistics ===")
        print(f"Average tokens/second: {avg_tps:.2f}")
        print(f"Average seconds per CUI: {avg_sec_per_cui:.3f}")
        print(f"Total generation time: {total_time:.3f}")
        print(f"Total tokens generated: {total_tokens:,}")

    return pl.DataFrame(all_outputs, schema=["CUI", "llm_output"], orient="row")


def main(chunk):
    # Load your input templates from a Parquet/CSV/NDJSON file
    user_prompts_path = f"data/user_prompts_quaero/sample_{chunk}.parquet"
    user_prompts_df = pl.read_parquet(user_prompts_path)

    model_path = "models/Llama-3.3-70B-Instruct"
    model, tokenizer = load_model_and_tokenizer(model_path)  # type: ignore

    result_df = generate_batches(
        model,
        tokenizer,
        user_prompts_df,
        max_new_tokens=1024,
        system_prompt=system_prompt_v4_french,
    )

    # Save the results
    result_df.write_parquet(f"data/SynthQUAERO/sample_{chunk}.parquet")
    print("✅ Finished.")

    # Process each sentence separately
    result_df = result_df.filter(~pl.col("llm_output").str.contains("FAIL"))
    bigbio_llm_all = []
    for i in tqdm(range(len(result_df))):
        cui = result_df["CUI"][i]
        sentences = result_df["llm_output"][i].split("\n")
        for j, sentence in enumerate(sentences):
            bigbio_llm_all.append(process_sentence(sentence, i, j, cui))

    # Save to JSON file
    with open(
        "data/bigbio_datasets/SynthQUAERO.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(bigbio_llm_all, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for generate synthetic data")
    parser.add_argument("--chunk", type=int, required=True, help="The partition number")

    # Pass the parsed argument to the main function
    args = parser.parse_args()
    main(args.chunk)
