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

system_prompt_mm = """
Your job is to generate synthetic examples for entity linking, mimicking the language style found in PubMed abstracts.

# Input Format
- **CUI**:
A unique concept ID
- **Definitions**:
One or more definitions of the concept
- **Mentions**:
A list of mentions (may include technical or unnatural terms)

# Tasks
Perform these tasks:
1. **Mention Selection & Generation:**
   - First evaluate which existing mentions sound most natural in professional biomedical writing (clinical or research contexts)
   - Keep only the most fluent mentions from the existing list
   - Then generate additional high-quality mentions that:
     * Add natural variations researchers would use
     * Cover different naming conventions (acronyms, full names, functional names)
     * Complement the existing mentions
   - The total should be 5 mentions (mix of kept existing and newly generated)
   - **Avoid**:
       - Commas (`,`), parentheses.
       - ALL_CAPS or underscored terms (e.g., `ALPHA_TOCOPHEROL`).
       - Chemical names.
       - Non-English terms or foreign language mentions.
2. **Contextual Usage Examples:**
   For each of the final 5 mentions:
   - Write one clear sentence (15-30 words) with the mention in brackets in the style of a biomedical abstract.
   - Each must:
       - Include the concept in `[brackets]`.
       - Start with `example: `.
       - Be **distinct in context** (e.g., different study types, organisms, biological systems, or outcomes)
       - Mimic the tone and vocabulary typical of **PubMed biomedical research** papers

# PubMed Abstract Style Examples
- Red ripe fruits were morphologically characterized and biochemically analyzed for their content in glycoalkaloids, phenols, amino acids, and Amadori products.
- "MeHg bioaccumulated and induced significant increase of the photosynthesis efficiency, while the algal growth, oxidative stress, and chlorophyll fluorescence were unaffected.
- "Students from a midsize private university (n = 159) completed a 15-minute anonymous questionnaire, including questions on risk behaviors, sleep habits, alcohol, and caffeine consumption.

# Output Format
## Final Mention Set
1. Mention 1
2. Mention 2
...
5. Mention 5

## Usage Examples
example: "Sentence with [mention 1] in context..."
example: "Sentence with [mention 2] in context..."
...
example: "Sentence with [mention 5] in context..."

# Examples
## Example 1
- **CUI**:
C0969677
- **Definitions**:
* 1. A naturally-occurring form of vitamin E, a fat-soluble vitamin with potent antioxidant properties. Considered essential for the stabilization of biological membranes (especially those with high amounts of polyunsaturated fatty acids), d-alpha-Tocopherol is a potent peroxyl radical scavenger and inhibits noncompetitively cyclooxygenase activity in many tissues, resulting in a decrease in prostaglandin production. Vitamin E also inhibits angiogenesis and tumor dormancy through suppressing vascular endothelial growth factor (VEGF) gene transcription. (NCI04)
* 2. A natural tocopherol and one of the most potent antioxidant tocopherols. It exhibits antioxidant activity by virtue of the phenolic hydrogen on the 2H-1-benzopyran-6-ol nucleus. It has four methyl groups on the 6-chromanol nucleus. The natural d form of alpha-tocopherol is more active than its synthetic dl-alpha-tocopherol racemic mixture.
* 3. The orally bioavailable alpha form of the naturally-occurring fat-soluble vitamin E, with potent antioxidant and cytoprotective activities. Upon administration, alpha-tocopherol neutralizes free radicals, thereby protecting tissues and organs from oxidative damage. Alpha-tocopherol gets incorporated into biological membranes, prevents protein oxidation and inhibits lipid peroxidation, thereby maintaining cell membrane integrity and protecting the cell against damage. In addition, alpha-tocopherol inhibits the activity of protein kinase C (PKC) and PKC-mediated pathways. Alpha-tocopherol also modulates the expression of various genes, plays a key role in neurological function, inhibits platelet aggregation and enhances vasodilation. Compared with other forms of tocopherol, alpha-tocopherol is the most biologically active form and is the form that is preferentially absorbed and retained in the body.
* 4. Tocopherol with three methyl groups on its chromanol ring.
- **Mentions**:
'D-alpha-tocopherol (substance)', '.ALPHA.-TOCOPHEROL, D-', 'Alpha-Tocopherol', '(+)-alpha-Tocopherol', 'D-alpha-Tocopherol', '3,4-Dihydro-2,5,7,8-tetramethyl-2-(4,8,12-trimethyltridecyl)-2H-1-benzopyran-6-ol', 'Alfa tocopherol', 'D alpha Tocopherol', 'Alpha Tocopherol', 'D-alpha-tocopherol', 'Alpha-Tocopherol (Chemical/Ingredient)', 'Alpha-tocopherol', 'Alpha tocopheryl product', 'Tocophérol alpha', '2,5,7,8-tetramethyl-2-(4,8,12-trimethyltridecyl)chroman-6-ol', 'R,R,R-alpha-Tocopherol', 'Alpha-Tocopherol preparation', 'D-Alpha-Tocopherol', 'Alpha-Tocophérol', 'Tocopherol, d-alpha', 'Alpha-tocopherol (substance)', '.ALPHA.-TOCOPHEROL', 'Alpha tocopherol', 'Alpha tocopheryl product (product)', '2,5,7,8-Tétramethyl-2-(4,8,12-triméthyltridécyl)-3,4-dihydro-2H-chromén-6-ol', 'D-alpha Tocopherol', '(+/-)-alpha-Tocopherol', '(+-)-alpha-Tocopherol'

## Final Mention Set
1. Alpha-tocopherol
2. D-alpha-tocopherol
3. Alpha tocopherol
4. Natural vitamin E
5. Alpha-tocopherol supplement

## Usage Examples
example: In rat liver microsomes, [alpha-tocopherol] markedly reduced lipid peroxidation induced by iron-mediated oxidative stress.
example: Oral administration of [D-alpha-tocopherol] significantly attenuated neuronal damage in a murine model of ischemia-reperfusion injury.
example: Plasma concentrations of [alpha tocopherol] were inversely correlated with markers of systemic inflammation in patients with metabolic syndrome.
example: Dietary intake of [natural vitamin E] was associated with improved endothelial function in individuals at risk for cardiovascular disease.
example: Long-term use of an [alpha-tocopherol supplement] decreased the incidence of age-related macular degeneration in a randomized controlled trial.

## Example 2
- **CUI**:
C0021760
- **Definitions**:
* 1. A cytokine that stimulates the growth and differentiation of B-LYMPHOCYTES and is also a growth factor for HYBRIDOMAS and plasmacytomas. It is produced by many different cells including T-LYMPHOCYTES; MONOCYTES; and FIBROBLASTS.
- **Mentions**:
'DIFFER FACTOR B CELL', 'Monocyte-granulocyte inducer type 2', 'Myeloid blood cell differentiation protein', 'Interleukine 6', 'Differentiation-Inducing Protein, Myeloid', 'Interleukin-6', 'B CELL DIFFER FACTOR', 'Facteur-2 de différenciation des lymphocytes B', 'B Cell Stimulatory Factor-2', 'Growth Factor, Plasmacytoma', 'Interleukin-6 (substance)', 'Interféron-bêta2', 'Plasmacytoma Growth Factor', 'DIFFER FACTOR 2 B CELL', 'Interferon beta2', 'Interferon beta 2', 'B-Cell Stimulatory Factor-2', 'Hepatocyte Stimulating Factor', 'Differentiation Factor, B-Cell', 'B Cell Differentiation Factor', 'IL-HP1', 'B CELL DIFFER FACTOR 2', 'HPGF', 'Facteur de croissance du myélome', 'PCT-GF', 'Facteur-2 de stimulation des lymphocytes B', 'HGF', 'Hybridoma/plasmocytoma growth factor', 'IFN-beta 2', 'B-cell stimulating factor 2', 'Plasmacytoma growth factor', 'Interleukine-6', 'Hepatocyte-Stimulating Factor', 'Differentiation Factor 2, B Cell', 'Interleukin-6 (Chemical/Ingredient)', 'Facteur de différenciation des cellules B', 'Beta-2, Interferon', 'MYELOID DIFFER INDUCING PROTEIN', 'Interleukin 6', 'IL 006', 'Facteur BSF-2', 'B-Cell Differentiation Factor', 'B-Cell Stimulatory Factor 2', 'Myeloid Differentiation Inducing Protein', 'IFN-bêta2', 'Differentiation Factor, B Cell', 'Hepatocyte stimulating factor', 'BSF-2', 'B Cell Stimulatory Factor 2', 'HSF', 'IL-6', 'IL6', 'INTERLEUKIN 006', 'Hybridoma growth factor', 'MGI-2A', 'Facteur de différenciation des lymphocytes B', 'B Cell Differentiation Factor 2', 'Facteur-2 de stimulation des cellules B', 'Myeloid Differentiation-Inducing Protein', 'Facteur HSF', 'Facteur-2 de différenciation des cellules B', 'Interferon beta-2', 'IFN beta2', '26kDa (inducible) protein', 'Differentiation Factor-2, B-Cell', 'MGI-2', 'Hybridoma Growth Factor', '26kDa (inducible) factor', 'B-Cell Differentiation Factor-2', 'Growth Factor, Hybridoma', 'Interféron-bêta-2'

## Final Mention Set
1. Interleukin-6
2. IL-6
3. B cell stimulatory factor 2
4. Plasmacytoma growth factor
5. Hybridoma growth factor

## Usage Examples
example: Elevated serum levels of [interleukin-6] were associated with poor prognosis in patients with advanced non-small cell lung cancer.
example: The expression of [IL-6] mRNA increased significantly in the hippocampus following traumatic brain injury in a rat model.
example: [B cell stimulatory factor 2] was shown to enhance immunoglobulin production in activated human peripheral blood B lymphocytes.
example: A dose-dependent proliferative response was observed in mouse myeloma cells upon exposure to [plasmacytoma growth factor].
example: Co-culture with macrophages led to increased secretion of [hybridoma growth factor], promoting survival of antigen-specific hybridoma clones.
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


def apply_chat_template(tokenizer, batch_user_prompts, system_prompt, instruct=True):
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
    system_prompt=system_prompt_mm,
    batch_size=4,
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
            tokenizer, batch_user_prompts, system_prompt, instruct=True
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
                    line.split("example: ")[1]
                    for line in decoded_text.split("\n")
                    if "example: " in line and bool(re.search(pattern, line))
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
    user_prompts_path = f"data/user_prompts_MM/sample_{chunk}.parquet"
    user_prompts_df = pl.read_parquet(user_prompts_path)

    model_path = "models/Llama-3.3-70B-Instruct"
    model, tokenizer = load_model_and_tokenizer(model_path)  # type: ignore

    result_df = generate_batches(
        model,
        tokenizer,
        user_prompts_df,
        max_new_tokens=1024,
        system_prompt=system_prompt_mm,
    )

    # Save the results
    result_df.write_parquet(f"data/SynthMM/sample_{chunk}.parquet")
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
        "data/bigbio_datasets/SynthMM.json",
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
