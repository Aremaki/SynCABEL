# Synthetic Data Generation CLI Guide

This folder provides a modular CLI pipeline to create synthetic biomedical context sentences for UMLS concepts and export them in BigBio JSON format.

## Overview Pipeline
1. (Optional) Prepare UMLS concept synonym & definition parquets (already produced elsewhere).
2. Build per‑CUI user prompts (chunked) via `prepare_concepts.py`.
3. Generate synthetic sentences for each chunk with `generate.py`.
4. Convert generated parquet outputs to BigBio JSON with `convert_to_bigbio.py` (or let `generate.py` write JSON directly).
5. (Optional) Aggregate / convert remote or directory sources to BigBio JSON.

---
## 1. Prepare Concept User Prompts
Create chunked parquet files (`sample_{i}.parquet`) containing columns: `CUI`, `user_prompt`.

```bash
python scripts/2_generate_synthetic_data/prepare_concepts.py \
  --mm-path data/UMLS_processed/MM/all_disambiguated.parquet \
  --quaero-path data/UMLS_processed/QUAERO/all_disambiguated.parquet \
  --mm-def data/UMLS_processed/MM/umls_def.parquet \
  --quaero-def data/UMLS_processed/QUAERO/umls_def.parquet \
  --out-mm data/synthetic_data/SynthMM/user_prompts \
  --out-quaero data/synthetic_data/SynthQUAERO/user_prompts \
  --chunk-size 2500
```
Options:
- `--shuffle / --no-shuffle` shuffle concepts before chunking.

Outputs: `data/user_prompts_MM/sample_0.parquet`, `sample_2500.parquet`, ...

---
## 2. Generate Synthetic Sentences
Generates a parquet (`CUI`, `llm_output`) and a BigBio JSON for one chunk.

```bash
python scripts/2_generate_synthetic_data/generate.py run \
    --chunk 0 \
    --user-prompts-dir data/synthetic_data/SynthMM/user_prompts \
    --out-dir data/synthetic_data/SynthMM \
    --bigbio-out data/bigbio_datasets/SynthMM.json \
    --model-path models/Llama-3.3-70B-Instruct \
    --system-prompt-path scripts/2_generate_synthetic_data/prompts/system_prompt_mm.txt \
    --batch-size 4 \
    --max-new-tokens 1024 \
    --max-retries 5
```

Key parameters:
- `--chunk`: Start offset that matches `sample_{chunk}.parquet`.
- `--out-dir`: Directory to store per‑chunk generation parquet.
- `--bigbio-out`: Final JSON (will be overwritten each run—use per‑chunk paths if aggregating later).
- `--batch-size`, `--max-new-tokens`, `--max-retries`: control generation throughput & robustness.

### Generate All MM Concepts
Loop over all MM user prompt chunks:
```bash
for chunk_file in data/synthetic_data/SynthMM/user_prompts/sample_*.parquet; do
    chunk=$(basename "$chunk_file" .parquet | sed 's/sample_//')
    python scripts/2_generate_synthetic_data/generate.py run \
        --chunk "$chunk" \
        --user-prompts-dir data/synthetic_data/SynthMM/user_prompts \
        --out-dir data/synthetic_data/SynthMM \
        --bigbio-out data/bigbio_datasets/SynthMM_chunk${chunk}.json \
        --model-path models/Llama-3.3-70B-Instruct \
        --system-prompt-path scripts/2_generate_synthetic_data/prompts/system_prompt_mm.txt \
        --batch-size 4 \
        --max-new-tokens 1024 \
        --max-retries 5
done
```

### Generate All QUAERO Concepts  
Loop over all QUAERO user prompt chunks:
```bash
for chunk_file in data/synthetic_data/SynthQUAERO/user_prompts/sample_*.parquet; do
    chunk=$(basename "$chunk_file" .parquet | sed 's/sample_//')
    python scripts/2_generate_synthetic_data/generate.py run \
        --chunk "$chunk" \
        --user-prompts-dir data/synthetic_data/SynthQUAERO/user_prompts \
        --out-dir data/synthetic_data/SynthQUAERO \
        --bigbio-out data/bigbio_datasets/SynthQUAERO_chunk${chunk}.json \
        --model-path models/Llama-3.3-70B-Instruct \
        --system-prompt-path scripts/2_generate_synthetic_data/prompts/system_prompt_quaero.txt \
        --batch-size 4 \
        --max-new-tokens 1024 \
        --max-retries 5
done
```

---
## 3. Convert Parquet(s) to BigBio JSON
`convert_to_bigbio.py` offers multiple commands.

### a) Directory of parquets
```bash
python scripts/2_generate_synthetic_data/convert_to_bigbio.py convert \
  --parquet data/SynthMM \
  --json-out data/synthetic_data/SynthMM/SynthMM_bigbio.json
```
```bash
python scripts/2_generate_synthetic_data/convert_to_bigbio.py convert \
  --parquet data/SynthQUAERO \
  --json-out data/synthetic_data/SynthQUAERO/SynthQUAERO_bigbio.json
```
Options: `--limit` (row cap), `--fail-pattern` (default FAIL).

### b) Directly from HuggingFace Hub dataset
```bash
python scripts/2_generate_synthetic_data/convert_to_bigbio.py from-hub \
  --repo-id Aremaki/SynCABEL \
  --split SynthMedMentions \
  --json-out data/synthetic_data/SynthMM/SynthMM_bigbio.json
```
```bash
python scripts/2_generate_synthetic_data/convert_to_bigbio.py from-hub \
  --repo-id Aremaki/SynCABEL \
  --split SynthQUAERO \
  --json-out data/synthetic_data/SynthQUAERO/SynthQUAERO_bigbio.json
```
Custom columns:
```bash
python scripts/2_generate_synthetic_data/convert_to_bigbio.py from-hub \
  --repo-id Aremaki/SynCABEL \
  --split train \
  --cui-col concept_id \
  --text-col generations \
  --fail-pattern FAIL \
  --limit 20000 \
  --json-out data/bigbio_datasets/Synth_custom.json
```
---
## BigBio Output Structure
Each JSON record includes:
- `id`, `document_id`
- `passages`: single passage with full sentence text
- `entities`: bracketed entities (type `LLM_generated`, normalized to UMLS CUI)
- Empty lists for `events`, `coreferences`, `relations` (placeholder fields)

---
## Tips
- Ensure GPU with enough memory for chosen model & batch size (reduce `--batch-size` if OOM).
- For reproducibility you can later extend scripts to accept a random seed (not yet added here).
- Inspect failures by filtering rows containing `FAIL` in `llm_output` before conversion if needed.
