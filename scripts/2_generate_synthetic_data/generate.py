"""Generate synthetic biomedical context sentences for concepts."""

import re
import time
from pathlib import Path

import polars as pl
import torch
import typer
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,  # type: ignore
)

from .convert_to_bigbio import write_bigbio_json  # type: ignore

app = typer.Typer(help="Generate synthetic sentences for MM prompts.")


def load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
    system_prompt: str,
    max_new_tokens: int = 512,
    batch_size: int = 4,
    max_retries: int = 5,
    pattern: str = r"\[([^]]+)\]",
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


@app.command()
def run(
    chunk: int = typer.Option(..., help="Chunk index used in input/output filenames"),
    user_prompts_dir: Path = typer.Option(
        Path("data/user_prompts_MM"), help="Directory containing sample_{chunk}.parquet"
    ),
    out_dir: Path = typer.Option(
        Path("data/SynthMM"), help="Directory to write synthesized parquet"
    ),
    bigbio_out: Path = typer.Option(
        Path("data/bigbio_datasets/SynthMM.json"), help="BigBio JSON output path"
    ),
    model_path: Path = typer.Option(
        Path("models/Llama-3.3-70B-Instruct"), help="Model path"
    ),
    system_prompt_path: Path = typer.Option(
        Path("scripts/2_generate_synthetic_data/prompts/system_prompt_mm.txt"),
        help="System prompt text file",
    ),
    max_new_tokens: int = 1024,
    batch_size: int = 4,
    max_retries: int = 5,
) -> None:
    """Generate synthetic sentences for a chunk of user prompts."""
    user_prompts_path = user_prompts_dir / f"sample_{chunk}.parquet"
    if not user_prompts_path.exists():
        raise typer.BadParameter(f"Missing input file: {user_prompts_path}")
    user_prompts_df = pl.read_parquet(user_prompts_path)

    model, tokenizer = load_model_and_tokenizer(str(model_path))  # type: ignore
    system_prompt = load_system_prompt(system_prompt_path)

    result_df = generate_batches(
        model,
        tokenizer,
        user_prompts_df,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_retries=max_retries,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"sample_{chunk}.parquet"
    result_df.write_parquet(result_path)
    typer.echo(f"✅ Parquet written: {result_path}")

    # Convert to BigBio via shared helper
    n = write_bigbio_json(result_df, bigbio_out)
    typer.echo(
        f"✅ BigBio JSON written: {bigbio_out} ({n} records)\nChunk {chunk} complete."
    )


if __name__ == "__main__":
    app()
