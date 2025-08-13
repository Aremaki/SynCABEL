"""Utilities & CLI to convert generated LLM parquet outputs to BigBio JSON.

Expected input parquet schema:
  - CUI: str
  - llm_output: str (multiple lines, each a sentence containing bracketed entities)

The conversion will:
  1. Drop rows whose llm_output contains the string 'FAIL'.
  2. Split remaining llm_output on newline to sentences.
  3. For each sentence, detect bracketed entities [ ... ] and create BigBio style
     entity annotations (type 'LLM_generated', normalized db_name 'UMLS').

Reusable functions:
  process_sentence -> build a BigBio record for a single sentence
  dataframe_to_bigbio -> build list[dict] from a dataframe and (optionally) write JSON

CLI usage:
  python convert_to_bigbio.py convert --parquet data/SynthMM/sample_0.parquet \
      --json-out data/bigbio_datasets/SynthMM.json
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import polars as pl
import typer
from datasets import load_dataset
from tqdm import tqdm

app = typer.Typer(help="Convert synthetic generation parquet(s) to BigBio JSON.")


def process_sentence(
    sentence: str, document_id: int, sentence_num: int, cui: str
) -> dict:
    """Convert one sentence with bracketed entities to a BigBio passage record."""
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
    entities = re.findall(r"\[(.*?)\]", sentence)
    for entity in set(entities):  # avoid duplicates
        escaped = re.escape(entity)
        for match_num, match in enumerate(re.finditer(escaped, sentence), 1):
            output["entities"].append({
                "id": f"{document_id}_{sentence_num}_T{match_num}",
                "type": "LLM_generated",
                "text": [entity],
                "offsets": [[match.start(), match.end()]],
                "normalized": [
                    {
                        "db_name": "UMLS",
                        "db_id": cui,
                    }
                ],
            })
    return output


def dataframe_to_bigbio(df: pl.DataFrame) -> list[dict]:
    """Convert a generation result dataframe to list of BigBio records."""
    if not {"CUI", "llm_output"}.issubset(df.columns):  # basic schema check
        raise ValueError("Input DataFrame must contain 'CUI' and 'llm_output' columns")
    filtered = df.filter(~pl.col("llm_output").str.contains("FAIL"))
    records: list[dict] = []
    for i in tqdm(range(len(filtered)), desc="Converting to BigBio"):
        cui = filtered["CUI"][i]
        for j, sentence in enumerate(filtered["llm_output"][i].split("\n")):
            if sentence.strip():
                records.append(process_sentence(sentence, i, j, cui))
    return records


def _write_json(objs: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(objs), f, ensure_ascii=False, indent=4)


@app.command()
def convert(
    parquet: Path = typer.Option(
        ..., exists=True, help="Input generation parquet file or folder"
    ),
    json_out: Path = typer.Option(..., help="Output BigBio JSON path"),
) -> None:
    """Convert a single parquet file (CUI,llm_output) into BigBio JSON."""
    df = pl.read_parquet(parquet)
    records = dataframe_to_bigbio(df)
    _write_json(records, json_out)
    typer.echo(f"✅ Wrote {json_out} ({len(records)} records)")


def write_bigbio_json(df: pl.DataFrame, out_path: Path) -> int:
    """Public helper to filter failures, build BigBio records and write JSON.

    Returns number of records written.
    """
    records = dataframe_to_bigbio(df)
    _write_json(records, out_path)
    return len(records)


@app.command()
def from_hub(
    repo_id: str = typer.Option("Aremaki/SynCABEL", help="HuggingFace dataset repo id"),
    split: str = typer.Option("train", help="Dataset split to convert"),
    json_out: Path = typer.Option(..., help="Output BigBio JSON path"),
) -> None:
    """Load a dataset split from the HuggingFace Hub and convert to BigBio JSON."""
    ds = load_dataset(repo_id, split=split)
    df = pl.from_pandas(ds.to_pandas())  # type: ignore
    records = dataframe_to_bigbio(df)
    _write_json(records, json_out)
    typer.echo(f"✅ Wrote {json_out} ({len(records)} records) from {repo_id}:{split}")


if __name__ == "__main__":  # pragma: no cover
    app()
