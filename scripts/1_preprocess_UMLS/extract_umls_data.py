"""Typer CLI to extract and prepare UMLS resources for SynCABEL.

Subcommands:
  codes       -> Extract unique CUIs from MRCONSO.RRF
  semantic    -> Extract semantic types from MRSTY.RRF and enrich with category info
  definitions -> Extract definitions from MRDEF.RRF (older QUAERO release used here)
  synonyms    -> Extract preferred titles and synonyms (EN / FR / main)
  all         -> Run the full pipeline

Examples:
  python extract_umls_data.py all --umls-zip /path/UMLS_2025AA.zip
  python extract_umls_data.py codes --umls-zip /path/UMLS_2017AB.zip
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path

import polars as pl
import typer
from tqdm import tqdm

app = typer.Typer(help="Extract UMLS data components into parquet files.")


# Static semantic group mapping (kept identical to original list)
SEMANTIC_INFO = [
    ("ACTI", "Activities & Behaviors", "T052", "Activity"),
    ("ACTI", "Activities & Behaviors", "T053", "Behavior"),
    ("ACTI", "Activities & Behaviors", "T056", "Daily or Recreational Activity"),
    ("ACTI", "Activities & Behaviors", "T051", "Event"),
    ("ACTI", "Activities & Behaviors", "T064", "Governmental or Regulatory Activity"),
    ("ACTI", "Activities & Behaviors", "T055", "Individual Behavior"),
    ("ACTI", "Activities & Behaviors", "T066", "Machine Activity"),
    ("ACTI", "Activities & Behaviors", "T057", "Occupational Activity"),
    ("ACTI", "Activities & Behaviors", "T054", "Social Behavior"),
    ("ANAT", "Anatomy", "T017", "Anatomical Structure"),
    ("ANAT", "Anatomy", "T029", "Body Location or Region"),
    ("ANAT", "Anatomy", "T023", "Body Part, Organ, or Organ Component"),
    ("ANAT", "Anatomy", "T030", "Body Space or Junction"),
    ("ANAT", "Anatomy", "T031", "Body Substance"),
    ("ANAT", "Anatomy", "T022", "Body System"),
    ("ANAT", "Anatomy", "T025", "Cell"),
    ("ANAT", "Anatomy", "T026", "Cell Component"),
    ("ANAT", "Anatomy", "T018", "Embryonic Structure"),
    ("ANAT", "Anatomy", "T021", "Fully Formed Anatomical Structure"),
    ("ANAT", "Anatomy", "T024", "Tissue"),
    ("CHEM", "Chemicals & Drugs", "T116", "Amino Acid, Peptide, or Protein"),
    ("CHEM", "Chemicals & Drugs", "T195", "Antibiotic"),
    ("CHEM", "Chemicals & Drugs", "T123", "Biologically Active Substance"),
    ("CHEM", "Chemicals & Drugs", "T122", "Biomedical or Dental Material"),
    ("CHEM", "Chemicals & Drugs", "T103", "Chemical"),
    ("CHEM", "Chemicals & Drugs", "T120", "Chemical Viewed Functionally"),
    ("CHEM", "Chemicals & Drugs", "T104", "Chemical Viewed Structurally"),
    ("CHEM", "Chemicals & Drugs", "T200", "Clinical Drug"),
    ("CHEM", "Chemicals & Drugs", "T196", "Element, Ion, or Isotope"),
    ("CHEM", "Chemicals & Drugs", "T126", "Enzyme"),
    ("CHEM", "Chemicals & Drugs", "T131", "Hazardous or Poisonous Substance"),
    ("CHEM", "Chemicals & Drugs", "T125", "Hormone"),
    ("CHEM", "Chemicals & Drugs", "T129", "Immunologic Factor"),
    ("CHEM", "Chemicals & Drugs", "T130", "Indicator, Reagent, or Diagnostic Aid"),
    ("CHEM", "Chemicals & Drugs", "T197", "Inorganic Chemical"),
    ("CHEM", "Chemicals & Drugs", "T114", "Nucleic Acid, Nucleoside, or Nucleotide"),
    ("CHEM", "Chemicals & Drugs", "T109", "Organic Chemical"),
    ("CHEM", "Chemicals & Drugs", "T121", "Pharmacologic Substance"),
    ("CHEM", "Chemicals & Drugs", "T192", "Receptor"),
    ("CHEM", "Chemicals & Drugs", "T127", "Vitamin"),
    ("CONC", "Concepts & Ideas", "T185", "Classification"),
    ("CONC", "Concepts & Ideas", "T077", "Conceptual Entity"),
    ("CONC", "Concepts & Ideas", "T169", "Functional Concept"),
    ("CONC", "Concepts & Ideas", "T102", "Group Attribute"),
    ("CONC", "Concepts & Ideas", "T078", "Idea or Concept"),
    ("CONC", "Concepts & Ideas", "T170", "Intellectual Product"),
    ("CONC", "Concepts & Ideas", "T171", "Language"),
    ("CONC", "Concepts & Ideas", "T080", "Qualitative Concept"),
    ("CONC", "Concepts & Ideas", "T081", "Quantitative Concept"),
    ("CONC", "Concepts & Ideas", "T089", "Regulation or Law"),
    ("CONC", "Concepts & Ideas", "T082", "Spatial Concept"),
    ("CONC", "Concepts & Ideas", "T079", "Temporal Concept"),
    ("DEVI", "Devices", "T203", "Drug Delivery Device"),
    ("DEVI", "Devices", "T074", "Medical Device"),
    ("DEVI", "Devices", "T075", "Research Device"),
    ("DISO", "Disorders", "T020", "Acquired Abnormality"),
    ("DISO", "Disorders", "T190", "Anatomical Abnormality"),
    ("DISO", "Disorders", "T049", "Cell or Molecular Dysfunction"),
    ("DISO", "Disorders", "T019", "Congenital Abnormality"),
    ("DISO", "Disorders", "T047", "Disease or Syndrome"),
    ("DISO", "Disorders", "T050", "Experimental Model of Disease"),
    ("DISO", "Disorders", "T033", "Finding"),
    ("DISO", "Disorders", "T037", "Injury or Poisoning"),
    ("DISO", "Disorders", "T048", "Mental or Behavioral Dysfunction"),
    ("DISO", "Disorders", "T191", "Neoplastic Process"),
    ("DISO", "Disorders", "T046", "Pathologic Function"),
    ("DISO", "Disorders", "T184", "Sign or Symptom"),
    ("GENE", "Genes & Molecular Sequences", "T087", "Amino Acid Sequence"),
    ("GENE", "Genes & Molecular Sequences", "T088", "Carbohydrate Sequence"),
    ("GENE", "Genes & Molecular Sequences", "T028", "Gene or Genome"),
    ("GENE", "Genes & Molecular Sequences", "T085", "Molecular Sequence"),
    ("GENE", "Genes & Molecular Sequences", "T086", "Nucleotide Sequence"),
    ("GEOG", "Geographic Areas", "T083", "Geographic Area"),
    ("LIVB", "Living Beings", "T100", "Age Group"),
    ("LIVB", "Living Beings", "T011", "Amphibian"),
    ("LIVB", "Living Beings", "T008", "Animal"),
    ("LIVB", "Living Beings", "T194", "Archaeon"),
    ("LIVB", "Living Beings", "T007", "Bacterium"),
    ("LIVB", "Living Beings", "T012", "Bird"),
    ("LIVB", "Living Beings", "T204", "Eukaryote"),
    ("LIVB", "Living Beings", "T099", "Family Group"),
    ("LIVB", "Living Beings", "T013", "Fish"),
    ("LIVB", "Living Beings", "T004", "Fungus"),
    ("LIVB", "Living Beings", "T096", "Group"),
    ("LIVB", "Living Beings", "T016", "Human"),
    ("LIVB", "Living Beings", "T015", "Mammal"),
    ("LIVB", "Living Beings", "T001", "Organism"),
    ("LIVB", "Living Beings", "T101", "Patient or Disabled Group"),
    ("LIVB", "Living Beings", "T002", "Plant"),
    ("LIVB", "Living Beings", "T098", "Population Group"),
    ("LIVB", "Living Beings", "T097", "Professional or Occupational Group"),
    ("LIVB", "Living Beings", "T014", "Reptile"),
    ("LIVB", "Living Beings", "T010", "Vertebrate"),
    ("LIVB", "Living Beings", "T005", "Virus"),
    ("OBJC", "Objects", "T071", "Entity"),
    ("OBJC", "Objects", "T168", "Food"),
    ("OBJC", "Objects", "T073", "Manufactured Object"),
    ("OBJC", "Objects", "T072", "Physical Object"),
    ("OBJC", "Objects", "T167", "Substance"),
    ("OCCU", "Occupations", "T091", "Biomedical Occupation or Discipline"),
    ("OCCU", "Occupations", "T090", "Occupation or Discipline"),
    ("ORGA", "Organizations", "T093", "Health Care Related Organization"),
    ("ORGA", "Organizations", "T092", "Organization"),
    ("ORGA", "Organizations", "T094", "Professional Society"),
    ("ORGA", "Organizations", "T095", "Self-help or Relief Organization"),
    ("PHEN", "Phenomena", "T038", "Biologic Function"),
    ("PHEN", "Phenomena", "T069", "Environmental Effect of Humans"),
    ("PHEN", "Phenomena", "T068", "Human-caused Phenomenon or Process"),
    ("PHEN", "Phenomena", "T034", "Laboratory or Test Result"),
    ("PHEN", "Phenomena", "T070", "Natural Phenomenon or Process"),
    ("PHEN", "Phenomena", "T067", "Phenomenon or Process"),
    ("PHYS", "Physiology", "T043", "Cell Function"),
    ("PHYS", "Physiology", "T201", "Clinical Attribute"),
    ("PHYS", "Physiology", "T045", "Genetic Function"),
    ("PHYS", "Physiology", "T041", "Mental Process"),
    ("PHYS", "Physiology", "T044", "Molecular Function"),
    ("PHYS", "Physiology", "T032", "Organism Attribute"),
    ("PHYS", "Physiology", "T040", "Organism Function"),
    ("PHYS", "Physiology", "T042", "Organ or Tissue Function"),
    ("PHYS", "Physiology", "T039", "Physiologic Function"),
    ("PROC", "Procedures", "T060", "Diagnostic Procedure"),
    ("PROC", "Procedures", "T065", "Educational Activity"),
    ("PROC", "Procedures", "T058", "Health Care Activity"),
    ("PROC", "Procedures", "T059", "Laboratory Procedure"),
    ("PROC", "Procedures", "T063", "Molecular Biology Research Technique"),
    ("PROC", "Procedures", "T062", "Research Activity"),
    ("PROC", "Procedures", "T061", "Therapeutic or Preventive Procedure"),
]


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _iter_rrf(zip_path: Path, inner_path: str) -> Iterable[list[str]]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(inner_path, mode="r") as fh:
            for raw in fh:  # streaming, not loading entire file
                parts = str(raw)[2:-3].split("|")  # preserve original parsing behavior
                yield parts


@app.command()
def codes(
    umls_zip: Path = typer.Option(
        ...,
        exists=True,
        readable=True,
        help="Path to UMLS release zip containing MRCONSO.RRF inside UMLS/",
    ),
    out_dir: Path = typer.Option(Path("data/UMLS"), help="Output directory"),
) -> None:
    """Extract unique CUIs to umls_codes.parquet."""
    _ensure_out_dir(out_dir)
    cuis: list[str] = []
    for parts in tqdm(_iter_rrf(umls_zip, "UMLS/MRCONSO.RRF"), desc="CUIs"):
        if parts and parts[0]:
            cuis.append(parts[0])
    pl.DataFrame({"CUI": cuis}).unique().write_parquet(out_dir / "umls_codes.parquet")
    typer.echo(f"Saved {out_dir / 'umls_codes.parquet'}")


@app.command()
def semantic(
    umls_zip: Path = typer.Option(
        ...,
        exists=True,
        readable=True,
        help="Path to UMLS release zip containing MRSTY.RRF",
    ),
    out_dir: Path = typer.Option(Path("data/UMLS"), help="Output directory"),
) -> None:
    """Extract semantic types and enrich with coarse category mapping."""
    _ensure_out_dir(out_dir)
    rows = {"CUI": [], "SEM_CODE": [], "TREE_CODE": [], "SEM_NAME": []}
    for parts in tqdm(_iter_rrf(umls_zip, "UMLS/MRSTY.RRF"), desc="Semantic"):
        if len(parts) >= 4:
            rows["CUI"].append(parts[0])
            rows["SEM_CODE"].append(parts[1])
            rows["TREE_CODE"].append(parts[2])
            rows["SEM_NAME"].append(parts[3])
    df = pl.DataFrame(rows).unique()
    df_sem_info = pl.DataFrame(
        SEMANTIC_INFO, schema=["CATEGORY", "GROUP", "SEM_CODE", "SEM_NAME"]
    )
    df = df.join(df_sem_info, on=["SEM_CODE", "SEM_NAME"], how="left")
    out_file = out_dir / "umls_semantic.parquet"
    df.write_parquet(out_file)
    typer.echo(f"Saved {out_file}")


@app.command()
def definitions(
    umls_zip: Path = typer.Option(
        ..., exists=True, readable=True, help="Path to UMLS zip containing MRDEF.RRF"
    ),
    out_dir: Path = typer.Option(Path("data/UMLS"), help="Output directory"),
) -> None:
    """Extract definitions grouped per CUI to umls_def.parquet."""
    _ensure_out_dir(out_dir)
    rows = {"CUI": [], "DEF": []}
    for parts in tqdm(_iter_rrf(umls_zip, "UMLS/MRDEF.RRF"), desc="Definitions"):
        if len(parts) > 5:
            rows["CUI"].append(parts[0])
            rows["DEF"].append(parts[5])
    df = pl.DataFrame(rows).unique().group_by("CUI").agg([pl.col("DEF").unique()])
    out_file = out_dir / "umls_def.parquet"
    df.write_parquet(out_file)
    typer.echo(f"Saved {out_file}")


def _decode_term(term: str) -> str:
    # Preserve original multi-step decoding pipeline
    return (
        term.encode("utf-8")
        .decode("unicode_escape")
        .encode("latin1", errors="ignore")
        .decode("utf-8", errors="ignore")
    )


@app.command()
def synonyms(
    umls_zip: Path = typer.Option(
        ..., exists=True, readable=True, help="Path to UMLS zip containing MRCONSO.RRF"
    ),
    out_dir: Path = typer.Option(Path("data/UMLS"), help="Output directory"),
) -> None:
    """Extract preferred titles (EN/FR/Main) and synonyms into umls_title_syn.parquet."""
    _ensure_out_dir(out_dir)
    main_title = {"CUI": [], "UMLS_Title_main": []}
    fr_title = {"CUI": [], "UMLS_Title_fr": []}
    en_title = {"CUI": [], "UMLS_Title_en": []}
    fr_syn = {"CUI": [], "UMLS_alias_fr": []}
    en_syn = {"CUI": [], "UMLS_alias_en": []}
    for parts in tqdm(_iter_rrf(umls_zip, "UMLS/MRCONSO.RRF"), desc="Synonyms"):
        if len(parts) < 15:
            continue
        cui = parts[0]
        lat = parts[1]
        ts = parts[2]
        sab = parts[11]
        term = _decode_term(parts[14])
        if sab == "MTH":
            main_title["CUI"].append(cui)
            main_title["UMLS_Title_main"].append(term)
        elif lat == "FRE":
            if ts == "P":
                fr_title["CUI"].append(cui)
                fr_title["UMLS_Title_fr"].append(term)
            else:
                fr_syn["CUI"].append(cui)
                fr_syn["UMLS_alias_fr"].append(term)
        elif lat == "ENG":
            if ts == "P":
                en_title["CUI"].append(cui)
                en_title["UMLS_Title_en"].append(term)
            else:
                en_syn["CUI"].append(cui)
                en_syn["UMLS_alias_en"].append(term)

    fr_syn_df = pl.DataFrame(fr_syn).unique()
    en_syn_df = pl.DataFrame(en_syn).unique()
    fr_title_df = pl.DataFrame(fr_title).unique()
    en_title_df = pl.DataFrame(en_title).unique()
    main_title_df = pl.DataFrame(main_title).unique()

    title_df = (
        fr_title_df.join(en_title_df, how="full", on="CUI", coalesce=True)
        .join(main_title_df, how="full", on="CUI", coalesce=True)
        .group_by("CUI")
        .agg([
            pl.col("UMLS_Title_main").unique(),
            pl.col("UMLS_Title_fr").unique(),
            pl.col("UMLS_Title_en").unique(),
        ])
    )
    syn_df = (
        fr_syn_df.join(en_syn_df, how="full", on="CUI", coalesce=True)
        .group_by("CUI")
        .agg([
            pl.col("UMLS_alias_fr").unique(),
            pl.col("UMLS_alias_en").unique(),
        ])
    )
    title_syn_df = title_df.join(syn_df, how="full", on="CUI", coalesce=True)
    out_file = out_dir / "umls_title_syn.parquet"
    title_syn_df.write_parquet(out_file)
    typer.echo(f"Saved {out_file}")


@app.command()
def all(
    umls_zip: Path = typer.Option(
        ...,
        exists=True,
        readable=True,
        help="Path to UMLS release zip (MRCONSO.RRF & MRSTY.RRF)",
    ),
    out_dir: Path = typer.Option(Path("data/UMLS"), help="Output directory"),
) -> None:
    """Run full extraction pipeline."""
    codes(umls_zip=umls_zip, out_dir=out_dir)
    semantic(umls_zip=umls_zip, out_dir=out_dir)
    definitions(umls_zip=umls_zip, out_dir=out_dir)
    synonyms(umls_zip=umls_zip, out_dir=out_dir)


if __name__ == "__main__":  # pragma: no cover
    app()
