import zipfile

import polars as pl
from tqdm import tqdm

# Prepare UMLS data for SynCABEL

# UMLS Codes
umls_codes = {"CUI": []}
path = "<YOUR_UMLS_PATH>.zip"
with zipfile.ZipFile(path) as zip_file:
    with zip_file.open("UMLS/MRCONSO.RRF", mode="r") as file:
        lines = file.readlines()
        print(f"ALL UMLS: {len(lines)} concepts")
        for line in tqdm(lines):
            line = str(line)[2:-3].split("|")
            umls_codes["CUI"].append(line[0])

umls_df = pl.DataFrame(umls_codes).unique()
umls_df.write_parquet("data/UMLS/umls_codes.parquet")


# UMLS Semantic
umls_semantic = {"CUI": [], "SEM_CODE": [], "TREE_CODE": [], "SEM_NAME": []}
with zipfile.ZipFile(path) as zip_file:
    with zip_file.open("UMLS/MRSTY.RRF", mode="r") as file:
        lines = file.readlines()
        for line in tqdm(lines):
            line = str(line)[2:-3].split("|")
            umls_semantic["CUI"].append(line[0])
            umls_semantic["SEM_CODE"].append(line[1])
            umls_semantic["TREE_CODE"].append(line[2])
            umls_semantic["SEM_NAME"].append(line[3])
umls_semantic_df = pl.DataFrame(umls_semantic).unique()

# Define the Semantic Info
data = [
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

# Create the DataFrame
df_sem_info = pl.DataFrame(data, schema=["CATEGORY", "GROUP", "SEM_CODE", "SEM_NAME"])

# Join the semantic info with the UMLS semantic DataFrame
umls_semantic_df = umls_semantic_df.join(
    df_sem_info, on=["SEM_CODE", "SEM_NAME"], how="left"
)

# Save the DataFrame to a Parquet file
umls_semantic_df.write_parquet("data/UMLS/umls_semantic.parquet")

# Prepare UMLS Definitions
umls_def = {"CUI": [], "DEF": []}
path = "UMLS/UMLS_2014_QUAERO.zip"
with zipfile.ZipFile(path) as zip_file:
    with zip_file.open("UMLS/MRDEF.RRF", mode="r") as file:
        lines = file.readlines()
        print(f"ALL UMLS 2014AA: {len(lines)} concepts")
        for line in tqdm(lines):
            line = str(line)[2:-3].split("|")
            umls_def["CUI"].append(line[0])
            umls_def["DEF"].append(line[5])
umls_def_df = pl.DataFrame(umls_def).unique()

umls_def_df = umls_def_df.group_by("CUI").agg([
    pl.col("DEF").unique(),
])

umls_def_df.write_parquet("data/UMLS/umls_def.parquet")

# UMLS Synonyms
umls_syn_df = []
umls_main_title = {"CUI": [], "UMLS_Title_main": []}
umls_fr_title = {"CUI": [], "UMLS_Title_fr": []}
umls_en_title = {"CUI": [], "UMLS_Title_en": []}
umls_fr_syn = {"CUI": [], "UMLS_alias_fr": []}
umls_en_syn = {"CUI": [], "UMLS_alias_en": []}
with zipfile.ZipFile(path) as zip_file:
    with zip_file.open("UMLS/MRCONSO.RRF", mode="r") as file:
        lines = file.readlines()
        print(f"ALL UMLS 2014AA: {len(lines)} concepts")
        for line in tqdm(lines):
            line = str(line)[2:-3].split("|")
            if line[11] == "MTH":
                umls_main_title["CUI"].append(line[0])
                umls_main_title["UMLS_Title_main"].append(
                    line[14]
                    .encode("utf-8")
                    .decode("unicode_escape")
                    .encode("latin1")
                    .decode("utf-8")
                )
            elif line[1] == "FRE":
                if line[2] == "P":
                    umls_fr_title["CUI"].append(line[0])
                    umls_fr_title["UMLS_Title_fr"].append(
                        line[14]
                        .encode("utf-8")
                        .decode("unicode_escape")
                        .encode("latin1")
                        .decode("utf-8")
                    )
                else:
                    umls_fr_syn["CUI"].append(line[0])
                    umls_fr_syn["UMLS_alias_fr"].append(
                        line[14]
                        .encode("utf-8")
                        .decode("unicode_escape")
                        .encode("latin1")
                        .decode("utf-8")
                    )
            elif line[1] == "ENG":
                if line[2] == "P":
                    umls_en_title["CUI"].append(line[0])
                    umls_en_title["UMLS_Title_en"].append(
                        line[14]
                        .encode("utf-8")
                        .decode("unicode_escape")
                        .encode("latin1")
                        .decode("utf-8")
                    )
                else:
                    umls_en_syn["CUI"].append(line[0])
                    umls_en_syn["UMLS_alias_en"].append(
                        line[14]
                        .encode("utf-8")
                        .decode("unicode_escape")
                        .encode("latin1")
                        .decode("utf-8")
                    )

umls_syn_fr_df = pl.DataFrame(umls_fr_syn).unique()
umls_en_syn_df = pl.DataFrame(umls_en_syn).unique()
umls_fr_title_df = pl.DataFrame(umls_fr_title).unique()
umls_en_title_df = pl.DataFrame(umls_en_title).unique()
umls_main_title_df = pl.DataFrame(umls_main_title).unique()

umls_title_df = (
    umls_fr_title_df.join(umls_en_title_df, how="full", on="CUI", coalesce=True)
    .join(umls_main_title_df, how="full", on="CUI", coalesce=True)
    .group_by("CUI")
    .agg([
        pl.col("UMLS_Title_main").unique(),
        pl.col("UMLS_Title_fr").unique(),
        pl.col("UMLS_Title_en").unique(),
    ])
)

umls_syn_df = (
    umls_syn_fr_df.join(umls_en_syn_df, how="full", on="CUI", coalesce=True)
    .group_by("CUI")
    .agg([
        pl.col("UMLS_alias_fr").unique(),
        pl.col("UMLS_alias_en").unique(),
    ])
)

umls_title_syn_df = umls_title_df.join(umls_syn_df, how="full", on="CUI", coalesce=True)

umls_title_syn_df.write_parquet("data/UMLS/umls_title_syn.parquet")
