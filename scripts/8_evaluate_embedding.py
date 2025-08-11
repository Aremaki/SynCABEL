import pickle

import polars as pl
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"  # Load model directly


# Load Model
model_name = "SapBERT-from-PubMedBERT-fulltext"
model_path = f"models/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)


# Create Embedding
def get_bert_embed(
    phrase_list,
    model,
    tokenizer,
    normalize=True,
    summary_method="CLS",
    tqdm_bar=True,
    batch_size=128,
):
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(
            tokenizer.encode_plus(
                phrase,
                max_length=32,
                add_special_tokens=True,
                truncation=True,
                pad_to_max_length=True,
            )["input_ids"]
        )
    model.eval()

    count = len(input_ids)
    now_count = 0
    with torch.no_grad():
        if tqdm_bar:
            pbar = tqdm.tqdm(total=count)
        while now_count < count:
            input_gpu_0 = torch.LongTensor(
                input_ids[now_count : min(now_count + batch_size, count)]
            ).to(device)
            if summary_method == "CLS":
                embed = model(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(model(input_gpu_0)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(embed, p=2, dim=1, keepdim=True).clamp(  # type: ignore
                    min=1e-12
                )
                embed = embed / embed_norm  # type: ignore
            if now_count == 0:
                output = embed  # type: ignore
            else:
                output = torch.cat((output, embed), dim=0)  # type: ignore
            if tqdm_bar:
                pbar.update(min(now_count + batch_size, count) - now_count)  # type: ignore
            now_count = min(now_count + batch_size, count)
        if tqdm_bar:
            pbar.close()  # type: ignore
    return output  # type: ignore


# UMLS Embeddings
legal_umls_token_all = pl.read_parquet(
    "../data/legal_umls_token_2017_short_syn_all_main.parquet"
)
Syn_to_annotation = legal_umls_token_all.with_columns(
    Syn=pl.col("Entity").str.split(" of type ").list.get(0)
)
Syn_to_CUI = {}
for category in Syn_to_annotation["SEM_NAME_MM"].unique():
    Syn_to_CUI[category] = dict(
        Syn_to_annotation.filter(pl.col("SEM_NAME_MM") == category)
        .group_by("Syn")
        .agg([pl.col("CUI").unique()])
        .iter_rows()
    )

# Process
umls_embedding = {}
for category in Syn_to_CUI.keys():
    print(category)
    cat_syn = list(Syn_to_CUI[category].keys())
    umls_embedding[category] = get_bert_embed(cat_syn, model, tokenizer)

with open(
    f"../data/UMLS_embeddings/{model_name}/medmentions_umls_2017_embeddings.pkl", "wb"
) as file:
    pickle.dump(umls_embedding, file, protocol=-1)
with open(
    f"../data/UMLS_embeddings/{model_name}/medmentions_umls_2017_syn_to_cui.pkl", "wb"
) as file:
    pickle.dump(Syn_to_CUI, file, protocol=-1)
