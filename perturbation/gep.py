from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


BERT_MODEL: str = "bert-base-uncased"


def load_vocab_embeddings(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    cache_path: str,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, List[str]]:
    if os.path.exists(cache_path):
        cache = torch.load(cache_path)
        return cache["embeddings"], cache["words"]
    vocab = list(tokenizer.get_vocab().keys())
    vocab = [w for w in vocab if not w.startswith("[") and len(w) > 2]
    embeddings: List[torch.Tensor] = []
    for i in tqdm(range(0, len(vocab), batch_size), desc="Embedding vocab"):
        batch = vocab[i : i + batch_size]
        batch_inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
        with torch.no_grad():
            outputs = model(**batch_inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.append(batch_embeddings.detach().cpu())
    all_embeddings = torch.cat(embeddings, dim=0)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"embeddings": all_embeddings, "words": vocab}, cache_path)
    return all_embeddings, vocab


def find_nearest_token(
    embedding: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    vocab_words: List[str],
) -> str:
    similarities = torch.nn.functional.cosine_similarity(
        embedding.unsqueeze(0), vocab_embeddings
    )
    best_idx = int(torch.argmax(similarities).item())
    return vocab_words[best_idx]


def perturb_entities_gep(
    df: pd.DataFrame,
    strength: float = 0.1,
    cache_path: str = "cache/bert_embeddings.pt",
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    vocab_embeddings, vocab_words = load_vocab_embeddings(tokenizer, model, cache_path)
    df = df.copy()
    df["entity_pairs"] = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="GEP perturbation"):
        entities = row["entities"]
        entity_pairs: List[Tuple[str, str]] = []
        for entity in entities:
            inputs = tokenizer(entity, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            entity_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            noise = torch.randn_like(entity_embedding) * strength
            perturbed_embedding = entity_embedding + noise
            perturbed_entity = find_nearest_token(
                perturbed_embedding.detach().cpu(), vocab_embeddings, vocab_words
            )
            entity_pairs.append((entity, perturbed_entity))
        df.at[idx, "entity_pairs"] = entity_pairs

    return df
