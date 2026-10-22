"""GEP: original BERT embeddings + isotropic Gaussian noise + cosine nearest word.

Ported from HelpfulHarm/perturb.py. Heavy libraries are imported only on use.
"""
import os
from evaluation.common import as_list


def _dependencies():
    global torch, BertTokenizer, BertModel, tqdm
    import torch
    from transformers import BertTokenizer, BertModel
    from tqdm import tqdm


def perturb_wiki_entity(df, save_path=None, perturbation_strength=0.1):
    """
    Use BERT embeddings to perturb entities
    
    Parameters:
    - df: DataFrame containing the data to perturb
    - save_path: Path to save results
    - perturbation_strength: Strength of perturbation (default: 0.1), higher values create more diverse perturbations
    
    Returns:
    - DataFrame with perturbed entities
    """
    _dependencies()
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Set model to evaluation mode
    model.eval()
    
    # Pre-compute or load word embedding table
    embedding_cache_path = "Cache/Soft/bert_embeddings.pt"
    os.makedirs(os.path.dirname(embedding_cache_path), exist_ok=True)
    vocab_embeddings, vocab_words = get_vocab_embeddings(tokenizer, model, embedding_cache_path)
    
    # Create a new column to store entity pairs before and after perturbation
    df['entity_pairs'] = None
    
    # Process each row
    for idx in tqdm(range(len(df)), desc="Perturbing entities"):
        row = df.iloc[idx]
        entities = as_list(row['entities'])
        text = row['text']
        entity_pairs = []
        
        try:
            for entity in entities:
                # Get entity embedding
                inputs = tokenizer(entity, return_tensors='pt', padding=True, truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    entity_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                
                # Perturb in embedding space
                # Add random noise with controlled strength
                perturbation = torch.randn_like(entity_embedding) * perturbation_strength
                perturbed_embedding = entity_embedding + perturbation
                
                # Find the closest word to the perturbed embedding
                perturbed_entity = get_similar_words(perturbed_embedding, vocab_embeddings, vocab_words)
                
                # Store entity pair
                entity_pairs.append((entity, perturbed_entity))
                print(f"Entity: {entity} -> Perturbed: {perturbed_entity}")
            
            # Update DataFrame
            df.at[idx, 'entity_pairs'] = entity_pairs
            
        except Exception as e:
            print(f"Error processing record {idx}: {e}")
            # Keep track of failed records
            if 'processing_failed' not in df.columns:
                df['processing_failed'] = False
            df.at[idx, 'processing_failed'] = True
    
    # Save final results
    if save_path:
        df.to_json(save_path, orient="records", force_ascii=False, indent=4)
        print(f"All perturbed data saved to {save_path}")
    
    return df

def get_vocab_embeddings(tokenizer, model, cache_path):
    """
    Pre-compute or load embeddings for all words in BERT vocabulary
    Use cache to avoid repeated computation
    """
    _dependencies()
    if os.path.exists(cache_path):
        print("Loading cached word embeddings...")
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cache['embeddings'], cache['words']
    
    print("Pre-computing vocabulary embeddings...")
    vocab = list(tokenizer.vocab.keys())
    # Filter out special tokens and very short words
    filtered_vocab = [w for w in vocab if not w.startswith('[') and len(w) > 2]
    
    # Compute embeddings in batches to avoid memory overflow
    batch_size = 64
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(filtered_vocab), batch_size)):
            batch = filtered_vocab[i:i+batch_size]
            batch_inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**batch_inputs)
            
            # Get embedding for each word (using CLS token embedding)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(batch_embeddings)
    
    # Concatenate embeddings from all batches
    all_embeddings = torch.cat(embeddings, dim=0)
    
    # Save cache
    cache = {
        'embeddings': all_embeddings,
        'words': filtered_vocab
    }
    torch.save(cache, cache_path)
    
    return all_embeddings, filtered_vocab

def get_similar_words(perturbed_embedding, vocab_embeddings, vocab_words):
    """
    Find the most similar word based on a perturbed embedding, using pre-computed embedding table for speed
    
    Parameters:
    - perturbed_embedding: The perturbed embedding vector
    - vocab_embeddings: Pre-computed vocabulary embeddings
    - vocab_words: Corresponding word list
    
    Returns:
    - The most similar word
    """
    _dependencies()
    # Calculate cosine similarity between perturbed embedding and all vocabulary words
    similarities = torch.nn.functional.cosine_similarity(
        perturbed_embedding.unsqueeze(0), 
        vocab_embeddings
    )
    
    # Find the word with highest similarity
    most_similar_idx = torch.argmax(similarities).item()
    most_similar_word = vocab_words[most_similar_idx]
    
    return most_similar_word



def main(argv=None):
    import argparse
    from evaluation.common import check_paths, read_records, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/sampled_wikientities.json")
    parser.add_argument("--output", default="data/perturbed_data.json")
    parser.add_argument("--strength", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    check_paths(args.input, args.output)
    if args.strength < 0:
        parser.error("strength must be non-negative")
    import pandas as pd
    import torch
    torch.manual_seed(args.seed)
    frame = pd.DataFrame(read_records(args.input))
    if frame.empty:
        write_json(args.output, [])
        return 0
    result = perturb_wiki_entity(frame, perturbation_strength=args.strength)
    # DataFrame conversion normalizes missing optional fields to JSON null.
    import json
    write_json(args.output, json.loads(result.to_json(orient="records")))
    failed = result.get("processing_failed")
    return int(failed is not None and failed.fillna(False).any())


if __name__ == "__main__":
    raise SystemExit(main())

