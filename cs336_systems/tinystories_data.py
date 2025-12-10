"""
Prepare TinyStories for training: download, tokenize, save as .pt (1D token ids).
Model vocab_size is 10000; we map tokenizer ids into 0..9999.
"""

from __future__ import annotations

import os
from pathlib import Path


def build_tinystories_pt(
    output_path: str | Path = "tinystories.pt",
    seq_len: int = 256,
    max_samples: int | None = None,
    vocab_size: int = 10000,
    split: str = "train",
) -> Path:
    """
    Download TinyStories, tokenize with GPT-2 tokenizer (ids % vocab_size), save 1D tensor.
    Returns path to the saved file.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("pip install transformers")

    output_path = Path(output_path)
    print("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split=split, trust_remote_code=True)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    text_key = "text" if "text" in ds.column_names else "story"
    texts = [ex[text_key] for ex in ds]

    print("Tokenizing...")
    tok = AutoTokenizer.from_pretrained("gpt2")
    # Allow long stories (we chunk by seq_len later); avoids "sequence length > 1024" warning
    tok.model_max_length = 1_000_000
    all_ids = []
    for t in texts:
        enc = tok.encode(t, add_special_tokens=False)
        all_ids.extend(enc)

    # Map to model vocab 0..vocab_size-1
    import torch
    arr = torch.tensor(all_ids, dtype=torch.long)
    arr = arr % vocab_size
    # Truncate to full chunks of (seq_len + 1)
    chunk = seq_len + 1
    n_chunks = arr.numel() // chunk
    arr = arr[: n_chunks * chunk]
    arr = arr.view(n_chunks, chunk)
    torch.save(arr, output_path)
    print(f"Saved {arr.shape[0]} chunks (seq_len+1={chunk}) to {output_path}")
    return output_path
