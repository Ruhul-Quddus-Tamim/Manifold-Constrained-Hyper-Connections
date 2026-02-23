"""Data loading, tokenization, and dataset classes for the mHC experiment."""

import random
from typing import List, Dict, Tuple, Optional

import torch

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MIN_LEN = 5
MAX_LEN = 15
TRAIN_SAMPLES = 20000
TEST_SAMPLES = 2000
BATCH_SIZE = 64
LM_SEQ_LEN = 128
MAX_SEQ_LEN = 3 + MAX_LEN + 4  # <s> + content + </s>


class CharTokenizer:
    """A simple character-level tokenizer."""

    def __init__(self, corpus: List[str]):
        self.chars = sorted(list(set("".join(corpus))))
        self.special_tokens = ["<pad>", "<s>", "</s>"]
        self.vocab = self.special_tokens + self.chars
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(
            [self.idx_to_char[t] for t in tokens if t != self.char_to_idx["<pad>"]]
        )


class CopyDataset(torch.utils.data.Dataset):
    """Generates pairs of (sequence, sequence) for the copying task."""

    def __init__(self, num_samples: int, min_len: int, max_len: int, alphabet: str):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.alphabet = list(alphabet)
        self.data = self._generate_data()

    def _generate_data(self) -> List[str]:
        samples = []
        for _ in range(self.num_samples):
            seq_len = random.randint(self.min_len, self.max_len)
            if seq_len <= len(self.alphabet):
                chars = random.sample(self.alphabet, seq_len)
            else:
                chars = random.choices(self.alphabet, k=seq_len)
            sequence = "".join(chars)
            samples.append(f"<s>{sequence}</s>")
        return samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]


def collate_fn(
    batch: List[str], tokenizer: CharTokenizer, max_len: int
) -> Dict[str, torch.Tensor]:
    """Collate batch for DataLoader."""
    pad_idx = tokenizer.char_to_idx["<pad>"]
    inputs, targets = [], []
    for item in batch:
        tokens = tokenizer.encode(item)
        tokens = tokens[:max_len]
        padded_tokens = tokens + [pad_idx] * (max_len - len(tokens))
        inputs.append(padded_tokens[:-1])
        targets.append(padded_tokens[1:])
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long),
        "target_ids": torch.tensor(targets, dtype=torch.long),
    }


def load_hf_dataset(
    dataset_id: str,
    config: Optional[str] = None,
    split_ratio: float = 0.9,
) -> Tuple[str, str]:
    """Load Hugging Face dataset and return (train_text, test_text)."""
    if load_dataset is None:
        raise ImportError(
            "Hugging Face 'datasets' library required. Install with: pip install datasets"
        )

    try:
        if config:
            ds = load_dataset(dataset_id, config)
        else:
            ds = load_dataset(dataset_id)
    except (RuntimeError, TypeError):
        try:
            if config:
                ds = load_dataset(dataset_id, config, trust_remote_code=True)
            else:
                ds = load_dataset(dataset_id, trust_remote_code=True)
        except Exception:
            raise

    def _get_text_column(split) -> str:
        cols = split.column_names
        for candidate in ["text", "Text", "content", "sentence"]:
            if candidate in cols:
                return candidate
        return cols[0]

    def _extract_text(split, join_char: str = ""):
        col = _get_text_column(split)
        rows = split[col]
        if not rows:
            return ""
        if isinstance(rows[0], str):
            return join_char.join(r for r in rows if r is not None and str(r).strip())
        return join_char.join(str(r) for r in rows if r is not None and str(r).strip())

    if "train" in ds and "test" not in ds:
        if "validation" in ds:
            train_text = _extract_text(ds["train"], "\n")
            test_text = _extract_text(ds["validation"], "\n")
        else:
            full_text = _extract_text(ds["train"], "")
            if not full_text:
                col = _get_text_column(ds["train"])
                full_text = "".join(str(t) for t in ds["train"][col])
            split_idx = int(len(full_text) * split_ratio)
            train_text = full_text[:split_idx]
            test_text = full_text[split_idx:]
        return train_text, test_text

    if "train" in ds and "test" in ds:
        train_text = _extract_text(ds["train"], "\n")
        test_text = _extract_text(ds["test"], "\n")
        return train_text, test_text

    raise ValueError(f"Dataset {dataset_id} has no train or test split: {list(ds.keys())}")


class HuggingFaceCopyDataset(torch.utils.data.Dataset):
    """Creates copy-task samples from Hugging Face text by sampling random spans."""

    def __init__(
        self,
        text: str,
        num_samples: int,
        min_len: int,
        max_len: int,
    ):
        self.text = text
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.data = self._generate_data()

    def _generate_data(self) -> List[str]:
        samples = []
        text_len = len(self.text)
        if text_len < self.min_len:
            raise ValueError(
                f"Text length ({text_len}) must be >= min_len ({self.min_len})"
            )
        for _ in range(self.num_samples):
            span_len = random.randint(self.min_len, min(self.max_len, text_len))
            max_start = text_len - span_len
            start = random.randint(0, max(0, max_start))
            span = self.text[start : start + span_len]
            samples.append(f"<s>{span}</s>")
        return samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]


def create_dataloaders_from_hf(
    dataset_id: str,
    dataset_config: Optional[str] = None,
    split_ratio: float = 0.9,
    train_samples: int = TRAIN_SAMPLES,
    test_samples: int = TEST_SAMPLES,
    min_len: int = MIN_LEN,
    max_len: int = MAX_LEN,
    max_seq_len: int = MAX_SEQ_LEN,
    batch_size: int = BATCH_SIZE,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, CharTokenizer, int]:
    """Create train and test DataLoaders from a Hugging Face dataset."""
    train_text, test_text = load_hf_dataset(dataset_id, dataset_config, split_ratio)

    train_dataset = HuggingFaceCopyDataset(
        train_text, train_samples, min_len, max_len
    )
    test_dataset = HuggingFaceCopyDataset(
        test_text, test_samples, min_len, max_len
    )

    all_samples = train_dataset.data + test_dataset.data
    tokenizer = CharTokenizer(all_samples)
    pad_idx = tokenizer.char_to_idx["<pad>"]

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_seq_len),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_seq_len),
    )

    return train_loader, test_loader, tokenizer, pad_idx


def create_dataloaders(
    max_seq_len: int = MAX_SEQ_LEN,
    batch_size: int = BATCH_SIZE,
):
    """Create train and test DataLoaders, tokenizer, and pad_idx."""
    train_dataset = CopyDataset(TRAIN_SAMPLES, MIN_LEN, MAX_LEN, ALPHABET)
    test_dataset = CopyDataset(TEST_SAMPLES, MIN_LEN, MAX_LEN, ALPHABET)
    tokenizer = CharTokenizer(train_dataset.data)
    pad_idx = tokenizer.char_to_idx["<pad>"]

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_seq_len),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_seq_len),
    )

    return train_loader, test_loader, tokenizer, pad_idx


# ---------------------------------------------------------------------------
# Character-level Language Modeling (sliding window)
# ---------------------------------------------------------------------------

class CharLMTokenizer:
    """Character-level tokenizer for language modeling (no special start/end tokens)."""

    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab = ["<pad>"] + self.chars
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(self.idx_to_char.get(t, "") for t in tokens if t != 0)


class CharLMDataset(torch.utils.data.Dataset):
    """Fixed-length non-overlapping chunks for next-token prediction."""

    def __init__(self, token_ids: List[int], seq_len: int):
        self.seq_len = seq_len
        n_chunks = len(token_ids) // (seq_len + 1)
        self.chunks = [
            token_ids[i * (seq_len + 1) : i * (seq_len + 1) + seq_len + 1]
            for i in range(n_chunks)
        ]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        return {
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "target_ids": torch.tensor(chunk[1:], dtype=torch.long),
        }


def create_lm_dataloaders(
    dataset_id: str = "Trelis/tiny-shakespeare",
    dataset_config: Optional[str] = None,
    seq_len: int = LM_SEQ_LEN,
    batch_size: int = BATCH_SIZE,
    split_ratio: float = 0.9,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, CharLMTokenizer, int]:
    """Create train/test DataLoaders for character-level LM from a HF dataset."""
    train_text, test_text = load_hf_dataset(dataset_id, dataset_config, split_ratio)

    tokenizer = CharLMTokenizer(train_text + test_text)
    pad_idx = tokenizer.char_to_idx["<pad>"]

    train_ids = tokenizer.encode(train_text)
    test_ids = tokenizer.encode(test_text)

    train_dataset = CharLMDataset(train_ids, seq_len)
    test_dataset = CharLMDataset(test_ids, seq_len)

    print(f"  Train chunks: {len(train_dataset):,} | Test chunks: {len(test_dataset):,}")
    print(f"  Vocab size: {tokenizer.vocab_size} | Seq len: {seq_len}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )

    return train_loader, test_loader, tokenizer, pad_idx


def create_lm_dataloader_full(
    dataset_id: str = "Trelis/tiny-shakespeare",
    dataset_config: Optional[str] = None,
    seq_len: int = LM_SEQ_LEN,
    batch_size: int = BATCH_SIZE,
) -> Tuple[torch.utils.data.DataLoader, CharLMTokenizer, int]:
    """Create a single DataLoader using the entire dataset for training (no test split)."""
    train_text, test_text = load_hf_dataset(dataset_id, dataset_config, split_ratio=0.9)
    full_text = train_text + test_text

    tokenizer = CharLMTokenizer(full_text)
    pad_idx = tokenizer.char_to_idx["<pad>"]

    token_ids = tokenizer.encode(full_text)
    dataset = CharLMDataset(token_ids, seq_len)

    print(f"  Total chunks: {len(dataset):,} (full dataset, no held-out split)")
    print(f"  Vocab size: {tokenizer.vocab_size} | Seq len: {seq_len}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    return loader, tokenizer, pad_idx
