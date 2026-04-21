import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = './NM_TinyRNN/data/AB_behaviour/WS16'
SEQUENCE_LENGTH = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _encode_df(df):
    df['forced_choice'] = df['forced_choice'].astype(int)
    df['outcome']       = df['outcome'].astype(int)
    df['choice']        = df['choice'].astype('category').cat.codes.astype(int)
    df['good_poke']     = df['good_poke'].astype('category').cat.codes.astype(int)
    return df

REQUIRED = ['forced_choice', 'outcome', 'choice', 'good_poke']


# ── AB_Dataset (sequence/batch based) ─────────────────────────────────────────

class AB_Dataset(Dataset):
    """Splits sessions into fixed-length sequences. Supports batch_size > 1."""

    def __init__(self, data_path, sequence_length=SEQUENCE_LENGTH, device=DEVICE):
        self.device = device
        self.sequence_length = sequence_length
        self.subject_df = self._load(data_path)
        self.inputs, self.targets = self._make_sequences()

    def _load(self, data_path):
        frames, n_blocks = [], 0
        for d in sorted(Path(data_path).iterdir()):
            if not d.is_dir():
                continue
            df = pd.read_csv(d / 'trials.htsv', sep='\t')
            assert all(c in df.columns for c in REQUIRED), f"{d.stem}: missing columns"
            remainder = len(df) % (self.sequence_length + 1)
            if remainder:
                df = df.iloc[:-remainder]
            df['session_folder_name'] = d.stem
            df['sequence_block_idx']  = np.arange(len(df)) // (self.sequence_length + 1) + n_blocks
            n_blocks += len(df) // (self.sequence_length + 1)
            frames.append(_encode_df(df))
        assert frames, f"No valid sessions found under {data_path}"
        return pd.concat(frames, ignore_index=True)

    def _make_sequences(self):
        raw = torch.tensor(
            self.subject_df[['forced_choice', 'outcome', 'choice']].values,
            dtype=torch.float32)
        remainder = len(raw) % (self.sequence_length + 1)
        if remainder:
            raw = raw[:-remainder]
        seqs = raw.view(-1, self.sequence_length + 1, raw.size(1))
        inputs  = seqs[:, :-1, :]
        targets = F.one_hot(seqs[:, 1:, 2].long(), num_classes=2).float()
        return inputs.to(self.device), targets.to(self.device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def _session_split(self, eval_frac=0.1, val_frac=0.1, seed_eval=42, seed_split=0):
        """80/10/10 split on sessions; returns index lists over sequence blocks."""
        folders = np.array(sorted(self.subject_df['session_folder_name'].unique()))
        n = len(folders)

        eval_folders = np.random.default_rng(seed_eval).choice(folders, size=math.ceil(n * eval_frac), replace=False)
        rest_folders = np.setdiff1d(folders, eval_folders)
        val_folders  = np.random.default_rng(seed_split).choice(rest_folders, size=math.ceil(len(rest_folders) * val_frac), replace=False)
        train_folders = np.setdiff1d(rest_folders, val_folders)

        block_map = self.subject_df.groupby('session_folder_name')['sequence_block_idx'].unique()
        def blocks(fs): return sorted(np.concatenate([block_map[f] for f in fs]).tolist())

        return {
            'train': blocks(train_folders),
            'val':   blocks(val_folders),
            'eval':  blocks(eval_folders),
        }

    def get_split_labels(self, splits):
        """Return per-trial split label array ('train'/'val'/'eval'/'unused')."""
        n_trials = len(self.subject_df)
        seq_len  = self.sequence_length
        labels   = np.full(n_trials, 'unused', dtype=object)
        for split_name, block_indices in splits.items():
            trial_indices = np.concatenate(
                [np.arange(b * seq_len, (b + 1) * seq_len) for b in block_indices])
            trial_indices = trial_indices[trial_indices < n_trials]
            labels[trial_indices] = split_name
        return labels


# ── AB_SessionDataset (one sample per session) ────────────────────────────────

class AB_SessionDataset(Dataset):
    """One sample per session. Returns (1, n_trials, 3) / (1, n_trials, 2)."""

    def __init__(self, data_path, device=DEVICE):
        self.device = device
        inputs, targets, names, index = [], [], [], []
        cursor = 0

        for d in sorted(Path(data_path).iterdir()):
            if not d.is_dir():
                continue
            df = pd.read_csv(d / 'trials.htsv', sep='\t')
            assert all(c in df.columns for c in REQUIRED), f"{d.stem}: missing columns"
            df = _encode_df(df)

            raw = torch.tensor(df[['forced_choice', 'outcome', 'choice']].values, dtype=torch.float32)
            if len(raw) < 2:
                continue

            inp = raw[:-1]
            tgt = F.one_hot(raw[1:, 2].long(), num_classes=2).float()
            inputs.append(inp); targets.append(tgt)
            names.append(d.stem); index.append((cursor, cursor + len(inp)))
            cursor += len(inp)

        assert inputs, f"No valid sessions found under {data_path}"
        self.flat_inputs   = torch.cat(inputs).to(device)
        self.flat_targets  = torch.cat(targets).to(device)
        self.session_index = torch.tensor(index, dtype=torch.long)
        self.session_names = np.array(names)

    def __len__(self):
        return len(self.session_index)

    def __getitem__(self, idx):
        s, e = self.session_index[idx].tolist()
        return self.flat_inputs[s:e].unsqueeze(0), self.flat_targets[s:e].unsqueeze(0)

    def _session_split(self, eval_frac=0.1, val_frac=0.1, seed_eval=42, seed_split=0):
        """80/10/10 session-level split. Eval fixed by seed_eval; train/val by seed_split."""
        n   = len(self)
        idx = np.arange(n)
        eval_idx  = np.random.default_rng(seed_eval).choice(idx, size=math.ceil(n * eval_frac), replace=False)
        rest      = np.setdiff1d(idx, eval_idx)
        val_idx   = np.random.default_rng(seed_split).choice(rest, size=math.ceil(len(rest) * val_frac), replace=False)
        return {
            'train': sorted(np.setdiff1d(rest, val_idx).tolist()),
            'val':   sorted(val_idx.tolist()),
            'eval':  sorted(eval_idx.tolist()),
        }

    def get_split_labels(self, splits):
        """Return per-trial split label array ('train'/'val'/'eval'/'unused')."""
        # session_index holds [start, end) row spans in the flat tensor
        n_trials = self.flat_inputs.size(0)
        labels   = np.full(n_trials, 'unused', dtype=object)
        for split_name, session_indices in splits.items():
            for i in session_indices:
                s, e = self.session_index[i].tolist()
                labels[s:e] = split_name
        return labels


# ── DataLoader factory ─────────────────────────────────────────────────────────

def get_dataloader(dataset, split, splits=None, shuffle=None, batch_size=8, seed=0):
    """
    Return a DataLoader for 'train', 'val', 'eval', or 'all'.

    For AB_SessionDataset, batch_size is always 1 (sessions are variable length).
    For AB_Dataset, batch_size is used as given; shuffling is seeded via `seed`.
    """
    if splits is None:
        splits = dataset._session_split()

    subset  = dataset if split == 'all' else Subset(dataset, splits[split])
    shuffle = (split == 'train') if shuffle is None else shuffle

    if isinstance(dataset, AB_SessionDataset):
        if batch_size != 1:
            print(f"Notice: AB_SessionDataset ignores batch_size={batch_size}; using 1.")
        # num_workers > 0 incompatible with CUDA tensors in dataset
        return DataLoader(subset, batch_size=1, shuffle=shuffle, num_workers=0,
                          collate_fn=lambda b: b[0])
    else:
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, generator=generator)