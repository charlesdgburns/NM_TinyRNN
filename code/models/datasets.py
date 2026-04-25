import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = './NM_TinyRNN/data/AB_behaviour/WS16'
SEQUENCE_LENGTH = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _encode_df(df):
    df['forced_choice'] = df['forced_choice'].astype(int)
    df['outcome']       = df['outcome'].astype(int)
    df['choice']        = df['choice'].astype('category').cat.codes.astype(int)
    df['good_poke']     = df['good_poke'].astype('category').cat.codes.astype(int)
    return df

REQUIRED = ['forced_choice', 'outcome', 'choice', 'good_poke']


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
# ── AB_Dataset (sequence/batch based) ─────────────────────────────────────────

class AB_Dataset(Dataset):
    """Splits sessions into fixed-length sequences. Supports batch_size > 1."""

    def __init__(self, subject_data_path, sequence_length=SEQUENCE_LENGTH, device=DEVICE):
        self.device = device
        self.sequence_length = sequence_length
        self.subject_data_path = Path(subject_data_path)
        self.subject_df = self._load_and_concat_data( )
        self.inputs, self.targets = self._create_sequences()

    def _load_and_concat_data(self):
        subject_data = []
        session_folder_name = []
        n_blocks = 0
        for session_dir in self.subject_data_path.iterdir():
            if session_dir.is_dir():
                trials_df = pd.read_csv(session_dir/'trials.htsv', sep = '\t')
                assert np.all([x in trials_df.columns for x in ['forced_choice', 'outcome', 'choice', 'good_poke']]), "DataFrame missing required columns"
                remainder = (len(trials_df)%(self.sequence_length+1))
                trials_df['session_folder_name'] = np.repeat(session_dir.stem, len(trials_df))
                trials_df['session_trial_idx'] = range(len(trials_df))
                trials_df['sequence_block_idx'] = np.arange(len(trials_df))//(self.sequence_length+1) + n_blocks
                trials_df['block_trial_idx'] = np.concatenate([np.arange(0,len(trials_df)-remainder),np.repeat(np.nan,remainder)])
                n_blocks+=len(trials_df)//(self.sequence_length+1)
                subject_data.append(trials_df)
                session_folder_name.extend(np.repeat(session_dir.stem, len(subject_data[-1]))) 
        self.session_folder_name = session_folder_name
        df =  pd.concat(subject_data)
        # Convert boolean and categorical columns to numerical
        df['forced_choice'] = df['forced_choice'].astype(int)
        df['outcome'] = df['outcome'].astype(int)
        df['choice'] = df['choice'].astype('category').cat.codes.astype(int) #this is consistent with below
        df['good_poke'] = df['good_poke'].astype('category').cat.codes.astype(int) #consistent with above
        percent_excluded = df.block_trial_idx.isna().mean()
        print(f'Sequence length {self.sequence_length} excludes {percent_excluded:.1%} of trials')
        return df

    def _create_sequences(self):
        # Convert to tensor and handle potential remainders
        data_tensor = torch.tensor(self.subject_df[['forced_choice', 'outcome', 'choice']].values, dtype=torch.float32)
        num_rows = data_tensor.size(0)
        remainder = num_rows % (self.sequence_length+1) #add one here and below to account for time shifting
        if remainder != 0:
            data_tensor = data_tensor[:-remainder] # Trim off the remainder

        # Reshape into sequences
        num_sequences = data_tensor.size(0) // (self.sequence_length+1)
        sequences = data_tensor.view(num_sequences, self.sequence_length+1, data_tensor.size(1))

        # Create inputs and targets
        # Inputs are 'forced_choice', 'outcome', and 'choice' at time t
        inputs = sequences[:, :-1, :]
        # Targets are 'choice' at time t+1, one-hot encoded
        targets_codes = sequences[:, 1:, 2].long() # Get the categorical codes as long tensor
        targets = torch.nn.functional.one_hot(targets_codes, num_classes=2).float() # One-hot encode
        return inputs, targets

    def __len__(self):
        return self.inputs.size(0)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def _session_split(self, eval_frac=0.1, val_frac=0.1, seed_eval=-1, seed_split=0):
        """80/10/10 split on sessions; returns index lists over sequence blocks."""
        folders = np.array(sorted(self.subject_df['session_folder_name'].unique()))
        n = len(folders)
        n_eval_folders = math.ceil(n*eval_frac)

        if seed_eval == -1:
            eval_folders = folders[-n_eval_folders:]
        else:
            eval_folders = np.random.default_rng(seed_eval).choice(folders, size=n_eval_folders, replace=False)

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

