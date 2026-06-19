import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path('./NM_TinyRNN/data/')
SEQUENCE_LENGTH = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

REQUIRED = ['forced_choice', 'outcome', 'choice', 'good_poke']


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _encode_df(df):
    df['forced_choice'] = df['forced_choice'].astype(int)
    df['outcome']       = df['outcome'].astype(int)
    df['choice']        = df['choice'].astype('category').cat.codes.astype(int)
    df['good_poke']     = df['good_poke'].astype('category').cat.codes.astype(int)
    return df

def input_encoder(inputs, input_encoding:str, input_forced_choice:bool):
    #assume inputs are given with shape #(n_batches, n_seq, n_features)
    #features ordered as forced_choice, choice, and outcome
    assert input_encoding in ['unipolar','encoder','bipolar','onehot','actonehot'], "input_encoding must be one of 'unipolar','bipolar','onehot','actonehot'"

    if not input_forced_choice:
        inputs = inputs[:,:,1:]
    if input_encoding == 'bipolar': #instead of 0,1 inputs are -1,1.
        inputs = inputs*2-1
    if input_encoding == 'actonehot': #onehot only the action inputs
        #separate outcome from action-related inputs
        outcome = inputs[:,:,-1]
        inputs = inputs[:,:,:-1] #may include forced choice context or not
    if 'onehot' in input_encoding: #we onehot encode the inputs.
        dims = inputs.shape[-1]
        # we use torch.functional after transforming the vector into 1D categories
        weights = 2 ** torch.arange(dims - 1, -1, -1, device=inputs.device)  
        # Dot product to get indices
        indices = (inputs * weights).sum(dim=-1).long()
        inputs = torch.nn.functional.one_hot(indices, num_classes=2**dims).to(torch.float32)   
    if input_encoding == 'actonehot': #now we combine with outcome again
        inputs = torch.concat([inputs,outcome.unsqueeze(-1)],dim=-1)
    return inputs
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
    """
    Loading sessions assumes folder structure of
    .../data/<subject_id>/<session_folder>/trials.htsv
    
    Splits sessions into fixed-length sequences. Supports batch_size > 1.
    """

    def __init__(self, subject_data_path, subject_ids=None, sequence_length=SEQUENCE_LENGTH, 
                 input_encoding = 'unipolar', input_forced_choice = False, device=DEVICE):
        self.device = device
        self.sequence_length = sequence_length
        self.subject_data_path = subject_data_path
        self.input_encoding = input_encoding
        self.input_forced_choice = input_forced_choice
        #handling an option for multiple subjects
        if subject_ids is None:
            self.subject_ids = [Path(subject_data_path).stem]
        else:
            self.subject_ids = subject_ids
        #load data and create input,target sequences
        self.subject_df = self._load_and_concat_data( )
        self.inputs, self.targets, self.forced_choice_mask = self._create_sequences()

    def _load_and_concat_data(self):
        subject_data = []
        session_folder_name = []
        n_blocks = 0

        parent_dir = DATA_PATH/'AB_behaviour'
        subject_dirs = [x for x in parent_dir.iterdir()
                        if x.is_dir() and any(subject_id in x.name for subject_id in self.subject_ids)]
        if not subject_dirs: #if this subdirectory is empty
            raise ValueError(f"No subject directories found under {parent_dir} matching {self.subject_ids}")
    
        session_dirs = []
        for subject_dir in subject_dirs:
            session_dirs.extend([x for x in subject_dir.iterdir() if x.is_dir()])

        session_dirs.sort()
        if not session_dirs:
            raise ValueError(f"No session directories found for path={self.subject_data_path} "
                             f"subject_ids={self.subject_ids}")

        for session_dir in session_dirs:
            trials_df = pd.read_csv(session_dir/'trials.htsv', sep = '\t')
            assert np.all([x in trials_df.columns for x in ['forced_choice', 'outcome', 'choice', 'good_poke']]), "DataFrame missing required columns"
            T, S = len(trials_df), (self.sequence_length)
            remainder = T%(S+1) # the +1 here is to account for the shift in trials between inputs and targets when creating sequences.
            this_session_folder_name = f'{session_dir.parent.stem}/{session_dir.stem}'
            trials_df['session_folder_name'] = np.repeat(this_session_folder_name, len(trials_df))
            trials_df['session_trial_idx'] = range(len(trials_df))
            trials_df['sequence_block_idx'] = np.concatenate([np.arange(T-remainder)//(S+1),
                                                              np.repeat(np.nan,remainder)]) + n_blocks
            trials_df['block_trial_idx'] = np.concatenate([np.arange(T-remainder),
                                                           np.repeat(np.nan,remainder)])
            n_blocks+=(T-remainder)//(S+1)
            subject_data.append(trials_df)
            session_folder_name.extend(np.repeat(this_session_folder_name, len(subject_data[-1]))) 
        self.session_folder_name = session_folder_name
        df =  pd.concat(subject_data)
        # Convert boolean and categorical columns to numerical
        df['forced_choice'] = df['forced_choice'].astype(int)
        df['choice'] = df['choice'].astype('category').cat.codes.astype(int) #this is consistent with below
        df['outcome'] = df['outcome'].astype(int)
        df['good_poke'] = df['good_poke'].astype('category').cat.codes.astype(int) #consistent with above
        percent_excluded = df.block_trial_idx.isna().mean()
        print(f'Sequence length {self.sequence_length} excludes {percent_excluded:.1%} of trials')
        return df

    def _create_sequences(self):
        # We want to train models faster by batching sessions into blocks of given sequence length
        # We want to respect sessions (no blocks across sessions), which is handled when loading the data.
        # this is handled by sequence_block_idx NaN values
        trials_data = self.subject_df.dropna(subset=['sequence_block_idx']) #we are dropping the np.nan values added when applying the blocks
        data_tensor = torch.tensor(trials_data[['forced_choice', 'choice','outcome']].values, dtype=torch.float32)
        num_rows = data_tensor.size(0)
        remainder = num_rows % (self.sequence_length+1) #add one here and below to account for time shifting
        if remainder != 0:
            data_tensor = data_tensor[:-remainder] # Trim off the remainder

        # Reshape into sequences
        num_sequences = data_tensor.size(0) // (self.sequence_length+1)
        sequences = data_tensor.view(num_sequences, self.sequence_length+1, data_tensor.size(1))

        # Create inputs and targets
        # Inputs are initially 'forced_choice', 'choice', and 'outcome' at time t
        inputs = sequences[:, :-1, :]
        inputs = input_encoder(inputs, self.input_encoding,self.input_forced_choice)
        # Targets are 'choice' at time t+1, one-hot encoded
        targets_codes = sequences[:, 1:, 1].long() # Get the categorical codes as long tensor
        #targets = torch.nn.functional.one_hot(targets_codes, num_classes=2).float() # One-hot encode
        targets = targets_codes
        #lastly we want to mask the targets that result from forced choices
        forced_choice_mask = sequences[:, 1:, 0]   # shape (num_seqs, seq_length), aligned with target (!) not current action.
        return inputs, targets, forced_choice_mask
    
    def __len__(self):
        return self.inputs.size(0)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.forced_choice_mask[idx]

    def _session_split(self, eval_frac=0.1, val_frac=0.1, seed_eval=-1, seed_split=0):
        """80/10/10 split on sessions; returns index lists over sequence blocks."""
        block_map = self.subject_df.dropna().groupby('session_folder_name')['sequence_block_idx'].unique()
        folders = np.array(sorted(block_map.index.astype(str)))
        if len(folders) == 0:
            raise ValueError("No valid session blocks found for splitting; check sequence_length and session lengths.")

        n = len(folders)
        n_eval_folders = math.ceil(n*eval_frac)

        if seed_eval == -1:
            eval_folders = folders[-n_eval_folders:]
        else:
            eval_folders = np.random.default_rng(seed_eval).choice(folders, size=n_eval_folders, replace=False)

        rest_folders = np.setdiff1d(folders, eval_folders)
        val_folders  = np.random.default_rng(seed_split).choice(rest_folders, size=math.ceil(len(rest_folders) * val_frac), replace=False)
        train_folders = np.setdiff1d(rest_folders, val_folders)
        print(f'Data split (session-level): {len(train_folders)} train, {len(val_folders)} validation, and {len(eval_folders)} evaluation')
        def blocks(fs): return sorted(int(x) for x in np.concatenate([block_map[f] for f in fs]).tolist())

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
                [np.arange(b * (seq_len+1), (b + 1) * (seq_len+1)) for b in block_indices])
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

