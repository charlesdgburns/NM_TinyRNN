import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path

#global variables #

DATA_PATH = './NM_TinyRNN/data/AB_behaviour/WS16'
SEQUENCE_LENGTH = 150 # Define your desired sequence length



class AB_Dataset(Dataset):
    def __init__(self, 
                 subject_data_path, 
                 sequence_length=SEQUENCE_LENGTH):
        self.subject_data_path = Path(subject_data_path)
        self.sequence_length = sequence_length
        self.subject_df = self._load_and_concat_data()
        self.inputs, self.targets = self._create_sequences()
        
    def _load_and_concat_data(self):
        subject_data = []
        session_folder_name = []
        for session_dir in self.subject_data_path.iterdir():
            if session_dir.is_dir():
                trials_df = pd.read_csv(session_dir/'trials.htsv', sep = '\t')
                trials_df['session_trial_idx'] = range(len(trials_df))
                trials_df['session_folder_name'] = np.repeat(session_dir.stem, len(trials_df))
                subject_data.append(trials_df)
                session_folder_name.extend(np.repeat(session_dir.stem, len(subject_data[-1]))) 
        self.session_folder_name = session_folder_name
        df =  pd.concat(subject_data)
        # Convert boolean and categorical columns to numerical
        df['forced_choice'] = df['forced_choice'].astype(int)
        df['outcome'] = df['outcome'].astype(int)
        df['choice'] = df['choice'].astype('category').cat.codes.astype(int) #this is consistent with below
        df['good_poke'] = df['good_poke'].astype('category').cat.codes.astype(int) #consistent with above
        return df

    def _create_sequences(self):
      
        # Convert to tensor and handle potential remainder
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

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Example usage:
# Define the data path and sequence length

# Create the dataset and dataloader
dataset = AB_Dataset(DATA_PATH, SEQUENCE_LENGTH)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Example of iterating through the dataloader
# for batch_inputs, batch_targets in dataloader:
#      print("Inputs shape:", batch_inputs.shape)
#      print("Targets shape:", batch_targets.shape)
#      break # Just printing the first batch shapes for demonstration