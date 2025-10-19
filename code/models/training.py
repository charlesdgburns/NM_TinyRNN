import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
from typing import List, Optional, Tuple, Dict, Any
from copy import deepcopy

from pathlib import Path

#Function to train a model on a subject's data

#Trainer class
TRAIN_SEED = 42 #fix the seed for splits and dataloading
# Set seeds for reproducibility
torch.manual_seed(TRAIN_SEED)
np.random.seed(TRAIN_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(TRAIN_SEED)

# utility that might be helpful elsewhere:
# we name our models consistently as follows:



## main class ##

class Trainer:
    def __init__(
        self,
        save_path: Path,
        weight_seeds: List[float] = [1,2,3,4,5,6,7,8,9,10], #[1,2,3,4,5,6,7,8,9,10],
        sparsity_lambdas: List[float] = [1e-5], #[1e-1,1e-3,1e-5],
        energy_lambdas: List[float] = [1e-2],
        hebbian_lambdas: List[float] = [None],
        learning_rate: float = 1e-4, #1e-4,#0.005,
        batch_size: int = 8,
        max_epochs: int = 10,
        early_stop: int = 200,
        train_seed: int = TRAIN_SEED
    ):
        """
        Simple and concise trainer for neural networks with hyperparameter tuning.
        
        Args:
            save_path: Directory to save model and results
            sparsity_lambdas: List of sparsity regularization values to try. Outputs best model among them.
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            max_epochs: Maximum number of training epochs
            early_stop: Number of epochs without validation improvement before stopping
            seed: Random seed for reproducibility
        """
        self.save_path = Path(save_path) #ensure it's a Path object
        self.sparsity_lambdas = sparsity_lambdas
        self.energy_lambdas = energy_lambdas
        self.hebbian_lambdas = hebbian_lambdas
        self.weight_seeds = weight_seeds
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stop = early_stop
        self.train_seed = train_seed
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
    def fit(self, model, dataset) -> pd.DataFrame:
        """
        Fit the model with hyperparameter tuning.
        
        Args:
            model: PyTorch model to train
            dataset: PyTorch dataset
            
        Returns:
            training_losses_df: DataFrame with training history for each sparsity value
        """
        print(f"Starting training with {len(self.sparsity_lambdas)} sparsity values...")
        print(f"Dataset size: {len(dataset)}")
        self.save_path.mkdir(parents=True,exist_ok=True)
        # Split dataset
        train_dataset, val_dataset, test_dataset,_ = self._split_dataset(dataset)
        print(f"Split sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Evaluation: {len(test_dataset)}")
        
        # Create dataloaders
        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False)
        test_loader = self._create_dataloader(test_dataset, shuffle=False)
        
        # Store all results
        all_results = []
        best_overall_val_loss = float('inf')
        best_model_info = None
        
        ## TRAINING ##
        for sparsity_lambda in self.sparsity_lambdas:
            for energy_lambda in self.energy_lambdas:
                for hebbian_lambda in self.hebbian_lambdas:
                    for weight_seed in self.weight_seeds:
                        print(f"\nTraining with  \n weight seed = {weight_seed},\n sparsity lambda = {sparsity_lambda}, \n energy lambda = {energy_lambda}, \n hebbian lambda = {hebbian_lambda}")
                        # Reset model to initial state for each set of values
                        model.sparsity_lambda = sparsity_lambda 
                        model.energy_lambda = energy_lambda
                        model.hebbian_lambda = hebbian_lambda
                        model.weight_seed = weight_seed
                        model.init_weights() #reset model weights before training.
                    
                        losses_dict, best_model_dict = self._train_single_model(
                            model, train_loader, val_loader)
                        #append all the hyperparameters for each run
                        n_epochs = len(losses_dict['train_prediction'])
                        losses_dict['sparsity_lambda'] = np.repeat(sparsity_lambda,n_epochs )
                        losses_dict['energy_lambda'] = np.repeat(energy_lambda, n_epochs)
                        losses_dict['hebbian_lambda'] = np.repeat(hebbian_lambda, n_epochs)
                        losses_dict['weight_seed'] = np.repeat(weight_seed,n_epochs)
                        losses_dict['epoch'] = np.arange(1,n_epochs+1)
                        
                        all_results.append(losses_dict)
                        
                        # Track best model across all sparsity values
                        if best_model_dict['val_pred_loss'] < best_overall_val_loss:
                            best_overall_val_loss = best_model_dict['val_pred_loss']
                            best_model_info = {
                                'sparsity_lambda': sparsity_lambda,
                                'energy_lambda': energy_lambda,
                                'hebbian_lambda':hebbian_lambda,
                                'weight_seed': weight_seed,
                                'val_pred_loss': best_model_dict['val_pred_loss'],
                                'model_state': best_model_dict['model_state']
                            }
                        # catch nan values
                        if best_model_info is None:
                            best_model_info = {
                                'sparsity_lambda': sparsity_lambda,
                                'energy_lambda': energy_lambda,
                                'hebbian_lambda':hebbian_lambda,
                                'weight_seed': weight_seed,
                                'val_pred_loss': best_model_dict['val_pred_loss'],
                                'model_state': best_model_dict['model_state']
                            }
                    
                
        ## EVAL AND SAVING ##
        # Generate model ID for saving data:
        model_id = model.get_model_id()
        torch.save(best_model_info['model_state'],self.save_path/f'{model_id}_model_state.pth')
        
        # Evaluate on test set using run_epoch // we only really care about prediction cross-entropy
        print(f"\nEvaluating best model (sparsity = {best_model_info['sparsity_lambda']:.0e}, energy = {best_model_info['energy_lambda']}, weight_seed = {best_model_info['weight_seed']}) on test set...")
        model.load_state_dict(best_model_info['model_state'])
        #need to set hyperparameters for generating the correct options_dict further down
        model.sparsity_lambda = best_model_info['sparsity_lambda'] 
        model.energy_lambda = best_model_info['energy_lambda']
        model.hebbian_lambda = best_model_info['hebbian_lambda']
        model.weight_seed = best_model_info['weight_seed']
        model.eval()
        eval_losses = self._run_epoch(model, test_loader, None, 
                                            training=False)
        best_model_info['eval_pred_loss'] = eval_losses['prediction']
        print(f"Evaluation loss: {eval_losses['prediction']:.6f}")
        
        # Save model info
        model_info = self.__dict__.copy()
        model_info['model_id'] = model_id
        model_info['save_path'] = str(model_info['save_path']) #fix possible posix path issues
        model_info['best_val_pred_loss'] = best_model_info['val_pred_loss']
        model_info['eval_pred_loss'] = best_model_info['eval_pred_loss']
        model_info['options_dict'] = model.get_options_dict()
        with open(os.path.join(self.save_path, f'{model_id}_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Create training losses DataFrame
        training_losses_df = pd.concat([pd.DataFrame(x) for x in all_results if type(x) is dict])
        
        # Save training history
        losses_path = os.path.join(self.save_path, f'{model_id}_training_losses.htsv')
        training_losses_df.to_csv(losses_path, sep='\t', index=False)
        
        print(f"\nTraining complete!")
        print(f"Best model (λ={best_model_info['sparsity_lambda']:.0e}) saved to: {self.save_path}")
        print(f"Best validation loss: {best_model_info['val_pred_loss']:.6f}")
        print(f"Test loss: {best_model_info['eval_pred_loss']:.6f}")
        
        print(f"Lastly, extracting activations on full dataset")
        trial_by_trial_df = self.get_model_trial_by_trial_df(model,dataset)
        trial_by_trial_df.to_csv(self.save_path/f'{model_id}_trials_data.htsv', sep = '\t', index=False)
        return training_losses_df
    
    def _split_dataset(self, dataset) -> Tuple[Subset, Subset, Subset]:
        """Split dataset into train/val/test (80/10/10) respecting session folders."""
    
        # Get unique folders and shuffle them
        unique_folders = dataset.subject_df['session_folder_name'].unique()
        np.random.seed(self.train_seed)
        np.random.shuffle(unique_folders)
        
        # Calculate split points
        n_folders = len(unique_folders)
        train_end = int(0.8 * n_folders)
        val_size = max(1, round(0.1 * n_folders))
        
        # Assign folders to splits
        train_folders = unique_folders[:train_end]
        val_folders = unique_folders[train_end:train_end + val_size]
        test_folders = unique_folders[train_end + val_size:]
        
        folder_name2idx = dataset.subject_df.groupby('session_folder_name')['sequence_block_idx'].unique()

        indices_dict = {}
        indices_dict['indices_train'] = np.concat([folder_name2idx[x] for x in train_folders])
        indices_dict['indices_validation'] = np.concat([folder_name2idx[x] for x in val_folders])
        indices_dict['indices_evaluation'] = np.concat([folder_name2idx[x] for x in test_folders])

        train_dataset = Subset(dataset, indices_dict['indices_train'])
        val_dataset = Subset(dataset, indices_dict['indices_validation'])
        test_dataset = Subset(dataset, indices_dict['indices_evaluation'])
      
        return train_dataset, val_dataset, test_dataset, indices_dict
    
    def _create_dataloader(self, dataset, shuffle: bool = True) -> DataLoader:
        """Create dataloader with deterministic shuffling."""
        generator = torch.Generator()
        generator.manual_seed(self.train_seed)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None
        )
    
    def _run_epoch(
        self,
        model,
        dataloader: DataLoader,
        optimizer=None,
        training: bool = True
    ) -> Tuple[float, float]:
        """
        Run a single epoch on the given dataloader.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader to iterate over
            optimizer: Optimizer for training (None for evaluation)
            sparsity_lambda: Sparsity regularization weight
            training: Whether this is a training epoch
            
        Returns:
           epoch_losses: dictionary with loss type as keys and avg loss as value
        """
        if training:
            model.train()
        else:
            model.eval()
        
        epoch_losses = {}
        
        context_manager = torch.no_grad() if not training else torch.enable_grad()
        with context_manager:
            for batch_inputs, batch_targets in dataloader:
                B, S, _ = batch_inputs.shape
                if training and optimizer is not None:
                    optimizer.zero_grad()
                predictions, hidden_states = model(batch_inputs)
                # remove the free choice trials from the loss!
                free_choice = (batch_inputs[:,:,0]==0)#this should be a bool size (n_batch, n_seq)
                #displace by one index, since forced choice input at t means prediction for t-1 is impossible.
                free_choice[:,:-1] = free_choice[:,1:].clone() 
                free_choice = free_choice.flatten()
                masked_pred = predictions.reshape(B*S,model.O)[free_choice]
                masked_targets = batch_targets.reshape(B*S,model.O)[free_choice]
                losses =  model.compute_losses(masked_pred,masked_targets, hidden_states)
                
                if training and optimizer is not None:
                    loss = sum([x for x in losses.values()])
                    loss.backward()
                    optimizer.step()
               #store losses for later plotting
                for loss_type, loss_value in losses.items():
                    if len(epoch_losses)<len(losses):#first epoch needs initialisation
                        epoch_losses[loss_type]= 0
                    epoch_losses[loss_type]+=(loss_value.item())/len(dataloader) #division to get mean
                
        return epoch_losses
    
    def _train_single_model(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Tuple[Dict[str, Any], float]:
        """
        Train a single model with given sparsity lambda.
        
        Returns:
            losses_dict: Dictionary containing training history
            best_val_loss: Best validation loss achieved
        """
        # Clone model to avoid modifying original
        model_copy = deepcopy(model)
        model_copy.load_state_dict(model.state_dict())
        
        optimizer = torch.optim.AdamW(model_copy.parameters(), 
                                      lr=self.learning_rate,
                                      weight_decay = 0.0)
        
        # Training history
        losses_dict={}
        best_model_dict = {}

        best_val_pred_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None
        
        for epoch in tqdm(range(self.max_epochs), desc=f"λ={model.sparsity_lambda:.0e}"):
            # Training epoch
            train_losses = self._run_epoch(
                model_copy, train_loader, optimizer, training=True)
            
            # Validation epoch // here we only care about cross-entropy of predictions
            validation_losses = self._run_epoch(model_copy, val_loader, None, 
                                           training=False)
            
            # Store losses
            for loss_type, loss_value in train_losses.items():
                if len(losses_dict) <= len(train_losses): #instantiate the first lists
                    losses_dict[f'train_{loss_type}']=[]
                    losses_dict['val_pred_losses'] = [] #instantiate this here also
                losses_dict[f'train_{loss_type}'].append(loss_value)
            losses_dict['val_pred_losses'].append(validation_losses['prediction'])    
            # Early stopping check
            if validation_losses['prediction'] < best_val_pred_loss:
                best_val_pred_loss = validation_losses['prediction']
                epochs_without_improvement = 0
                best_model_state = model_copy.to('cpu').state_dict().copy()
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= self.early_stop:
                print(f"Early stopping after {epoch + 1} epochs")
                break
            
            if np.isnan(train_losses['hebbian']) or np.isnan(validation_losses['prediction']):
                print('NaN hebbian loss, skipping training')
                best_val_pred_loss = np.nan
                epochs_without_improvement = np.nan
                best_model_state = model_copy.to('cpu').state_dict().copy()
                break

        # Store further training history
        best_model_dict['epochs_trained'] = epoch+1
        best_model_dict['val_pred_loss'] = best_val_pred_loss
        best_model_dict['model_state'] = best_model_state

        return losses_dict, best_model_dict
    
    def get_model_trial_by_trial_df(self, model, dataset):
        '''Runs through the entire dataset (also training data)'''
        model.eval()
        trial_by_trial_data = {}
        inputs_reshaped = dataset.subject_df[['forced_choice','outcome','choice']].values[None,:,:]
        inputs_reshaped = torch.tensor(inputs_reshaped,dtype = torch.float32)
        with torch.no_grad():
            predictions, _ = model(inputs_reshaped)  
            if not model.rnn_type == 'vanilla': 
                if not model.input_forced_choice:
                    inputs_reshaped = inputs_reshaped[:,:,1:]
                if model.input_encoding == 'bipolar':
                    inputs_reshaped = inputs_reshaped*2-1 #maps 0 to -1 and 1 to 1.
                hidden_state, gate_activations = model.rnn(inputs_reshaped, return_gate_activations = True)
                # add hidden activations
                for each_unit in range(model.H):
                    trial_by_trial_data[f'hidden_{each_unit+1}'] = hidden_state[:,:,each_unit].flatten()
                # add gate activations
                for gate_label, activations in gate_activations.items():    
                    for each_unit in range(activations.shape[-1]):
                        trial_by_trial_data[f'gate_{gate_label}_{each_unit+1}'] = activations[:,:,each_unit].flatten()
        
        # add logit and probabilities data:
        log_probs= predictions.log_softmax(dim=2)
        logits = (log_probs[:,:,0] - log_probs[:,:,1]).flatten() #subtraction in log space is division before log.
        trial_by_trial_data['logit_value'] = logits
        trial_by_trial_data['logit_past'] = torch.cat([torch.tensor([torch.nan]),logits[:-1]])
        trial_by_trial_data['logit_change'] = torch.cat([torch.tensor([torch.nan]),(logits[1:]-logits[:-1])])
        trial_by_trial_data['prob_A'] = torch.exp(log_probs[:,:,0]).flatten()
        trial_by_trial_data['prob_b'] = torch.exp(log_probs[:,:,1]).flatten()
        #add actual trial data and whether trial was used in training, validation, or evaluation
        for column in dataset.subject_df.columns: 
            trial_by_trial_data[column] = dataset.subject_df.loc[:,column].values
        #add trial type as well for easy plotting later.
        labels = ['A1, R=0','A1, R=1', 'A2, R=0', 'A2, R=1']
        trial_type = trial_by_trial_data['choice']*2 + trial_by_trial_data['outcome']
        trial_by_trial_data['trial_type'] = [labels[i] for i in trial_type]
        
        _,_,_, indices_dict = self._split_dataset(dataset)
        for idx_label, idx_values in indices_dict.items():
            # Generate all trial indices for the given batches
            n_seq = dataset.inputs.shape[1] #(n_batch, n_seq, n_features) 
            n_trials = len(dataset.subject_df)
            trial_indices = np.concatenate([np.arange(idx * n_seq, (idx + 1) * n_seq) 
                                            for idx in idx_values])
            # Create boolean mask using isin
            all_trial_indices = np.arange(n_trials)
            label_indices = np.isin(all_trial_indices, trial_indices)
            indices_dict[idx_label] = label_indices
        trial_by_trial_data.update(indices_dict)
        #and what session folder to save to
        trial_by_trial_data['session_folder_name'] = dataset.session_folder_name[:n_trials]
        df = pd.DataFrame(trial_by_trial_data)
        return df