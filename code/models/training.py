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
SEED = 42
# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

class Trainer:
    def __init__(
        self,
        save_path: Path,
        sparsity_lambda: List[float] = [1,1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        learning_rate: float = 0.05,
        batch_size: int = 8,
        max_epochs: int = 10000,
        early_stop: int = 200,
    ):
        """
        Simple and concise trainer for neural networks with hyperparameter tuning.
        
        Args:
            save_path: Directory to save model and results
            sparsity_lambda: List of sparsity regularization values to try
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            max_epochs: Maximum number of training epochs
            early_stop: Number of epochs without validation improvement before stopping
            seed: Random seed for reproducibility
        """
        self.save_path = Path(save_path) #ensure it's a Path object
        self.sparsity_lambda = sparsity_lambda
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stop = early_stop
        self.seed = SEED
        
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
        print(f"Starting training with {len(self.sparsity_lambda)} sparsity values...")
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
        for sparsity_lambda in self.sparsity_lambda:
            print(f"\nTraining with sparsity lambda = {sparsity_lambda}")
            
            # Reset model to initial state for each sparsity value
            model_state = model.state_dict().copy()
            
            losses_dict, val_pred_loss = self._train_single_model(
                model, train_loader, val_loader, sparsity_lambda
            )
            
            all_results.append(losses_dict)
            
            # Track best model across all sparsity values
            if val_pred_loss < best_overall_val_loss:
                best_overall_val_loss = val_pred_loss
                best_model_info = {
                    'sparsity_lambda': sparsity_lambda,
                    'val_pred_loss': val_pred_loss,
                    'model_state': losses_dict['best_model_state']
                }
            
            # Reset model for next sparsity value
            model.load_state_dict(model_state)
        
        ## EVAL AND SAVING ##
        # Generate model ID for saving data:
        model_id = f'{model.H}_unit_{model.rnn_type}'
        if model.rnn_type == 'NMRNN':
            model_id=f'{model.H}_unit_{model.rnn_type}_{model.nm_size}_subunits_{model.nm_dim}_{model.nm_mode}'
        torch.save(best_model_info['model_state'],self.save_path/f'{model_id}_model_state.pth')
        
        # Evaluate on test set using run_epoch // we only really care about prediction cross-entropy
        print(f"\nEvaluating best model (λ={best_model_info['sparsity_lambda']:.0e}) on test set...")
        model.load_state_dict(best_model_info['model_state'])
        eval_pred_loss, _ = self._run_epoch(model, test_loader, None, training=False)
        best_model_info['eval_pred_loss'] = eval_pred_loss
        print(f"Evaluation loss: {eval_pred_loss:.6f}")
        
        # Save model info
        model_info = self.__dict__.copy()
        model_info['save_path'] = str(model_info['save_path']) #fix possible posix path issues
        model_info['best_sparsity_lambda'] = best_model_info['sparsity_lambda']
        model_info['best_val_pred_loss'] = best_model_info['val_pred_loss']
        model_info['eval_pred_loss'] = best_model_info['eval_pred_loss']
        with open(os.path.join(self.save_path, f'{model_id}_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Create training losses DataFrame
        df_data = []
        for result in all_results:
            max_epochs = len(result['train_pred_losses'])
            for epoch in range(max_epochs):
                df_data.append({
                    'sparsity_lambda': result['sparsity_lambda'],
                    'epoch': epoch,
                    'val_pred_loss': result['val_pred_losses'][epoch],
                    'train_pred_loss': result['train_pred_losses'][epoch],
                    'train_sparsity_loss': result['train_sparsity_losses'][epoch],
                })
        
        training_losses_df = pd.DataFrame(df_data)
        
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
        """Split dataset into train/val/test (80/10/10)."""
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        indices_dict = {}
        # Use numpy for deterministic shuffling
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        
        # 80/10/10 split
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        
        indices_dict['indices_train'] = indices[:train_size]
        indices_dict['indices_validation'] = indices[train_size:train_size + val_size]
        indices_dict['indices_evaluation'] = indices[train_size + val_size:]
        
        train_dataset = Subset(dataset, indices_dict['indices_train'])
        val_dataset = Subset(dataset, indices_dict['indices_validation'])
        test_dataset = Subset(dataset, indices_dict['indices_evaluation'])
        
        return train_dataset, val_dataset, test_dataset, indices_dict
    
    def _create_dataloader(self, dataset, shuffle: bool = True) -> DataLoader:
        """Create dataloader with deterministic shuffling."""
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
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
            avg_pred_loss: Average prediction loss
            avg_sparsity_loss: Average sparsity loss (weighted by lambda)
        """
        if training:
            model.train()
        else:
            model.eval()
        
        total_pred_loss = 0
        total_sparsity_loss = 0
        
        context_manager = torch.no_grad() if not training else torch.enable_grad()
        with context_manager:
            for batch_inputs, batch_targets in dataloader:
                B, S, _ = batch_inputs.shape
                if training and optimizer is not None:
                    optimizer.zero_grad()
                predictions = model(batch_inputs)
                
                if not training: # remove the free choice trials from the validation loss!
                    free_choice = (batch_inputs[:,:,0]==0).flatten() #this should be a bool size (n_batch, n_seq)
                    masked_pred = predictions.reshape(B*S,model.O)[free_choice]
                    masked_targets = batch_targets.reshape(B*S,model.O)[free_choice]
                    prediction_loss, sparsity_loss = model.compute_losses(masked_pred,masked_targets)
                    
                prediction_loss, sparsity_loss = model.compute_losses(
                    predictions.view(-1, model.O),
                    batch_targets.view(-1, model.O))
                
                if training and optimizer is not None:
                    loss = prediction_loss + sparsity_loss
                    loss.backward()
                    optimizer.step()
                
                total_pred_loss += prediction_loss.item()
                total_sparsity_loss += (sparsity_loss).item()
        
        avg_pred_loss = total_pred_loss / len(dataloader)
        avg_sparsity_loss = total_sparsity_loss / len(dataloader)
        
        return avg_pred_loss, avg_sparsity_loss
    
    def _train_single_model(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        sparsity_lambda: float
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
        model_copy.sparsity_lambda = sparsity_lambda 
        
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=self.learning_rate)
        
        # Training history
        train_pred_losses = []
        train_sparsity_losses = []
        val_pred_losses = []
        
        best_val_pred_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None
        
        for epoch in tqdm(range(self.max_epochs), desc=f"λ={sparsity_lambda:.0e}"):
            # Training epoch
            train_pred_loss, train_sparsity_loss = self._run_epoch(
                model_copy, train_loader, optimizer, training=True)
            
            # Validation epoch // here we only care about cross-entropy of predictions
            val_pred_loss, _ = self._run_epoch(model_copy, val_loader, None, training=False)
            
            # Store losses
            train_pred_losses.append(train_pred_loss)
            train_sparsity_losses.append(train_sparsity_loss)
            val_pred_losses.append(val_pred_loss)
                        
            # Early stopping check
            if val_pred_loss < best_val_pred_loss:
                best_val_pred_loss = val_pred_loss
                epochs_without_improvement = 0
                best_model_state = model_copy.to('cpu').state_dict().copy()
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= self.early_stop:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        
        # Store training history
        losses_dict = {
            'sparsity_lambda': sparsity_lambda,
            'train_pred_losses': train_pred_losses,
            'train_sparsity_losses': train_sparsity_losses,
            'val_pred_losses': val_pred_losses,
            'epochs_trained': len(train_pred_losses),
            'best_val_pred_loss': best_val_pred_loss,
            'best_model_state': best_model_state
        }
        
        return losses_dict, best_val_pred_loss

    


    def get_model_trial_by_trial_df(self, model, dataset):
        '''Runs through the entire dataset (also training data)'''
        model.eval()
        trial_by_trial_data = {}
        n_batches, n_seq, n_inputs = dataset.inputs.shape
        n_trials = n_batches*n_seq
        inputs_reshaped = dataset.inputs.reshape(n_trials,n_inputs).unsqueeze(0)
        with torch.no_grad():
            predictions = model(inputs_reshaped)  
            if not model.rnn_type == 'vanilla': 
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
        logits = (log_probs[:,:,0] - log_probs[:,:,1]).flatten()
        trial_by_trial_data['logit_value'] = logits
        trial_by_trial_data['logit_change'] = torch.cat([torch.tensor([torch.nan]),(logits[1:]-logits[:-1])])
        trial_by_trial_data['prob_A'] = torch.exp(log_probs[:,:,0]).flatten()
        trial_by_trial_data['prob_b'] = torch.exp(log_probs[:,:,1]).flatten()
        
        #add actual trial data and whether trial was used in training, validation, or evaluation
        _,_,_, indices_dict = self._split_dataset(dataset)
        trials_inputs = dataset.inputs.reshape(-1, 3).numpy().astype(int)
        for i, label in enumerate(['forced_choice','outcome','choice']): #this ordering matters.
            trial_by_trial_data[label] = trials_inputs[:,i]
        for idx_label, idx_values in indices_dict.items():
            # Generate all trial indices for the given batches
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