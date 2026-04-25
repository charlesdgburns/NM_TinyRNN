import torch
from torch.func import vmap, stack_module_state, functional_call
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import itertools
from NM_TinyRNN.code.models.datasets import get_dataloader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_SEED = 42 #seed to use for ordering to ensure reproducibility

class TrainerGPU:
    def __init__(
        self,
        weight_seeds: list = list(range(1,11)),
        sparsity_lambdas: list = [1e-1,1e-2,1e-3,1e-4, 1e-5],
        energy_lambdas: list = [0.1, 1e-2, 1e-3],
        hebbian_lambdas: list = [0.0], # Changed None to 0.0 for tensor compatibility
        learning_rate: float = 1e-2,
        batch_size: int = 16,
        max_epochs: int = 1000,
        early_stop: int = 20,
        train_seed: int = 42,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stop = early_stop
        self.train_seed = train_seed
        self.weight_seeds = weight_seeds
        self.sparsity_lambdas = sparsity_lambdas
        self.energy_lambdas = energy_lambdas
        self.hebbian_lambdas = hebbian_lambdas
        
        # 1. Flatten all hyperparameter combinations into a single list
        self.configs = list(itertools.product(
            sparsity_lambdas, energy_lambdas, hebbian_lambdas, weight_seeds
        ))
        self.num_models = len(self.configs)
        
        # Convert configs to tensors for GPU-side vectorized loss calculation
        config_tensor = torch.tensor(self.configs, dtype=torch.float32)
        self.sparsity_vec = config_tensor[:, 0]
        self.energy_vec   = config_tensor[:, 1]
        self.hebbian_vec  = config_tensor[:, 2]

    def _initialize_parallel_models(self, base_model):
        """Creates N copies of the model and stacks their parameters."""
        models = []
        for _, _, _, seed in self.configs:
            m = deepcopy(base_model)
            m.weight_seed = int(seed)
            m.init_weights()
            models.append(m)
        
        # Stack parameters into a single dict of tensors: [Num_Models, weight_dims...]
        params, buffers = stack_module_state(models)
        return params, buffers

    def fit(self, base_model, dataset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Parallelizing {self.num_models} models on {device}")

        # Prepare Data
        splits = dataset._session_split(seed_split=self.train_seed)
        train_loader = get_dataloader(dataset, 'train', splits, batch_size=self.batch_size)
        val_loader   = get_dataloader(dataset, 'val',   splits, batch_size=self.batch_size)
        
        # Initialize Parallel State
        params, buffers = self._initialize_parallel_models(base_model)
        # Move params to device and enable gradients
        params = {k: v.to(device).detach().requires_grad_(True) for k, v in params.items()}
        buffers = {k: v.to(device) for k, v in buffers.items()}
        
        # set random seed AFTER initialising weights. 
        # This ensures reproducibility in training loop  
        torch.manual_seed(TRAIN_SEED)
        np.random.seed(TRAIN_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(TRAIN_SEED)
            
        optimizer = torch.optim.AdamW(params.values(), 
                                      lr=self.learning_rate,
                                      weight_decay = 0.0) #important detail here, otherwise sparsity_lambda is misleading

        # Trackers for each model
        best_val_losses = torch.full((self.num_models,), float('inf'), device=device)
        epochs_no_improve = torch.zeros(self.num_models, device=device)
        active_mask = torch.ones(self.num_models, device=device, dtype=torch.bool)
        best_params = {k: v.clone() for k, v in params.items()}

        # Move lambda vectors to device
        s_vec = self.sparsity_vec.to(device)
        e_vec = self.energy_vec.to(device)
        h_vec = self.hebbian_vec.to(device)

        # Define the Functional Forward/Loss Pass for one instance
        def compute_single_model_loss(p, b, x, y, s_lam, e_lam, h_lam):
            """
            This function is executed for EACH model instance in parallel via vmap.
            p: parameters for 1 model
            b: buffers for 1 model
            x: input batch [Batch, Seq, Features]
            y: target batch [Batch, Seq, Outputs]
            """
            # 1. Functional forward pass
            # We use (p, b) to ensure we use the specific weights for this hyperparameter set
            predictions, hidden_states = functional_call(base_model, (p, b), x)
            forced_choice_mask = x[:,:,0]
            loss_dict = base_model.compute_losses(
                params=p,
                predictions=predictions,
                targets=y,
                forced_choice_mask = forced_choice_mask,
                hidden_states=hidden_states,
                sparsity_lambda=s_lam,
                energy_lambda=e_lam,
                hebbian_lambda=h_lam
            )
            # 5. Return the sum of all components
            # Inside vmap, this results in a single scalar per model.
            # vmap then stacks these into a tensor of shape [num_models].
            return loss_dict
        
        # Vectorize the loss function across the model dimension (dim 0)
        # in_dims: (0, 0, None, None, 0, 0, 0) means params/lambdas are unique per model, 
        # but data (x, y) is shared (None).
        vectorized_loss_fn = vmap(compute_single_model_loss, in_dims=(0, 0, None, None, 0, 0, 0))

        for epoch in tqdm(range(self.max_epochs)):
            if not active_mask.any(): break

            # --- Training Step ---
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # Compute parallel losses
                loss_dict = vectorized_loss_fn(params, buffers, batch_x, batch_y, s_vec, e_vec, h_vec)
                
                # We only want to backprop for models that haven't early-stopped
                masked_loss = (sum(loss_dict.values()) * active_mask).sum()
                masked_loss.backward()
                optimizer.step()

            # --- Validation Step ---
            with torch.no_grad():
                current_val_losses = torch.zeros(self.num_models, device=device)
                for v_x, v_y in val_loader:
                    v_x, v_y = v_x.to(device), v_y.to(device)
                    loss_dict = vectorized_loss_fn(params, buffers, v_x, v_y, s_vec, e_vec, h_vec)
                    current_val_losses += loss_dict['prediction']
                current_val_losses /= len(val_loader)

                # Update best states and early stopping counters
                improved = current_val_losses < best_val_losses
                for k in params:
                    best_params[k][improved] = params[k][improved].clone()
                
                best_val_losses[improved] = current_val_losses[improved]
                epochs_no_improve[improved] = 0
                epochs_no_improve[~improved] += 1
                
                # Early stopping mask
                active_mask = epochs_no_improve < self.early_stop

        # Final Separation: Find the best index overall
        best_idx = torch.argmin(best_val_losses)
        print(f"Search complete. Best model index: {best_idx}. Val. loss: {best_val_losses[best_idx]}")
        self._last_best_val_loss = best_val_losses[best_idx]
        # Extract the single best model's state dict
        final_state_dict = {k: v[best_idx].cpu() for k, v in best_params.items()}
        return final_state_dict, self.configs[best_idx]