import torch
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from NM_TinyRNN.code.models.datasets import AB_SessionDataset, get_dataloader

TRAIN_SEED = 42
torch.manual_seed(TRAIN_SEED)
np.random.seed(TRAIN_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(TRAIN_SEED)


class Trainer:
    def __init__(
        self,
        save_path: Path,
        weight_seeds: List[float] = [1, 2, 3, 4, 5],
        sparsity_lambdas: List[float] = [1e-7],
        energy_lambdas: List[float] = [1e-3],
        hebbian_lambdas: List[float] = [0.0],
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        max_epochs: int = 1000,
        early_stop: int = 20,
        train_seed: int = TRAIN_SEED,
    ):
        self.save_path        = Path(save_path)
        self.weight_seeds     = weight_seeds
        self.sparsity_lambdas = sparsity_lambdas
        self.energy_lambdas   = energy_lambdas
        self.hebbian_lambdas  = hebbian_lambdas
        self.learning_rate    = learning_rate
        self.batch_size       = batch_size
        self.max_epochs       = max_epochs
        self.early_stop       = early_stop
        self.train_seed       = train_seed
        self.save_path.mkdir(parents=True, exist_ok=True)

    def fit(self, model, dataset) -> pd.DataFrame:
        print(f"Starting training | dataset size: {len(dataset)}")

        splits = dataset._session_split(seed_split=self.train_seed, eval_frac=0.1, val_frac=0.1)
        train_loader = get_dataloader(dataset, 'train', splits, batch_size=self.batch_size, seed=self.train_seed)
        val_loader   = get_dataloader(dataset, 'val',   splits, batch_size=self.batch_size)
        eval_loader  = get_dataloader(dataset, 'eval',  splits, batch_size=self.batch_size)
        print(f"Split sizes — Train: {len(splits['train'])}, Val: {len(splits['val'])}, Eval: {len(splits['eval'])}")

        all_results = []
        best_overall_val_loss = float('inf')
        best_model_info = None

        for sparsity_lambda in self.sparsity_lambdas:
            for energy_lambda in self.energy_lambdas:
                for hebbian_lambda in self.hebbian_lambdas:
                    for weight_seed in self.weight_seeds:
                        print(f"\nweight_seed={weight_seed}, sparsity={sparsity_lambda:.0e}, "
                              f"energy={energy_lambda}, hebbian={hebbian_lambda}")
                        model.sparsity_lambda = sparsity_lambda
                        model.energy_lambda   = energy_lambda
                        model.hebbian_lambda  = hebbian_lambda
                        model.weight_seed     = weight_seed
                        model.init_weights()

                        losses_dict, best_model_dict = self._train_single_model(model, train_loader, val_loader)

                        n = len(losses_dict['train_prediction'])
                        losses_dict.update({
                            'sparsity_lambda': np.repeat(sparsity_lambda, n),
                            'energy_lambda':   np.repeat(energy_lambda, n),
                            'hebbian_lambda':  np.repeat(hebbian_lambda, n),
                            'weight_seed':     np.repeat(weight_seed, n),
                            'epoch':           np.arange(1, n + 1),
                        })
                        all_results.append(losses_dict)

                        if best_model_dict['val_pred_loss'] < best_overall_val_loss:
                            best_overall_val_loss = best_model_dict['val_pred_loss']
                            best_model_info = {**best_model_dict,
                                               'sparsity_lambda': sparsity_lambda,
                                               'energy_lambda':   energy_lambda,
                                               'hebbian_lambda':  hebbian_lambda,
                                               'weight_seed':     weight_seed}
                        if best_model_info is None:  # catch all-NaN runs
                            best_model_info = {**best_model_dict,
                                               'sparsity_lambda': sparsity_lambda,
                                               'energy_lambda':   energy_lambda,
                                               'hebbian_lambda':  hebbian_lambda,
                                               'weight_seed':     weight_seed}

        # ── Eval & saving ──────────────────────────────────────────────────────
        model_id = model.get_model_id()
        torch.save(best_model_info['model_state'], self.save_path / f'{model_id}_model_state.pth')

        print(f"\nEvaluating best model (sparsity={best_model_info['sparsity_lambda']:.0e}, "
              f"weight_seed={best_model_info['weight_seed']}) on eval set...")
        model.load_state_dict(best_model_info['model_state'])
        model.sparsity_lambda = best_model_info['sparsity_lambda']
        model.energy_lambda   = best_model_info['energy_lambda']
        model.hebbian_lambda  = best_model_info['hebbian_lambda']
        model.weight_seed     = best_model_info['weight_seed']
        model.eval()

        eval_losses = self._run_epoch(model, eval_loader, training=False)
        best_model_info['eval_pred_loss'] = eval_losses['prediction']
        print(f"Eval loss: {eval_losses['prediction']:.6f}")

        model_info = {**{k: (str(v) if isinstance(v, Path) else v)
                         for k, v in self.__dict__.items()},
                      'model_id':          model_id,
                      'best_val_pred_loss': best_model_info['val_pred_loss'],
                      'epochs_trained':     best_model_info['epochs_trained'],
                      'eval_pred_loss':     best_model_info['eval_pred_loss'],
                      'options_dict':       model.get_options_dict()}
        with open(self.save_path / f'{model_id}_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)

        training_losses_df = pd.concat([pd.DataFrame(x) for x in all_results if isinstance(x, dict)])
        training_losses_df.to_csv(self.save_path / f'{model_id}_training_losses.htsv', sep='\t', index=False)

        print(f"\nTraining complete! Best val loss: {best_model_info['val_pred_loss']:.6f} | "
              f"Eval loss: {best_model_info['eval_pred_loss']:.6f}")

        print("Extracting trial-by-trial activations...")
        trial_df = self.get_model_trial_by_trial_df(model, dataset, splits)
        trial_df.to_csv(self.save_path / f'{model_id}_trials_data.htsv', sep='\t', index=False)

        return training_losses_df

    def _run_epoch(self, model, dataloader, optimizer=None, training=True):
        model.train() if training else model.eval()
        epoch_losses = {}

        with (torch.enable_grad() if training else torch.no_grad()):
            for batch_inputs, batch_targets in dataloader:
                B, S, _ = batch_inputs.shape
                if training and optimizer is not None:
                    optimizer.zero_grad()

                predictions, hidden_states = model(batch_inputs)

                free_choice = (batch_inputs[:, :, 0] == 0)
                free_choice[:, :-1] = free_choice[:, 1:].clone()
                free_choice = free_choice.flatten()
                params_dict = {k:v for k,v in model.rnn.named_parameters()}
                losses = model.compute_losses(params_dict,
                    predictions.reshape(B * S, model.O)[free_choice],
                    batch_targets.reshape(B * S, model.O)[free_choice],
                    hidden_states, model.sparsity_lambda, model.energy_lambda, model.hebbian_lambda)

                if training and optimizer is not None:
                    sum(losses.values()).backward()
                    optimizer.step()

                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0) + v.item() / len(dataloader)

        return epoch_losses

    def _train_single_model(self, model, train_loader, val_loader):
        model_copy = deepcopy(model)
        optimizer  = torch.optim.AdamW(model_copy.parameters(), lr=self.learning_rate, weight_decay=0.0)

        losses_dict, best_model_dict = {}, {}
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in tqdm(range(self.max_epochs), desc=f"λ={model.sparsity_lambda:.0e}"):
            train_losses = self._run_epoch(model_copy, train_loader, optimizer, training=True)
            val_losses   = self._run_epoch(model_copy, val_loader, training=False)

            for k, v in train_losses.items():
                losses_dict.setdefault(f'train_{k}', []).append(v)
            losses_dict.setdefault('val_pred_losses', []).append(val_losses['prediction'])

            if val_losses['prediction'] < best_val_loss:
                best_val_loss      = val_losses['prediction']
                epochs_no_improve  = 0
                best_model_state   = deepcopy(model_copy.cpu().state_dict())
                model_copy.to(model_copy.device if hasattr(model_copy, 'device') else 'cpu')
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if np.isnan(train_losses.get('hebbian', 0)) or np.isnan(val_losses['prediction']):
                print("NaN loss encountered, stopping early")
                best_val_loss    = np.nan
                epochs_no_improve = np.nan
                best_model_state  = deepcopy(model_copy.cpu().state_dict())
                break

        best_model_dict['epochs_trained'] = epoch + 1
        best_model_dict['val_pred_loss']  = best_val_loss
        best_model_dict['model_state']    = best_model_state
        return losses_dict, best_model_dict

    def get_model_trial_by_trial_df(self, model, dataset, splits):
        """Run model over the full dataset and return a trial-by-trial DataFrame."""
        model.eval()
        data = {}

        # Full-dataset forward pass (always use flat/concatenated representation)
        if isinstance(dataset, AB_SessionDataset):
            inputs_tensor = dataset.flat_inputs.unsqueeze(0)   # (1, total_trials, 3)
        else:
            inputs_tensor = torch.tensor(
                dataset.subject_df[['forced_choice', 'outcome', 'choice']].values,
                dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            predictions, hidden_states = model(inputs_tensor)
            # hidden states are always extracted — shape (1, total_trials, H)
            for u in range(model.H):
                data[f'hidden_{u+1}'] = hidden_states[:, :, u].flatten().cpu().numpy()
            # gate activations only available for non-vanilla RNNs
            if model.rnn_type != 'vanilla':
                inp = inputs_tensor if model.input_forced_choice else inputs_tensor[:, :, 1:]
                if model.input_encoding == 'bipolar':
                    inp = inp * 2 - 1
                _, gate_activations = model.rnn(inp, return_gate_activations=True)
                for gate, acts in gate_activations.items():
                    for u in range(acts.shape[-1]):
                        data[f'gate_{gate}_{u+1}'] = acts[:, :, u].flatten().cpu().numpy()

        log_probs = predictions.log_softmax(dim=2)
        logits    = (log_probs[:, :, 0] - log_probs[:, :, 1]).flatten()
        data['logit_value']  = logits.cpu().numpy()
        data['logit_past']   = np.concatenate([[np.nan], data['logit_value'][:-1]])
        data['logit_change'] = np.concatenate([[np.nan], np.diff(data['logit_value'])])
        data['prob_A']       = log_probs[:, :, 0].exp().flatten().cpu().numpy()
        data['prob_B']       = log_probs[:, :, 1].exp().flatten().cpu().numpy()

        # Trial-level metadata from subject_df (AB_Dataset) or reconstructed (AB_SessionDataset)
        if isinstance(dataset, AB_SessionDataset):
            # Rebuild a minimal subject_df from the flat tensors for consistent downstream code
            flat = dataset.flat_inputs.cpu().numpy()
            subject_df = pd.DataFrame(flat, columns=['forced_choice', 'outcome', 'choice'])
            subject_df['session_folder_name'] = np.concatenate(
                [np.repeat(name, dataset.session_index[i, 1] - dataset.session_index[i, 0])
                 for i, name in enumerate(dataset.session_names)])
        else:
            subject_df = dataset.subject_df

        for col in subject_df.columns:
            data[col] = subject_df[col].values

        labels = ['A1,R=0', 'A1,R=1', 'A2,R=0', 'A2,R=1']
        data['trial_type'] = [labels[int(c * 2 + o)] for c, o in
                              zip(data['choice'], data['outcome'])]

        # Add split membership label per trial
        data['split'] = dataset.get_split_labels(splits)

        return pd.DataFrame(data)