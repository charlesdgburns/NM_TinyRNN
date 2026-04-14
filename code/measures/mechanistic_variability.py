""" 
The following script contains code to perform an analysis of mechanistic variability.
This is written for the 2-armed bandit reversal task modelled trial-by-trial.

1. train models on a subjects with multiple train seeds and weight seeds.
2. compare evaluation performance as across train and weight seeds.
3. compare similarity of activations (hidden units and gates).s
4. compare the parameters (weights) of models.

Note that comparisons must be somehow aligned in the hidden unit activations.
"""
#imports
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(Path('/home/charlesdgburns/Coding/')) #set the working directory to the root of the project
##local imports
from NM_TinyRNN.code.models import parallelised_training as pat
from NM_TinyRNN.code.measures import analysis

# GLOBAL VARIABLES 
AB_DATA_PATH = Path("NM_TinyRNN/data/AB_behaviour")
SAVE_PATH = Path("NM_TinyRNN/data/rnns/mech_var")
# FUNCTIONS #

# %1 code to run the training #
def train_models(train_seeds =list(range(1, 21)),
                 weight_seeds =list(range(1, 21)),
                 subjects= ["WS16"]):
    for each_subject in subjects:
        data_path = AB_DATA_PATH / f"{each_subject}"
        for each_train_seed in train_seeds:

            save_path = SAVE_PATH/f"{each_subject}/train_seed_{each_train_seed}"
            #train the standard GRU's
            pat.train_parallel(data_path=data_path,
                               save_path=save_path,
                                train_seed=each_train_seed,
                                weight_seeds=weight_seeds,
                                n_jobs=-1,
                                model_type = "GRU",
                                nonlinearity='tanh',
                                constraint='sparsity')
            #then train the monoGRU's
            #train the standard GRU's
            pat.train_parallel(data_path=data_path,
                               save_path=save_path,
                                train_seed=each_train_seed,
                                weight_seeds=weight_seeds,
                                n_jobs=-1,
                                model_type = "monoGRU",
                                nonlinearity='relu',
                                constraint='energy')
    print("Training complete.")
    return None
train_models()
# %2 code to compare performance across train and weight seeds #
def build_analysis_df(save_path=SAVE_PATH):
    '''Iterates over the saved models and builds a dataframe
    This lets us load the different '''
    analysis_path = Path(save_path)
    data_rows = []

    # Iterate through the structure: subject/train_seed/weight_seed
    for path in analysis_path.glob("*/*/*"):
        if path.is_dir():
            # Extract folder levels
            subject_id = path.parts[-3]
            train_seed = path.parts[-2][-1]
            weight_seed = path.parts[-1][-1]

            # Find the info file to determine the MODEL_ID
            info_files = list(path.glob("*_info.json"))
            for info_file in info_files:
                # Extract MODEL_ID (e.g., "model123_info.json" -> "model123")
                model_id = info_file.name.replace("_info.json", "")
                
                # Construct absolute paths for all expected data types
                data_rows.append({
                    "subject_id": subject_id,
                    "train_seed": train_seed,
                    "weight_seed": weight_seed,
                    "model_id": model_id,
                    "info_path": path / f"{model_id}_info.json",
                    "model_state_path": path / f"{model_id}_model_state.pth",
                    "training_losses_path": path / f"{model_id}_training_losses.htsv",
                    "trials_data_path": path / f"{model_id}_trials_data.htsv"
                })

    return pd.DataFrame(data_rows)

def add_eval_CE(analysis_df):
    '''Open the info dictionary and add the cross-entropy loss predicted on the evaluation data.'''
    eval_CEs = []
    for each_row in analysis_df.itertuples():
        print(each_row.info_path)
        info_dict = analysis.load_data(str(each_row.info_path))
        eval_CEs.append(info_dict['eval_pred_loss'])
    analysis_df['eval_CE'] = eval_CEs
    return analysis_df

analysis_df = build_analysis_df(SAVE_PATH)
analysis_df = add_eval_CE(analysis_df)

sns.stripplot(analysis_df,x='train_seed',y='eval_CE', hue = 'model_id')
plt.show()
# %3 code to compare similarity of activations (hidden units and gates) #

    
def compute_similarities(analysis_df):
    '''Compute similarity of activations within each model across weight and train seeds.'''
    #we will need to load the model states and extract the activations for a common set of trials
    #then we can compute similarity measures (e.g., cosine similarity, CKA, etc.) between the activations of different models
    
    #we create a dataframe to store similarities
    
    model_ids = []
    model_A_weight_seeds = []
    model_A_train_seeds = []
    model_B_weight_seeds = []
    model_B_train_seeds = []
    hidden_state_similarities = []
    update_gate_similarities = []
    reset_gate_similarities = []


    unique_model_ids = analysis_df['model_id'].unique()
    
    for each_model in unique_model_ids:
        model_rows = analysis_df[analysis_df['model_id'] == each_model]
        #we can compute similarities between all pairs of models with the same model_id
        for i, row_i in model_rows.iterrows():
            for j, row_j in model_rows.iterrows():
                if i < j: #to avoid redundant comparisons
                    
                    #load activations for both models
                    data_A = analysis.load_data(row_i.trials_data_path)
                    data_A = _standardize_hidden_units(data_A) #this ensures that hidden_1 always correlates with logit_value, and hidden_2 with prob_A, across all models. This is necessary for the similarity comparisons to be meaningful.
                    data_B = analysis.load_data(row_j.trials_data_path)
                    data_B = _standardize_hidden_units(data_B) #same reordering for model B
                    if data_A is None or data_B is None:
                        #exclude model if no unit correlates with the logits.
                        hidden_state_similarity = np.nan
                        update_gate_similarity = np.nan
                        reset_gate_similarity = np.nan
                    else:
                        #compute similarity (this is a placeholder, you would replace this with your actual similarity computation)#
                        for each_component in ['hidden', 'gate_update', 'gate_reset']:
                            if not any([x.startswith(each_component) for x in data_A.columns]):
                                #if the component doesn't exist in either model, we set similarity to nan
                                if each_component == 'hidden':
                                    hidden_state_similarity = np.nan
                                elif each_component == 'gate_update':
                                    update_gate_similarity = np.nan
                                elif each_component == 'gate_reset':
                                    reset_gate_similarity = np.nan
                                continue
                            component_A = np.concat([data_A[x] for x in data_A.columns if x.startswith(each_component)])
                            component_B = np.concat([data_B[x] for x in data_B.columns if x.startswith(each_component)])
                            similarity = np.corrcoef(component_A, component_B)[0, 1]
                            if each_component == 'hidden':
                                hidden_state_similarity = similarity
                            elif each_component == 'gate_update':
                                update_gate_similarity = similarity
                            elif each_component == 'gate_reset':
                                reset_gate_similarity = similarity
                    
                    #store the results in lists
                    model_ids.append(each_model)
                    model_A_train_seeds.append(row_i.train_seed)
                    model_A_weight_seeds.append(row_i.weight_seed)
                    model_B_train_seeds.append(row_j.train_seed)
                    model_B_weight_seeds.append(row_j.weight_seed)
                    hidden_state_similarities.append(hidden_state_similarity)
                    update_gate_similarities.append(update_gate_similarity)
                    reset_gate_similarities.append(reset_gate_similarity)
    similarity_df = pd.DataFrame({
        "model_id": model_ids,
        "model_A_train_seed": model_A_train_seeds,
        "model_A_weight_seed": model_A_weight_seeds,
        "model_B_train_seed": model_B_train_seeds,
        "model_B_weight_seed": model_B_weight_seeds,
        "hidden_state_similarity": hidden_state_similarities,
        "update_gate_similarity": update_gate_similarities,
        "reset_gate_similarity": reset_gate_similarities
    })
    return similarity_df

def _standardize_hidden_units(trials_df: pd.DataFrame, verbose: bool = False):
    """
    Standardizes a 2-unit network so hidden_1 is positively correlated with logits.
    Detects tanh vs ReLU automatically to apply the correct inversion.
    """
    # 1. Calculate correlations
    corr1 = np.corrcoef(trials_df.hidden_1, trials_df.logit_value)[0,1]
    corr2 = np.corrcoef(trials_df.hidden_2, trials_df.logit_value)[0,1]
    if verbose:
        print(f"corr1:{corr1}\n corr2: {corr2}")
    #treat nan's as zero correlation, which means we won't reorder or flip the units, but we also won't exclude the model from the analysis.
    if np.isnan(corr1):
        corr1 = -np.inf
    if np.isnan(corr2):
        corr2 = -np.inf
    # 2. Permutation: Ensure Unit 1 is the 'most positive' correlation
    if corr2 > corr1:
        if verbose: print("Swapping Unit 1 and Unit 2.")
        for prefix in ['hidden_', 'gate_update_', 'gate_reset_']:
            col1, col2 = f"{prefix}1", f"{prefix}2"
            if col1 in trials_df.columns and col2 in trials_df.columns:
                trials_df[col1], trials_df[col2] = trials_df[col2], trials_df[col1]
        corr1 = corr2 # Update after swap

    # 3. Inversion: If Unit 1 is still negative, flip it based on activation type
    if corr1 < -0.1:
        # Check if it's tanh (centered) or ReLU (positive-only)
        h1_min = trials_df.hidden_1.min()
        
        if h1_min < -0.1:
            # Tanh logic: Simply flip the sign around 0
            if verbose: print("Inverting Unit 1 (tanh detected).")
            trials_df['hidden_1'] = -trials_df['hidden_1']
        else:
            # ReLU/Sigmoid logic: Flip within the observed range
            if verbose: print("Inverting Unit 1 (ReLU/Sigmoid detected).")
            trials_df['hidden_1'] = trials_df['hidden_1'].max() - trials_df['hidden_1']
            
    return trials_df
best_idx = analysis_df.groupby(['model_id','train_seed'])['eval_CE'].idxmin()
best_models_df = analysis_df.loc[best_idx]
sim_df = compute_similarities(best_models_df)

fig, ax = plt.subplots(1,3,figsize=(15,5))
sns.stripplot(best_models_df,x='model_id',y='eval_CE', ax=ax[0])
sns.stripplot(sim_df, x='model_id',y='hidden_state_similarity', ax=ax[1])
sns.stripplot(sim_df, x='model_id',y='update_gate_similarity', ax = ax[2])

ax[0].set(title='performance')
ax[1].set(title='hidden states')
ax[2].set(title='gating mechanism')
fig.tight_layout()
plt.show()

for each_model in best_models_df.itertuples():
    trials_data = analysis.load_data(each_model.trials_data_path)
    trials_data = _standardize_hidden_units(trials_data)
    fig, ax = plt.subplots(figsize = (3,3))
    sns.scatterplot(trials_data, x='hidden_1',y='hidden_2',hue='logit_value',ax=ax, palette='coolwarm')
    fig.suptitle(each_model.model_id)
# %4 code to compare the parameters (weights) of models #





