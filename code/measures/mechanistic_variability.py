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

#os.chdir(Path('/home/charlesdgburns/Coding/')) #set the working directory to the root of the project
##local imports
from NM_TinyRNN.code.models import parallelised_training as pat
from NM_TinyRNN.code.measures import analysis

# GLOBAL VARIABLES 
AB_DATA_PATH = Path("NM_TinyRNN/data/AB_behaviour")
SAVE_PATH = Path("NM_TinyRNN/data/rnns/mech_var")
# FUNCTIONS #

# %1 code to run the training #
def train_models(train_seeds =list(range(1, 6)),
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
            #train standard GRU with 'biological constraints'
            pat.train_parallel(data_path=data_path,
                               save_path=save_path,
                                train_seed=each_train_seed,
                                weight_seeds=weight_seeds,
                                n_jobs=-1,
                                model_type = "GRU",
                                nonlinearity='relu',
                                constraint='energy')
            #then train the monoGRU's
            pat.train_parallel(data_path=data_path,
                               save_path=save_path,
                                train_seed=each_train_seed,
                                weight_seeds=weight_seeds,
                                n_jobs=-1,
                                model_type = "monoGRU",
                                nonlinearity='relu',
                                constraint='energy')
            
            pat.train_parallel(data_path=data_path,
                               save_path=save_path,
                               train_seed=each_train_seed,
                               weight_seeds=weight_seeds,
                                n_jobs=-1,
                                model_type = "vanilla",
                                nonlinearity='relu',
                               constraint='sparsity')
            
            pat.train_parallel(data_path=data_path,
                               save_path=save_path,
                                train_seed=each_train_seed,
                               weight_seeds=weight_seeds,
                                n_jobs=-1,
                                model_type = "constGate",
                                nonlinearity='relu',
                                constraint='energy')
            
            ##vanilla RNN for comparison
            pat.train_parallel(data_path=data_path,
                               save_path=save_path,
                                train_seed=each_train_seed,
                                weight_seeds=weight_seeds,
                                n_jobs=-1,
                                model_type = "vanilla",
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

def add_performance(analysis_df):
    '''Open the info dictionary and add the cross-entropy loss predicted on the evaluation data.'''
    evaluation_CEs = []
    validation_CEs = []
    for each_row in analysis_df.itertuples():
        print(each_row.info_path)
        info_dict = analysis.load_data(str(each_row.info_path))
        evaluation_CEs.append(info_dict['eval_pred_loss'])
        validation_CEs.append(info_dict['best_val_pred_loss'])
    analysis_df['eval_CE'] = evaluation_CEs
    analysis_df['val_CE'] = validation_CEs
    return analysis_df

analysis_df = build_analysis_df(SAVE_PATH)
analysis_df = add_performance(analysis_df)

best_idx = analysis_df.groupby(['model_id','train_seed'])['val_CE'].idxmin()
best_models_df = analysis_df.loc[best_idx]
sns.stripplot(analysis_df,x='train_seed',y='eval_CE', hue = 'model_id')
plt.show()

fig, ax= plt.subplots()
sns.stripplot(best_models_df, x='model_id',y='eval_CE',
              hue='model_id',dodge=True, legend=True)
sns.move_legend(ax,"upper left", bbox_to_anchor=(1, 1))
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
    logit_similarities = []

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
                        logit_similarity = np.nan
                    else:
                        #compute similarity (this is a placeholder, you would replace this with your actual similarity computation)#
                        for each_component in ['hidden', 'gate_update', 'gate_reset','logit_value']:
                            if not any([x.startswith(each_component) for x in data_A.columns]):
                                #if the component doesn't exist in either model, we set similarity to nan
                                if each_component == 'gate_update':
                                    update_gate_similarity = np.nan
                                elif each_component == 'gate_reset':
                                    reset_gate_similarity = np.nan
                                continue
                            try:
                                component_A = np.concat([data_A[x] for x in data_A.columns if x.startswith(each_component)])
                                component_B = np.concat([data_B[x] for x in data_B.columns if x.startswith(each_component)])
                                similarity = np.corrcoef(component_A, component_B)[0, 1]
                            except Exception as e:
                                print(f"Error {each_component}: {e} \n {row_i.trials_data_path} vs {row_j.trials_data_path}")
                                similarity = np.nan
                            if each_component == 'hidden':
                                hidden_state_similarity = similarity
                            elif each_component == 'gate_update':
                                update_gate_similarity = similarity
                            elif each_component == 'gate_reset':
                                reset_gate_similarity = similarity
                            elif each_component == 'logit_value':
                                logit_similarity = similarity
                    
                    #store the results in lists
                    model_ids.append(each_model)
                    model_A_train_seeds.append(row_i.train_seed)
                    model_A_weight_seeds.append(row_i.weight_seed)
                    model_B_train_seeds.append(row_j.train_seed)
                    model_B_weight_seeds.append(row_j.weight_seed)
                    hidden_state_similarities.append(hidden_state_similarity)
                    update_gate_similarities.append(update_gate_similarity)
                    reset_gate_similarities.append(reset_gate_similarity)
                    logit_similarities.append(logit_similarity)

    similarity_df = pd.DataFrame({
        "model_id": model_ids,
        "model_A_train_seed": model_A_train_seeds,
        "model_A_weight_seed": model_A_weight_seeds,
        "model_B_train_seed": model_B_train_seeds,
        "model_B_weight_seed": model_B_weight_seeds,
        "hidden_state_similarity": hidden_state_similarities,
        "update_gate_similarity": update_gate_similarities,
        "reset_gate_similarity": reset_gate_similarities,
        "logit_similarity": logit_similarities
    })
    return similarity_df

def _standardize_hidden_units(trials_df: pd.DataFrame, verbose: bool = False):
    """
    Standardizes a 2-unit network:
    1. hidden_1: Positively correlated with logits.
    2. hidden_2: Negatively correlated with logits.
    Detects tanh vs ReLU for correct inversion.
    """
    # 1. Calculate correlations
    corr1 = np.corrcoef(trials_df.hidden_1, trials_df.logit_value)[0,1]
    corr2 = np.corrcoef(trials_df.hidden_2, trials_df.logit_value)[0,1]
    
    # Treat dead units (NaN correlation) as -inf to push them to hidden_2 slot
    corr1 = -np.inf if np.isnan(corr1) else corr1
    corr2 = -np.inf if np.isnan(corr2) else corr2

    if verbose:
        print(f"Initial - corr1: {corr1:.3f}, corr2: {corr2:.3f}")

    # 2. Permutation: Ensure Unit 1 is the 'most positive' correlation
    if corr2 > corr1:
        if verbose: print("Swapping Unit 1 and Unit 2.")
        for prefix in ['hidden_', 'gate_update_', 'gate_reset_']:
            col1, col2 = f"{prefix}1", f"{prefix}2"
            if col1 in trials_df.columns and col2 in trials_df.columns:
                trials_df[col1], trials_df[col2] = trials_df[col2], trials_df[col1]
        # Re-assign after swap
        corr1, corr2 = corr2, corr1

    # 3. Align Hidden 1 (Target: Positive Correlation)
    if corr1 < 0: # If even the 'best' unit is negative, flip it
        h1_min = trials_df.hidden_1.min()
        if h1_min < -0.1: # Tanh
            trials_df['hidden_1'] = -trials_df['hidden_1']
        else: # ReLU
            trials_df['hidden_1'] = trials_df['hidden_1'].max() - trials_df['hidden_1']
        if verbose: print("Inverted Unit 1 to be positive.")

    # 4. Align Hidden 2 (Target: Negative Correlation)
    # Re-calculate corr2 because a swap might have happened, 
    # but we don't flip if it's a dead unit (-np.inf)
    corr2_current = np.corrcoef(trials_df.hidden_2, trials_df.logit_value)[0,1]
    if verbose:        print(f"Post-Unit1 Alignment - corr2: {corr2_current:.3f}")
    if not np.isnan(corr2_current) and corr2_current > 0.0:
        h2_min = trials_df.hidden_2.min()
        if h2_min < -0.1: # Tanh
            trials_df['hidden_2'] = -trials_df['hidden_2']
        else: # ReLU
            trials_df['hidden_2'] = trials_df['hidden_2'].max() - trials_df['hidden_2']
        if verbose: print("Inverted Unit 2 to be negative.")
            
    return trials_df

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
def parameter_contribution_df(best_models_df):
    '''Input: a pandas dataframe where each row corresponds to a model with 'model_state_path' column'''
    contributions_dict = {'model_id':[],
                        'train_seed':[],
                        'weight_seed':[],
                        'variable':[],
                        'value':[]}
    for each_model in best_models_df.itertuples():
        model_params = analysis.load_data(each_model.model_state_path)
        params_dict = {k:v.numpy() for k,v in model_params.items()}
        
        for each_input in ['outcome','past_choice','past_hidden']:
            for each_output in ['update_gate','reset_gate','hidden_state']:
                if each_output == 'hidden_state':
                    param_keys = ['rnn.W_ih', 'rnn.W_hh']
                elif each_output == 'update_gate':
                    param_keys = ['rnn.W_iz', 'rnn.W_hz']
                elif each_output == 'reset_gate':
                    param_keys = ['rnn.W_ir', 'rnn.W_hr']

                contributions_dict['variable'].append(f"{each_input}_to_{each_output}")

                if not np.all([x in params_dict.keys() for x in param_keys]):
                    contributions_dict['value'].append(np.nan) #if the model doesn't have the relevant component, we set contribution to nan
                else: #if the model has a given component, we compute the weighed contribution of each input to that component as the sum of the absolute values of the relevant weights.
                    total_abs_weights = np.sum([np.sum(np.abs(params_dict[k])) for k in param_keys])
                    
                    if each_input == 'outcome':
                        all_input_weights = params_dict[param_keys[0]] #the input weights, hardcoded as first in the list above.
                        input_weights = all_input_weights[0,:] #the weights from the outcome input, hardcoded as the first row of the input weights. This must be consistent with datasets.py and training.py
                    if each_input == 'past_choice':
                        all_input_weights = params_dict[param_keys[0]]
                        input_weights = all_input_weights[1,:] #the weights from the past choice input, hardcoded as the second row of the input weights. This must be consistent with datasets.py and training.py
                    if each_input == 'past_hidden':
                        all_input_weights = params_dict[param_keys[1]] #the recurrent weights, hardcoded as second in the list above.
                        input_weights = all_input_weights #we sum across the hidden units to get an overall contribution of the past hidden state to the given component.
                    #here we normalise and add this to our dictionary
                    contributions_dict['value'].append(np.sum(np.abs(input_weights)) / total_abs_weights)

                contributions_dict['model_id'].append(each_model.model_id)
                contributions_dict['train_seed'].append(each_model.train_seed)
                contributions_dict['weight_seed'].append(each_model.weight_seed)


    return pd.DataFrame(contributions_dict)
    

cont_df = parameter_contribution_df(best_models_df)

update_gate_df = cont_df[['update' in x for x in cont_df.variable]]
fig, ax = plt.subplots(figsize=(10,5))
sns.stripplot(update_gate_df, x='variable', y='value', 
              hue='model_id',dodge=True, ax = ax)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.show()

## example logit plot


min_idx = best_models_df.query('model_id=="2_unit_vanilla_relu_unipolar"').eval_CE.idxmin()

model = best_models_df.loc[min_idx]
params = analysis.load_data(model.model_state_path)

trials_data = analysis.load_data(model.trials_data_path)

sns.scatterplot(trials_data, x= 'logit_past',y='logit_change',hue='trial_type')

fig.tight_layout()