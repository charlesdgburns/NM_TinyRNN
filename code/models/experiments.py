"""Experiment dataframes shared by training and analysis.

Each dataframe row describes one model configuration, subject/group and outer
fold. It holds model settings, source/result paths, completion status, sequence
length and batch size. It contains no SLURM resource or submission settings.
The get_*_info_df names are retained for compatibility; use experiment_df for
variables holding these dataframes. measures.analysis expands dataframes into
one row per saved inner-fold model before performance/other analyses.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from NM_TinyRNN.code.models import datasets, nested_cv, training

SAVE_PATH = Path('./NM_TinyRNN/data/rnns')
PROCESSED_DATA_PATH = Path('./NM_TinyRNN/data/AB_behaviour/')


def training_folder_name(sequence_length=datasets.SEQUENCE_LENGTH,
                         batch_size=training.BATCH_SIZE, prefix='nested_DA'):
    """S = predictions per sequence; B = sequences per training minibatch."""
    sequence_length = datasets.validate_sequence_length(sequence_length)
    if (isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    return f'{prefix}_S{sequence_length}_B{batch_size}'


def _finish_experiment_df(df_dict, sequence_length, batch_size):
    experiment_df = pd.DataFrame(df_dict)
    experiment_df['sequence_length'] = sequence_length
    experiment_df['batch_size'] = batch_size
    return experiment_df


def get_DA_info_df(processed_data_path = PROCESSED_DATA_PATH,
                    save_path = SAVE_PATH, folder_name = None,
                    sequence_length=datasets.SEQUENCE_LENGTH,
                    batch_size=training.BATCH_SIZE, model_types=None):
    '''Organise the architecture information in a large dataframe.
    Key arguments here are 'model_id', 'data_path','save_path' and 'outer_loop_n'
    '''
    default_folder = training_folder_name(sequence_length, batch_size)
    folder_name = default_folder if folder_name is None else folder_name
    #see each subject
    df_dict = {'subject_id':[],'outer_loop_n':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'constraint':[],'decoder_bias':[],
               'input_encoding':[],'input_forced_choice':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}

    if model_types is not None:
        model_list = list(model_types)
    elif folder_name == 'nested_DA_128': # legacy analysis notebook architecture set
        model_list = ['vanilla','GRU', 'lightGRU', 'monoGRU','constGate','monoGRU_abs','monoGRU_no_hidden','monoGRU_hidden_only']
    else:
        model_list = ['GRU','monoGRU']

    for subdir in processed_data_path.iterdir():
        subject_ID = subdir.stem
        if not "WS" in subject_ID:
            continue
        data_path = processed_data_path/subject_ID
        if not subdir.is_dir():
            continue

        for outer_loop_n in range(1,nested_cv.N_OUTER_LOOPS+1): #10 loops is recommended

            for model_type in model_list:
                #Default parameters
                hidden_sizes = [1,2]
                input_encodings = ['unipolar']
                nonlinearities = ['relu']
                input_forced_choices = [True]
                decoder_biases = [True]
                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored

                if model_type == 'GRU':
                    nonlinearities = ['relu','tanh']
                    input_encodings = ['unipolar'] #'encoder','onehot'
                if model_type == 'vanilla':
                    nonlinearities = ['relu','tanh']
                    input_encodings = ['unipolar'] #'encoder','onehot'
                if  model_type == 'monoGRU_abs':
                    nonlinearities = ['tanh']
                    hidden_sizes = [1]
                if model_type in ['monoGRU_no_hidden','monoGRU_hidden_only']:
                    hidden_sizes = [2]
                    decoder_biases = [False]
                if model_type == 'monoGRU':
                    decoder_biases = [True,False]

                for nonlinearity in nonlinearities:
                    for input_forced_choice in input_forced_choices:
                        for input_encoding in input_encodings:
                            for hidden_size in hidden_sizes:
                                if input_encoding == 'unipolar' and nonlinearity =='relu' and model_type in ['vanilla', 'GRU','monoGRU']:
                                    decoder_biases = [True,False]
                                elif model_type in ['monoGRU_no_hidden','monoGRU_hidden_only']:
                                    decoder_biases = [False]
                                else:
                                    decoder_biases = [True]
                                for decoder_bias in decoder_biases:
                                    model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                                    if input_forced_choice:
                                        model_id+= '_forced'
                                    if decoder_bias == False:
                                        model_id+= '_ndb'

                                    constraint = 'energy' if nonlinearity == 'relu' else 'sparsity'
                                    model_save_path = Path(save_path)/folder_name/subject_ID/model_type/constraint
                                    completed = 1
                                    for inner_loop_n in range(0,nested_cv.N_OUTER_LOOPS-1):
                                        completed *= (model_save_path/f'outer_fold_{outer_loop_n}'/f'inner_fold_{inner_loop_n}'/f'{model_id}_trials_data.htsv').exists()
                                    for k,v in zip(df_dict.keys(),
                                        [subject_ID,outer_loop_n,model_type,hidden_size,
                                        nonlinearity,constraint,decoder_bias,
                                        input_encoding, input_forced_choice,
                                        nm_size,nm_dim,nm_mode,
                                        model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                                        df_dict[k].append(v)

    return _finish_experiment_df(df_dict, sequence_length, batch_size)

def get_job_info_df(processed_data_path = PROCESSED_DATA_PATH,
                    save_path = SAVE_PATH, folder_name=None,
                    sequence_length=datasets.SEQUENCE_LENGTH,
                    batch_size=training.BATCH_SIZE):
    '''Organise the architecture information in a large dataframe.
    Key arguments here are 'model_id', 'data_path','save_path' and 'outer_loop_n'
    '''
    default_folder = training_folder_name(sequence_length, batch_size)
    folder_name = default_folder if folder_name is None else folder_name
    #see each subject
    df_dict = {'subject_id':[],'outer_loop_n':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'constraint':[],'decoder_bias':[],
               'input_encoding':[],'input_forced_choice':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}

    for subdir in processed_data_path.iterdir():
        subject_ID = subdir.stem
        if not "WS" in subject_ID:
            continue
        data_path = processed_data_path/subject_ID
        if not subdir.is_dir():
            continue

        for outer_loop_n in range(1,nested_cv.N_OUTER_LOOPS+1): #10 loops is recommended
            for model_type in ['monoGRU','GRU','vanilla']:# 'vanilla', 'constGate', 'monoGRU_abs'  ['vanilla','GRU','LSTM','NMRNN', 'monoGRU','monoGRU2','stereoGRU']:
                decoder_biases = [True,False]
                hidden_sizes = [1,2]
                nonlinearities = ['relu','tanh']
                #nonlinearity = 'relu' if constraint =='energy' else 'tanh'
                input_encodings = ['unipolar','onehot']
                input_forced_choices = [True,False]
                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                #if model_type in ['GRU','monoGRU']:
                #    input_forced_choices = [True, False]
                #    decoder_biases = [True,False]
                if model_type == 'monoGRU_abs':
                    nonlinearities = ['tanh']
                    hidden_sizes = [1]
                if model_type in ['constGate','lightGRU']:
                    input_encodings = ['unipolar']
                    nonlinearities = ['relu']
                    input_forced_choices = [True]
                    decoder_biases = [False]
                for nonlinearity in nonlinearities:
                    for input_forced_choice in input_forced_choices:
                        for input_encoding in input_encodings:
                            constraint= 'energy' if nonlinearity == 'relu' else 'sparsity'
                            for hidden_size in hidden_sizes:
                                for decoder_bias in decoder_biases:
                                    model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                                    if input_forced_choice:
                                        model_id+= '_forced'
                                    if decoder_bias == False:
                                        model_id+= '_ndb'
                                    model_save_path = Path(save_path)/folder_name/subject_ID/model_type/constraint
                                    completed = 1
                                    for inner_loop_n in range(0,nested_cv.N_OUTER_LOOPS-1):
                                        completed *= (model_save_path/f'outer_fold_{outer_loop_n}'/f'inner_fold_{inner_loop_n}'/f'{model_id}_trials_data.htsv').exists()
                                    for k,v in zip(df_dict.keys(),
                                        [subject_ID,outer_loop_n,model_type,hidden_size,
                                        nonlinearity,constraint,decoder_bias,
                                        input_encoding, input_forced_choice,
                                        nm_size,nm_dim,nm_mode,
                                        model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                                        df_dict[k].append(v)

    return _finish_experiment_df(df_dict, sequence_length, batch_size)

def get_test_info_df(processed_data_path = PROCESSED_DATA_PATH,
                    save_path = SAVE_PATH, folder_name=None,
                    sequence_length=datasets.SEQUENCE_LENGTH,
                    batch_size=training.BATCH_SIZE):
    '''Organise the architecture information in a large dataframe.
    Key arguments here are 'model_id', 'data_path','save_path' and 'outer_loop_n'
    '''
    default_folder = training_folder_name(sequence_length, batch_size)
    folder_name = default_folder if folder_name is None else folder_name
    #see each subject
    df_dict = {'subject_id':[],'outer_loop_n':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'constraint':[],'decoder_bias':[],
               'input_encoding':[],'input_forced_choice':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}

    for subdir in processed_data_path.iterdir():
        subject_ID = subdir.stem
        if "WS" in subject_ID or "MA" in subject_ID: ##OBS - ONLY SIMULATED DATA FOR TESTING
            continue
        data_path = processed_data_path/subject_ID
        if not subdir.is_dir():
            continue

        for outer_loop_n in range(1,11): #10 loops is recommended. Minimum is 3.
            for model_type in ['GRU','monoGRU']: #,'constGate','vanilla'
                nonlinearities = ['relu','tanh']
                input_encodings = ['unipolar','onehot']
                decoder_biases = [True, False]
                input_forced_choices = [True, False]
                hidden_sizes = [1,2]
                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                for input_forced_choice in input_forced_choices:
                    for nonlinearity in nonlinearities:
                        for input_encoding in input_encodings:
                            for decoder_bias in decoder_biases:
                                constraint= 'energy' if nonlinearity == 'relu' else 'sparsity'
                                for hidden_size in hidden_sizes:
                                    model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                                    if input_forced_choice:
                                        model_id+= '_forced'
                                    if decoder_bias == False:
                                        model_id+= '_ndb'
                                    model_save_path = Path(save_path)/folder_name/subject_ID/model_type/constraint
                                    completed = 1
                                    for inner_loop_n in range(0,nested_cv.N_OUTER_LOOPS-1):
                                        completed *= (model_save_path/f'outer_fold_{outer_loop_n}'/f'inner_fold_{inner_loop_n}'/f'{model_id}_trials_data.htsv').exists()
                                    for k,v in zip(df_dict.keys(),
                                        [subject_ID,outer_loop_n,model_type,hidden_size,
                                        nonlinearity,constraint,decoder_bias,
                                        input_encoding, input_forced_choice,
                                        nm_size,nm_dim,nm_mode,
                                        model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                                        df_dict[k].append(v)

    return _finish_experiment_df(df_dict, sequence_length, batch_size)


def get_ws_all_info_df(processed_data_path = PROCESSED_DATA_PATH,
                    save_path = SAVE_PATH, folder_name=None,
                    sequence_length=datasets.SEQUENCE_LENGTH,
                    batch_size=training.BATCH_SIZE):
    '''Organise the architecture information in a large dataframe.
    Key arguments here are 'model_id', 'data_path','save_path' and 'outer_loop_n'
    '''
    default_folder = training_folder_name(sequence_length, batch_size, prefix="test_input_encodings")
    folder_name = default_folder if folder_name is None else folder_name
    #see each subject
    df_dict = {'subject_id':[],'outer_loop_n':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'constraint':[], 'decoder_bias':[],
               'input_encoding':[],'input_forced_choice':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
    #generate a list of subjects to group together
    subject_ids = ['WS_all']
    for subject_id in subject_ids:
        data_path = processed_data_path/'WS'
        for outer_loop_n in range(1,11): #10 loops is recommended. Minimum is 3.
           for model_type in ['monoGRU','GRU', 'monoGRU_abs','stereoGRU','stereoGRU2','stereoGRUx']:#['vanilla','GRU','LSTM','NMRNN', 'monoGRU','monoGRU2','stereoGRU']:
                decoder_bias = True
                hidden_sizes = [1,2]
                nonlinearities = ['relu','tanh', 'softplus']
                #nonlinearity = 'relu' if constraint =='energy' else 'tanh'
                input_encodings = ['unipolar','onehot']
                input_forced_choice = False
                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                if model_type == 'monoGRU_abs':
                    nonlinearities = ['tanh']
                for nonlinearity in nonlinearities:
                    for input_encoding in input_encodings:
                        for input_forced_choice in [True, False]:
                            constraint= 'energy' if nonlinearity == 'relu' else 'sparsity'
                            for hidden_size in hidden_sizes:
                                model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                                if input_forced_choice:
                                    model_id+='_forced'
                                model_save_path = Path(save_path)/folder_name/subject_id/model_type/constraint
                                completed = 1
                                for inner_loop_n in range(0,nested_cv.N_OUTER_LOOPS-1):
                                    completed *= (model_save_path/f'outer_fold_{outer_loop_n}'/f'inner_fold_{inner_loop_n}'/f'{model_id}_trials_data.htsv').exists()
                                for k,v in zip(df_dict.keys(),
                                    [subject_id,outer_loop_n,model_type,hidden_size,
                                    nonlinearity, constraint, decoder_bias,
                                    input_encoding, input_forced_choice,
                                    nm_size,nm_dim,nm_mode,
                                    model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                                    df_dict[k].append(v)

    return _finish_experiment_df(df_dict, sequence_length, batch_size)


