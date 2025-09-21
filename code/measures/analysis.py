''' Some code to plot dynamical analysis to investigate different RNNS'''

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from NM_TinyRNN.code.models import training

# Global variables # 

DATA_PATH = Path('./NM_TinyRNN/data/')
RNNS_PATH = DATA_PATH/'rnns' #this folder should contain a folder per subject, and then all models fit to said subject.

# functions #
# we want to get an overview of models that are available


# Analyzer class #
# we want to load all the models for a subject in an accessible way

class Analyzer():
    '''wrapper class for a model that allows'''
class model_data():
    '''This is a class that gathers data for each model as attributes'''
    def __init__(self, subject_id, model_id):
       return None 
        

# utilities #

def load_data(filepath):
    if filepath.endswith(".json"):
        with open(filepath, "r") as f:
            data = json.load(f)
    elif filepath.endswith(".htsv"):
        # assuming htsv = tab-separated values
        data = pd.read_csv(filepath, sep="\t")
    elif filepath.endswith(".pth"):
        data = torch.load(filepath, weights_only = True)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")
    return data