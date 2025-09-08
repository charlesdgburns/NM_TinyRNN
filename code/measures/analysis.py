''' Some code to plot dynamical analysis to investigate different RNNS'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from NM_TinyRNN.code.models.parallelised_training import get_train_info_df

