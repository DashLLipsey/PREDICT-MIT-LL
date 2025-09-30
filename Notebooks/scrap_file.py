# Basic Package Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, r2_score

# imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Non-basic package imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import requests

# Packages I don't understand
from fcd_torch import FCD
import rdkit
from collections import Counter
import gc
import pickle

# Add the Python_files directory to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'Python_files'))

# Now you can import your modules
# import functions_enc as f
import function_depot as fd

# We are working with the June 25 dataset, with the Morgan Fingerprints and cannonical SMILES included
df5 = pd.read_csv("/home/dlipsey/MITLincolnLabs/MIT_LL_data/MIT_LL_data5.csv")
# print(df5.shape)
# df5.head()

# First order of business is to standardize our SMILES column. We want to use canonical smiles rather than SMILES_spectra but 
# we will keep the column name SMILES_spectra for consistency with previous code
df5 = df5.drop('SMILES_spectra', axis=1) # Drop
df5 = df5.rename(columns={'canonical_smiles': 'SMILES_spectra'}) # Rename
cols = df5.columns.tolist()
cols.remove('SMILES_spectra') 
df5 = df5[['SMILES_spectra'] + cols] # Move to front

# Next we want to standardize the Ionization column
# print(df5["Ionization_Mode"].unique()) # Check unique values
df5["Ionization_Mode"] = df5["Ionization_Mode"].replace("'Positive'", "'positive'") # Fix capitaliztion
df5 = df5[df5["Ionization_Mode"] != "'N/A'"] # Remove N/A 
# print(df5["Ionization_Mode"].unique()) # Check unique values

# Remove single quotes from all columns
df5 = df5.applymap(lambda x: x.replace("'", "") if isinstance(x, str) else x)

# Select specific groups for subset
selected_groups = ['Q-Orbitrap-positive', 'Q-TOF-positive', 'LTQ-Orbitrap-positive']

# Create subset with only selected groups
df5_subset = df5[df5['Group'].isin(selected_groups)]

print(df5_subset.shape)
df5_subset.head()

# SPECTRA DATAFRAME
# Create dataframe with spectra using spectrum_string_to_dataframe
df5_spectra = fd.spectrum_string_to_dataframe(df5_subset, spectrum_col='Spectrum', smiles_col='SMILES_spectra')

# Add Group and Response columns by mapping directly from df5_subset
# Create dictionaries for faster lookup
smiles_to_group = df5_subset.set_index('SMILES_spectra')['Group'].to_dict()
smiles_to_response = df5_subset.set_index('SMILES_spectra')['Response'].to_dict()

# Map the values directly
df5_spectra['Group'] = df5_spectra['SMILES_spectra'].map(smiles_to_group)
df5_spectra['Response'] = df5_spectra['SMILES_spectra'].map(smiles_to_response)

print("=== SPECTRA DATAFRAME ===")
print(f"Shape: {df5_spectra.shape}")
print(f"Unique SMILES: {df5_spectra['SMILES_spectra'].nunique()}")
print(f"Columns: {list(df5_spectra.columns[:3])} ... {list(df5_spectra.columns[-3:])}")  # Show first and last few columns


# df5_spectra = pd.read_csv('/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_spectra.csv')
# df5_subset = pd.read_csv('/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_subset.csv')


# Define your parameters
bin_sizes = [0.05, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
save_directory = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes"

# Create all datasets
all_datasets = fd.binning_loop(df5_spectra, df5_subset, bin_sizes, thresholds, save_directory, indx_id_indx=-3, startindx=1, stopindx=-3)