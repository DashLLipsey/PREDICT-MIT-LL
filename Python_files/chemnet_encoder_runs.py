# Basic Package Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
import functions_enc as f
import function_depot as fd


chemnet_config = {
        'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
        'wandb_project': 'MIT-Lincoln-Lab',
        'gpu':True,
        'embedding_type' : "ChemNet",
        'encoder_type' : "ChemNet Encoder"
    }

print("=== CHEMNET ENCODER TRAINING TEST ===")

# Training and validation dataset split for ChemNet encoder
data = pd.read_pickle('/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes/bin0_1_thresh0_001_df_spectra.pkl')

# Load ChemNet embeddings 
chemnet_embeddings_df = pd.read_csv('/home/dlipsey/MITLincolnLabs/MIT_LL_data/ChemNet_of_df4_QQpos.csv')

print(f"Data shape: {data.shape}")
print(f"ChemNet embeddings shape: {chemnet_embeddings_df.shape}")

# Count occurrences of each SMILES and keep only SMILES with at least 4 entries
counts = data['SMILES_spectra'].value_counts()
valid_smiles = counts[counts >= 4].index
filtered_dataset = data[data['SMILES_spectra'].isin(valid_smiles)].copy()

print(f"After filtering (>=4 spectra per SMILES): {filtered_dataset.shape}")

train_indices = []
test_indices = []

np.random.seed(42) 

for smiles, group in filtered_dataset.groupby('SMILES_spectra'):
    idx = group.index.tolist()
    n = len(idx)
    np.random.shuffle(idx)  # shuffle for randomness
    split = n // 2
    test_idx = idx[:split]
    train_idx = idx[split:]
    # If odd, train gets the extra
    train_indices.extend(train_idx)
    test_indices.extend(test_idx)

train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)

# Add an 'index' column 
train_data['index'] = train_data.index
test_data['index'] = test_data.index

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Use test_data as validation data
val_data = test_data

# ChemNet encoder training
device = fd.set_up_gpu() if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# Set model hyperparameters
batch_size = 128
output_size = 512
num_layers = 6
learning_rate = 0.0001
epochs = 500

# Get spectra columns and embedding columns
spectra_cols = [col for col in data.columns if col != 'SMILES_spectra']
embedding_cols = [col for col in chemnet_embeddings_df.columns if col != 'SMILES_spectra']

print(f"Number of spectra columns: {len(spectra_cols)}")
print(f"Number of embedding columns: {len(embedding_cols)}")

# Find the start and stop indices for spectra columns
start_idx = 1  # Skip SMILES column
stop_idx = len(spectra_cols) + 1  # Include all spectra columns

# Training set
train_embeddings_tensor, train_spectra_tensor, train_indices_tensor = fd.create_dataset_tensors_fixed(
    train_data, chemnet_embeddings_df, device, start_idx=start_idx, stop_idx=stop_idx)

# Validation set
val_embeddings_tensor, val_spectra_tensor, val_indices_tensor = fd.create_dataset_tensors_fixed(
    val_data, chemnet_embeddings_df, device, start_idx=start_idx, stop_idx=stop_idx)

train_dataset = TensorDataset(train_spectra_tensor, train_embeddings_tensor, train_indices_tensor)
val_dataset = TensorDataset(val_spectra_tensor, val_embeddings_tensor, val_indices_tensor)
train_loader_enc = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader_enc = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Set up model parameters
input_size = len(spectra_cols)

# Create ChemNet encoder using the updated architecture
chemnet_encoder = fd.base_Encoder(input_size=input_size, output_size=output_size, num_layers=num_layers).to(device)

# Use MSELoss for continuous embeddings
criterion = nn.MSELoss()
# Now create the complete config with all hyperparameters
chemnet_config = {
        'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
        'wandb_project': 'MIT-Lincoln-Lab',
        'gpu': True,
        'embedding_type': "ChemNet",
        'encoder_type': "ChemNet Encoder",
        # Model hyperparameters
        'batch_size': batch_size,
        'output_size': output_size,
        'num_layers': num_layers,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'Bin': 0.01,
        'Threshold': 0.001
    }
# Train the model with loss tracking
model_chemnet, train_loss_history, val_loss_history = fd.train_model_chemnet_encoder(
    model=chemnet_encoder,
    train_data=train_loader_enc,
    val_data=val_loader_enc,
    epochs=epochs,
    learning_rate=learning_rate,
    criterion=criterion,
    device=device,
    config=chemnet_config
)

print("Training complete! Model can now predict ChemNet embeddings from spectra.")
print("ChemNet encoder test completed successfully!")