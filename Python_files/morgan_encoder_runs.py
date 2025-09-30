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


print("=== MORGAN ENCODER TRAINING TEST ===")

# Training and validation dataset split for Morgan encoder
data = pd.read_pickle('/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes/bin0_1_thresh0_001_df_spectra.pkl')

# Load Morgan embeddings 
morgan_embeddings_df = pd.read_csv('/home/dlipsey/MITLincolnLabs/MIT_LL_data/Morgan_fp_df4_QQpos.csv')

print(f"Data shape: {data.shape}")
print(f"Morgan embeddings shape: {morgan_embeddings_df.shape}")

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

# Morgan encoder training
device = fd.set_up_gpu() if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# Set model hyperparameters
batch_size = 128
output_size = 2048  # Morgan fingerprints are typically 2048-bit
num_layers = 6
learning_rate = 0.0001
epochs = 500

# Get spectra columns and embedding columns
spectra_cols = [col for col in data.columns if col != 'SMILES_spectra']
embedding_cols = [col for col in morgan_embeddings_df.columns if col != 'SMILES_spectra']

print(f"Number of spectra columns: {len(spectra_cols)}")
print(f"Number of embedding columns: {len(embedding_cols)}")

# Find the start and stop indices for spectra columns
start_idx = 1  # Skip SMILES column
stop_idx = len(spectra_cols) + 1  # Include all spectra columns

# Training set
train_embeddings_tensor, train_spectra_tensor, train_indices_tensor = fd.create_dataset_tensors_fixed(
    train_data, morgan_embeddings_df, device, start_idx=start_idx, stop_idx=stop_idx)

# Validation set
val_embeddings_tensor, val_spectra_tensor, val_indices_tensor = fd.create_dataset_tensors_fixed(
    val_data, morgan_embeddings_df, device, start_idx=start_idx, stop_idx=stop_idx)

train_dataset = TensorDataset(train_spectra_tensor, train_embeddings_tensor, train_indices_tensor)
val_dataset = TensorDataset(val_spectra_tensor, val_embeddings_tensor, val_indices_tensor)
train_loader_enc = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader_enc = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Set up model parameters
input_size = len(spectra_cols)

# Create Morgan encoder using the updated architecture
morgan_encoder = fd.Morgan_fp_Encoder(input_size=input_size, output_size=output_size, num_layers=num_layers).to(device)

criterion = nn.MSELoss()

# Now create the complete config with all hyperparameters
morgan_config = {
        'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
        'wandb_project': 'MIT-Lincoln-Lab',
        'gpu': True,
        'embedding_type': "Morgan",
        'encoder_type': "Morgan Encoder",
        # Model hyperparameters
        'batch_size': batch_size,
        'output_size': output_size,
        'num_layers': num_layers,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'Bin': 0.01,
        'Threshold': 0.001,
    }

# Train the model with loss tracking
model_morgan, train_loss_history, val_loss_history = fd.train_model_morgan_fp_encoder(
    model=morgan_encoder,
    train_data=train_loader_enc,
    val_data=val_loader_enc,
    epochs=epochs,
    learning_rate=learning_rate,
    criterion=criterion,
    device=device,
    config=morgan_config
)

print("Training complete! Model can now predict Morgan embeddings from spectra.")
print("Morgan encoder test completed successfully!")

# Function for absolute percentage error
def absolute_percentage_error_loss(predicted, actual, epsilon=1e-8):
    """
    Compute absolute percentage error loss.
    
    Parameters:
    predicted: tensor of predicted values
    actual: tensor of actual values  
    epsilon: small value to avoid division by zero
    
    Returns:
    Mean absolute percentage error as a tensor
    """
    return torch.mean(torch.abs((actual - predicted) / (actual + epsilon)) * 100)

class AbsolutePercentageErrorLoss(nn.Module):
    """Custom loss function for absolute percentage error"""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predicted, actual):
        return torch.mean(torch.abs((actual - predicted) / (actual + self.epsilon)) * 100)
    
# === TOXICITY PREDICTION USING ENCODED FEATURES ===

# Create encoded datasets for toxicity prediction
print("Creating encoded datasets...")
model_morgan.eval()

# Get encoded features for training data
train_spectra = train_data.iloc[:, start_idx:stop_idx]
train_spectra_tensor = torch.Tensor(train_spectra.values).to(device)

with torch.no_grad():
    train_encoded_features = model_morgan(train_spectra_tensor)

# Get toxicity targets for training data
train_log_tox = torch.Tensor(train_data["log_response"].values).unsqueeze(1).to(device)
train_indices_tox = torch.Tensor(train_data['index'].to_numpy()).to(device)

# Get encoded features for validation data
val_spectra = val_data.iloc[:, start_idx:stop_idx]
val_spectra_tensor = torch.Tensor(val_spectra.values).to(device)

with torch.no_grad():
    val_encoded_features = model_morgan(val_spectra_tensor)

# Get toxicity targets for validation data
val_log_tox = torch.Tensor(val_data["log_response"].values).unsqueeze(1).to(device)
val_indices_tox = torch.Tensor(val_data['index'].to_numpy()).to(device)

print(f"Encoded train features shape: {train_encoded_features.shape}")
print(f"Encoded val features shape: {val_encoded_features.shape}")

# Create data loaders for toxicity prediction
train_tox_dataset = TensorDataset(train_encoded_features, train_log_tox, train_indices_tox)
val_tox_dataset = TensorDataset(val_encoded_features, val_log_tox, val_indices_tox)

train_tox_loader = DataLoader(train_tox_dataset, batch_size=batch_size, shuffle=True)
val_tox_loader = DataLoader(val_tox_dataset, batch_size=batch_size, shuffle=False)

# Set up toxicity MLP parameters
tox_input_size = output_size  # Same as Morgan encoder output size (2048)
tox_output_size = 1  # Predicting single toxicity value
tox_num_layers = 8
tox_learning_rate = 0.0001
tox_epochs = 200

# Create toxicity MLP
toxicity_mlp = fd.ToxMLP_Reg(
    input_size=tox_input_size, 
    output_size=tox_output_size, 
    num_layers=tox_num_layers
).to(device)

# Use absolute percentage error as criterion
criterion_tox = AbsolutePercentageErrorLoss()

# Create config for toxicity MLP
tox_config = {
    'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
    'wandb_project': 'MIT-Lincoln-Lab',
    'gpu': True,
    'embedding_type': "Morgan-Encoded Features",
    'encoder_type': "Toxicity MLP",
    # Model hyperparameters
    'batch_size': batch_size,
    'input_size': tox_input_size,
    'output_size': tox_output_size,
    'num_layers': tox_num_layers,
    'learning_rate': tox_learning_rate,
    'epochs': tox_epochs,
    'criterion': 'AbsolutePercentageError',
    'Bin': 0.01,
    'Threshold': 0.001,
    # Parent model info
    'parent_encoder': 'Morgan Encoder',
    'parent_epochs': epochs,
    'parent_output_size': output_size,
}

# Train the toxicity MLP
print("Training toxicity MLP...")
toxicity_model, tox_train_losses, tox_val_losses = fd.train_model_MLP(
    model=toxicity_mlp,
    train_data=train_tox_loader,
    val_data=val_tox_loader,
    epochs=tox_epochs,
    learning_rate=tox_learning_rate,
    criterion=criterion_tox,
    device=device,
    config=tox_config
)

print("Toxicity MLP training complete!")
print(f"Final toxicity prediction error: {tox_val_losses[-1]:.3f}% APE")

print("Pipeline completed successfully!")
print("="*50)
print("SUMMARY:")
print(f"1. Morgan Encoder - Final Val Loss: {val_loss_history[-1]:.6f}")
print(f"2. Toxicity MLP - Final Val APE: {tox_val_losses[-1]:.3f}%")
print("="*50)