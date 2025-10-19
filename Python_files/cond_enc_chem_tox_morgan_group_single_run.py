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
import wandb

# Add the Python_files directory to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'Python_files'))

# Now you can import your modules
import functions_enc as f
import function_depot as fd

# ====== SPECIFY BIN SIZE AND THRESHOLD ======
bin_size = 100   # Change this to your desired bin size (e.g., 0.05, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000)
threshold = 1    # Change this to your desired threshold (e.g., 0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100)

# Function to create dataset name from bin and threshold
def create_dataset_name(bin_size, threshold):
    """Create dataset name from bin size and threshold"""
    bin_str = str(bin_size).replace('.', '_')
    
    if threshold == 0:
        return f"bin{bin_str}_thresh_zero_df_spectra"
    else:
        thresh_str = str(threshold).replace('.', '_')
        return f"bin{bin_str}_thresh{thresh_str}_df_spectra"

# Function to extract bin size and threshold from dataset name
def parse_dataset_name(dataset_name):
    """Extract bin size and threshold from dataset name"""
    if 'thresh_zero' in dataset_name:
        # Extract bin size
        bin_part = dataset_name.split('_thresh_zero')[0].replace('bin', '')
        bin_size = float(bin_part.replace('_', '.'))
        threshold = 0.0
    else:
        # Extract bin size and threshold
        parts = dataset_name.split('_thresh')
        bin_part = parts[0].replace('bin', '')
        bin_size = float(bin_part.replace('_', '.'))
        
        thresh_part = parts[1].split('_df_spectra')[0]
        threshold = float(thresh_part.replace('_', '.'))
    
    return bin_size, threshold

# Conditional Encoder Architecture: Set the parameters and the loss function
output_size = 2561  # ChemNet + Toxicity + Morgan
num_layers = 4
batch_size = 512
epochs = 250
lr = 0.0003
lambda1 = 1
lambda2 = 5
lambda3 = 1  # For Morgan fingerprints
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()
criterion3 = nn.MSELoss()

print("=== CONDITIONAL ENCODER (ChemNet + Toxicity + Morgan + Group) SINGLE DATASET TRAINING ===")

# Set up device and load reference datasets
device = fd.set_up_gpu()
name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_chemnet.parquet")
morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_morganfp.parquet")
df5_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_subset.parquet")
df5_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_spectra.parquet")

# Define folders
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes"
output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/cond_enc_full_outputs"
os.makedirs(output_folder, exist_ok=True)

# Create the dataset name
dataset_name = create_dataset_name(bin_size, threshold)
print(f"Processing single dataset: {dataset_name}")

# Create Group mapping once at the start
print("Creating Group mapping from df5_spectra...")
id_to_group = dict(zip(df5_spectra['index_id'], df5_spectra['Group']))
print(f"Group mapping created with {len(id_to_group)} entries")

try:
    # Load dataset from pickle file
    dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.pkl")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
    dataset = pd.read_pickle(dataset_path)

    # Convert to DataFrame if it's not already one
    if not isinstance(dataset, pd.DataFrame):
        dataset = pd.DataFrame(dataset)

    print(f"Loaded {dataset_name} - Shape: {dataset.shape}")
    
    # Add Group column if not present
    if 'Group' not in dataset.columns:
        dataset = dataset.copy()  # Defragment DataFrame
        dataset['Group'] = dataset['index_id'].map(id_to_group).fillna('Unknown')
        print(f"Added Group column. Unique groups: {dataset['Group'].nunique()}")
            
    # Apply filtering
    counts = dataset['SMILES_spectra'].value_counts()
    valid_smiles = counts[counts >= 4].index
    filtered_dataset = dataset[dataset['SMILES_spectra'].isin(valid_smiles)].copy()
    
    print(f"After filtering (>=4 spectra per SMILES): {filtered_dataset.shape}")
    
    # Vectorized train/test split
    smiles_groups = filtered_dataset.groupby('SMILES_spectra')
    train_indices = []
    test_indices = []
    
    np.random.seed(42)
    for smiles, group in smiles_groups:
        idx = group.index.values  # Use .values for faster access
        n = len(idx)
        np.random.shuffle(idx)
        split = n // 2
        test_indices.extend(idx[:split])
        train_indices.extend(idx[split:])
    
    train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
    test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)
    
    # Add index column
    train_data['index'] = range(len(train_data))
    test_data['index'] = range(len(test_data))
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Process both datasets together to avoid duplication
    train_data_processed = fd.add_response_and_log_response(train_data.copy(), df5_subset, smiles_col='SMILES_spectra')
    test_data_processed = fd.add_response_and_log_response(test_data.copy(), df5_subset, smiles_col='SMILES_spectra')
    
    # Create tensors
    print("Creating tensors...")
    x_train_with_group, y_train_emb, y_train_tox, y_train_morgan, train_indices_tensor = fd.create_dataset_tensors_condenc_full(
            train_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-5)

    x_val_with_group, y_val_emb, y_val_tox, y_val_morgan, val_indices_tensor = fd.create_dataset_tensors_condenc_full(
        test_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-5)

    # Get the actual input size and create model accordingly
    actual_input_size = x_train_with_group.shape[1]
    print(f"Creating model with input size: {actual_input_size} for {dataset_name}")

    # Create model with correct input size
    cond_encoder_current = fd.Cond_Encoder_full(input_size=actual_input_size,
                                                output_size=output_size, 
                                                num_layers=num_layers).to(device)

    # Create DataLoaders
    train_dataset = TensorDataset(x_train_with_group, y_train_emb, y_train_tox, y_train_morgan, train_indices_tensor)
    val_dataset = TensorDataset(x_val_with_group, y_val_emb, y_val_tox, y_val_morgan, val_indices_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
            
    # Parse dataset parameters for wandb config
    parsed_bin_size, parsed_threshold = parse_dataset_name(dataset_name)
    
    # Create wandb config for this dataset
    chemnet_tox_morgan_config = {
        'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
        'wandb_project': 'MIT-Lincoln-Lab',
        'wandb_name': f"cond_enc_full_{dataset_name}",
        'gpu': True,
        'encoder_type': "Conditional Encoder ChemNet + Toxicity + Morgan + Group",
        # Model hyperparameters
        'batch_size': batch_size,
        'output_size': output_size,
        'num_layers': num_layers,
        'learning_rate': lr,
        'epochs': epochs,
        'lambda1': lambda1,
        'lambda2': lambda2,
        'lambda3': lambda3,
        # Dataset-specific parameters
        'Bin': parsed_bin_size,
        'Threshold': parsed_threshold,
    }
    
    print(f"Starting training for {epochs} epochs with learning rate {lr}...")
    
    # Train conditional encoder using the group-conditioned training function
    trained_cond_encoder = fd.train_model_condenc_full(
        model=cond_encoder_current,
        train_data=train_loader,
        val_data=val_loader,
        epochs=epochs,
        learning_rate=lr,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        criterion1=criterion1,
        criterion2=criterion2,
        criterion3=criterion3,
        device=device,
        config=chemnet_tox_morgan_config
    )
    
    # Generate predictions from the 513th element (index 512) - toxicity regression output
    cond_encoder_current.eval()
    with torch.no_grad():
        # Train predictions - extract toxicity prediction from element 512
        train_predictions = cond_encoder_current(x_train_with_group).cpu().numpy()
        train_tox_predictions = train_predictions[:, 512]  # Element 512 (0-indexed for 513th element)
        
        # Test predictions - extract toxicity prediction from element 512
        test_predictions = cond_encoder_current(x_val_with_group).cpu().numpy()
        test_tox_predictions = test_predictions[:, 512]  # Element 512 (0-indexed for 513th element)
    
    # Get true toxicity values
    train_tox_true = y_train_tox.cpu().numpy().flatten()
    test_tox_true = y_val_tox.cpu().numpy().flatten()
    train_tox_pred = train_tox_predictions.flatten()
    test_tox_pred = test_tox_predictions.flatten()
    
    # Calculate percent errors (undo log transform to get back to original response scale)
    train_response_true = np.exp(train_tox_true)
    train_response_pred = np.exp(train_tox_pred)
    test_response_true = np.exp(test_tox_true)
    test_response_pred = np.exp(test_tox_pred)
    
    # Calculate absolute percent errors
    train_median_percent_error = 100 * np.median(np.abs(train_response_pred - train_response_true) / train_response_true)
    test_median_percent_error = 100 * np.median(np.abs(test_response_pred - test_response_true) / test_response_true)
    train_mean_percent_error = 100 * np.mean(np.abs(train_response_pred - train_response_true) / train_response_true)
    test_mean_percent_error = 100 * np.mean(np.abs(test_response_pred - test_response_true) / test_response_true)
    
    # Generate 2561-dimensional outputs for ALL data (train + test combined)
    full_data_processed = pd.concat([train_data_processed, test_data_processed], ignore_index=True)
    
    # Create tensors for full dataset
    x_full_with_group, y_full_emb, y_full_tox, y_full_morgan, full_indices_tensor = fd.create_dataset_tensors_condenc_full(
        full_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-5)
    
    # Generate conditional encoder outputs
    cond_encoder_current.eval()
    with torch.no_grad():
        full_cond_outputs = cond_encoder_current(x_full_with_group).cpu().numpy()
    
    print(f"Generated full conditional encoder outputs shape: {full_cond_outputs.shape}")
    
    # Create output DataFrame with 2561 dimensions + metadata
    emb_cols = [f'cond_emb_{j}' for j in range(512)]  # ChemNet embedding dimensions
    morgan_cols = [f'cond_morgan_{j}' for j in range(2048)]  # Morgan fingerprint dimensions
    
    output_df = pd.DataFrame(full_cond_outputs[:, :512], columns=emb_cols)
    output_df['cond_tox_pred'] = full_cond_outputs[:, 512]  # Toxicity prediction
    
    # Add Morgan fingerprint predictions
    morgan_pred_df = pd.DataFrame(full_cond_outputs[:, 513:], columns=morgan_cols)
    output_df = pd.concat([output_df, morgan_pred_df], axis=1)
    
    # Add metadata
    output_df['SMILES_spectra'] = full_data_processed['SMILES_spectra'].values
    output_df['Response'] = full_data_processed['Response'].values
    output_df['log_response'] = full_data_processed['log_response'].values
    output_df['index_id'] = full_data_processed['index'].values
    
    # Parse bin size and threshold from dataset name for filename
    if 'thresh_zero' in dataset_name:
        bin_part = dataset_name.split('_thresh_zero')[0]  # Keep bin format
        threshold_part = "thresh_zero"
    else:
        parts = dataset_name.split('_thresh')
        bin_part = parts[0]  # Keep bin format
        
        thresh_part = parts[1].split('_df_spectra')[0]
        threshold_part = f"thresh{thresh_part}"  # Keep thresh format
    
    # Save conditional encoder outputs (2561 dimensions + 4 metadata = 2565 columns total)
    predictions_filename = f"cond_enc_full_{bin_part}_{threshold_part}_df_spectra.pkl"
    predictions_path = os.path.join(output_folder, predictions_filename)
    output_df.to_pickle(predictions_path)

    print(f"\n=== CONDITIONAL ENCODER TRAINING COMPLETED ===")
    print(f"Dataset: {dataset_name}")
    print(f"Bin Size: {bin_size}")
    print(f"Threshold: {threshold}")
    print(f"Train Samples: {len(train_data_processed)}")
    print(f"Test Samples: {len(test_data_processed)}")
    print(f"Total Samples: {len(full_data_processed)}")
    print(f"Input Features: {actual_input_size}")
    print(f"Output Dimensions: {output_size}")
    print(f"Final Output Shape: {output_df.shape}")
    
    print(f"\nToxicity Prediction Performance (from 513th encoder output):")
    print(f"Train Median % Error: {train_median_percent_error:.1f}%")
    print(f"Train Mean % Error: {train_mean_percent_error:.1f}%")
    print(f"Test Median % Error: {test_median_percent_error:.1f}%")
    print(f"Test Mean % Error: {test_mean_percent_error:.1f}%")
    
    print(f"\nSaved prediction dataframe to: {predictions_filename}")
    print(f"Full path: {predictions_path}")
    print(f"\nSuccessfully created and saved conditional encoder outputs!")
    
except Exception as e:
    print(f"Error processing {dataset_name}: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # Clear GPU memory
    if 'x_train_with_group' in locals():
        del x_train_with_group, x_val_with_group, y_train_emb, y_train_tox, y_train_morgan, y_val_emb, y_val_tox, y_val_morgan
    if 'train_indices_tensor' in locals():
        del train_indices_tensor, val_indices_tensor
    if 'cond_encoder_current' in locals():
        del cond_encoder_current, trained_cond_encoder
    torch.cuda.empty_cache()