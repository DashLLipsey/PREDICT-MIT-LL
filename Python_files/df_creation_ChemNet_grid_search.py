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

# Create folder for ChemNet datasets
chemnet_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/chemnet_grid_search_dataframes"

# ENCODER TRAINING LOOP - Process all datasets
device = fd.set_up_gpu()
# device = torch.device('cpu')
name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_chemnet.parquet")

# Get all dataset files from the grid search folder
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes"
dataset_files = [f for f in os.listdir(grid_search_folder) if f.endswith('.pkl') and 'df_spectra' in f]

# Filter out datasets with bin size 0.01 and 0.05
filtered_dataset_files = []
for f in dataset_files:
    # Check if the file contains bin0_01 or bin0_05 (which represents bin sizes 0.01 and 0.05)
    if 'bin0_01' not in f:
        filtered_dataset_files.append(f)
    else:
        if 'bin0_01' in f:
            print(f"Skipping bin size 0.01 dataset: {f}")

dataset_names = [f.replace('.pkl', '') for f in filtered_dataset_files]

print(f"Found {len(dataset_files)} total datasets")
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

# Storage for encoder results
encoder_results = []
chemnet_datasets = {}

# Training parameters
batch_size = 512
epochs=300
lr=0.0001
criterion=nn.MSELoss()
output_size = 512
num_layers = 4

# Loop through each dataset
for i, dataset_name in enumerate(sorted(dataset_names), 1):
    print(f"\nProcessing {i}/{len(dataset_names)}: {dataset_name}")
    
    try:
        # Parse dataset parameters
        bin_size, threshold = parse_dataset_name(dataset_name)
        
        # Create adaptive config for this dataset
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
            'learning_rate': lr,
            'epochs': epochs,
            # Dataset-specific parameters
            'Bin': bin_size,
            'Threshold': threshold,
            'Original_Features': None,  # Will be updated after loading data
        }
        
        # Load dataset from pickle file
        dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.pkl")
        current_dataset = pd.read_pickle(dataset_path)
        
        print(f"Loaded {dataset_name} - Shape: {current_dataset.shape}")
        print(f"Config - Bin: {bin_size}, Threshold: {threshold}")
        
        # Fix data types - be more selective about which columns to convert
        exclude_cols = ['SMILES_spectra', 'index_id', 'Group', 'Response', 'log_response']
        numeric_cols = [col for col in current_dataset.columns if col not in exclude_cols]

        for col in numeric_cols:
            try:
                if current_dataset[col].dtype == 'object':
                    current_dataset[col] = pd.to_numeric(current_dataset[col], errors='coerce')
                current_dataset[col] = current_dataset[col].fillna(0.0).astype(np.float32)
            except Exception as e:
                print(f"Warning: Could not convert column {col}: {str(e)}")
                continue

        # Apply train/test split
        counts = current_dataset['SMILES_spectra'].value_counts()

        # Apply train/test split
        counts = current_dataset['SMILES_spectra'].value_counts()
        valid_smiles = counts[counts >= 4].index
        filtered_dataset = current_dataset[current_dataset['SMILES_spectra'].isin(valid_smiles)].copy()
        
        print(f"After filtering (>=4 spectra per SMILES): {filtered_dataset.shape}")
        
        train_indices = []
        test_indices = []
        
        np.random.seed(42)
        for smiles, group in filtered_dataset.groupby('SMILES_spectra'):
            idx = group.index.tolist()
            n = len(idx)
            np.random.shuffle(idx)
            split = n // 2
            test_idx = idx[:split]
            train_idx = idx[split:]
            train_indices.extend(train_idx)
            test_indices.extend(test_idx)
        
        train_data_current = filtered_dataset.loc[train_indices].reset_index(drop=True)
        test_data_current = filtered_dataset.loc[test_indices].reset_index(drop=True)
        train_data_current['index'] = train_data_current.index
        test_data_current['index'] = test_data_current.index
        
        print(f"Train data shape: {train_data_current.shape}")
        print(f"Test data shape: {test_data_current.shape}")
        
        # Create tensors
        y_train_enc, x_train_enc, train_indices_tensor = fd.create_dataset_tensors(
            train_data_current, name_smiles_embedding_df, device, start_idx=1, stop_idx=-3)
        
        y_val_enc, x_val_enc, val_indices_tensor = fd.create_dataset_tensors(
            test_data_current, name_smiles_embedding_df, device, start_idx=1, stop_idx=-3)
        
        print(f"Training tensor shapes: x_train: {x_train_enc.shape}, y_train: {y_train_enc.shape}")
        
        # Update config with actual feature count
        chemnet_config['Original_Features'] = x_train_enc.shape[1]
        
        train_dataset = TensorDataset(x_train_enc, y_train_enc, train_indices_tensor)
        val_dataset = TensorDataset(x_val_enc, y_val_enc, val_indices_tensor)
        train_loader_enc = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader_enc = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create and train encoder
        encoder_current = fd.ChemNet_Encoder(input_size=x_train_enc.shape[1], output_size=output_size, num_layers=num_layers).to(device)
        
        # Train encoder with adaptive config
        trained_encoder = fd.train_model_chemnet_encoder(
            model=encoder_current,
            train_data=train_loader_enc,
            val_data=val_loader_enc,
            epochs=epochs,
            learning_rate=lr,
            criterion=criterion,
            device=device,
            config=chemnet_config  # Pass the adaptive config
        )
        
        # Generate embeddings
        encoder_current.eval()
        with torch.no_grad():
            train_embeddings = encoder_current(x_train_enc).cpu().numpy()
            test_embeddings = encoder_current(x_val_enc).cpu().numpy()
        
        print(f"Generated embeddings shapes: train: {train_embeddings.shape}, test: {test_embeddings.shape}")
        
        # Create ChemNet dataset with embeddings
        train_chemnet_df = pd.DataFrame(train_embeddings, columns=[f'emb_{j}' for j in range(output_size)])
        train_chemnet_df['SMILES_spectra'] = train_data_current['SMILES_spectra'].values
        train_chemnet_df['Response'] = train_data_current['Response'].values
        train_chemnet_df['log_response'] = train_data_current['log_response'].values
        train_chemnet_df['index_id'] = train_data_current['index_id'].values
        
        test_chemnet_df = pd.DataFrame(test_embeddings, columns=[f'emb_{j}' for j in range(output_size)])
        test_chemnet_df['SMILES_spectra'] = test_data_current['SMILES_spectra'].values
        test_chemnet_df['Response'] = test_data_current['Response'].values
        test_chemnet_df['log_response'] = test_data_current['log_response'].values
        test_chemnet_df['index_id'] = test_data_current['index_id'].values
        
        # Combine train and test
        full_chemnet_df = pd.concat([train_chemnet_df, test_chemnet_df], ignore_index=True)
        
        print(f"Final ChemNet dataset shape: {full_chemnet_df.shape}")
        
        # Save to chemnet folder
        chemnet_dataset_name = f"chemnet_emb_{dataset_name}"
        save_path = os.path.join(chemnet_folder, f"{chemnet_dataset_name}.pkl")
        full_chemnet_df.to_pickle(save_path)
        print(f"Saved to: {save_path}")
        
        # Store results with config
        encoder_results.append({
            'Original_Dataset': dataset_name,
            'ChemNet_Dataset': chemnet_dataset_name,
            'Train_Samples': len(train_data_current),
            'Test_Samples': len(test_data_current),
            'Total_Samples': len(full_chemnet_df),
            'Embedding_Dim': output_size,
            'Original_Features': x_train_enc.shape[1],
            'Bin_Size': bin_size,
            'Threshold': threshold,
            'Config': chemnet_config
        })
        
        # Store in memory as well
        chemnet_datasets[chemnet_dataset_name] = full_chemnet_df
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        continue

print(f"\n=== ENCODER TRAINING COMPLETED ===")
print(f"Successfully processed {len(encoder_results)} datasets")
print(f"Created ChemNet embedding datasets saved to: {chemnet_folder}")

# Display results summary
results_df = pd.DataFrame(encoder_results)
print("\nProcessing Summary:")
print(results_df)