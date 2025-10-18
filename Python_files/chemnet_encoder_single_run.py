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

# Create folder for super test sets if it doesn't exist
super_test_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/super_test_sets"
os.makedirs(super_test_folder, exist_ok=True)

# ENCODER TRAINING - Single Dataset
device = fd.set_up_gpu()
# device = torch.device('cpu')
name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_chemnet.parquet")

# ====== SUPER TEST SET SMILES (COMMENT OUT TO INCLUDE IN TRAINING) ======
super_test_smiles = [
    'NC(=S)Nc1ccccc1',                                  # 6
    'COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1',   # 12
    'CCNc1nc(Cl)nc(NC(C)(C)C#N)n1',                     # 6 
    'C#CCN(C)Cc1ccccc1',                                # 6
    'COP(=S)(OC)Oc1ccc(SC)c(C)c1',                      # 6
    'Nc1cccc2c(N)cccc12',                               # 6
    'Cn1c(=O)c2c(ncn2CCO)n(C)c1=O',                     # 40
    'CNC(=O)N(C)c1nnc(C(C)(C)C)s1',                     # 6
    'Nc1ccc(Sc2ccc(N)cc2)cc1',                          # 15
    'COc1ccc2ccc(=O)oc2c1CC=C(C)C',                     # 6
]

# ====== SPECIFY BIN SIZE AND THRESHOLD ======
bin_size = 1   # Also try 0.1, 1, 200
threshold = 100  # Also try 0.1, 50, 100

# Function to create dataset name from bin and threshold
def create_dataset_name(bin_size, threshold):
    """Create dataset name from bin size and threshold"""
    bin_str = str(bin_size).replace('.', '_')
    
    if threshold == 0:
        return f"bin{bin_str}_thresh_zero_df_spectra"
    else:
        thresh_str = str(threshold).replace('.', '_')
        return f"bin{bin_str}_thresh{thresh_str}_df_spectra"

# Create the dataset name
dataset_name = create_dataset_name(bin_size, threshold)
print(f"Processing single dataset: {dataset_name}")

# Get the grid search folder
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes"

# Training parameters
batch_size = 256
epochs = 500
lr = 0.0001  
criterion = nn.MSELoss()
output_size = 512
num_layers = 4

try:
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
        'Super_test': True,
    }
    
    # Load dataset from pickle file
    dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.pkl")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
    current_dataset = pd.read_pickle(dataset_path)
    
    print(f"Loaded {dataset_name} - Shape: {current_dataset.shape}")
    
    # ====== REMOVE SUPER TEST SET FROM TRAINING DATA ======
    # Save super test set before removing
    super_test_df = current_dataset[current_dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
    print(f"Super test set size: {len(super_test_df)} samples")
    print(f"Super test SMILES found: {super_test_df['SMILES_spectra'].nunique()} unique SMILES")
    
    # Remove super test set from training data
    original_count = len(current_dataset)
    current_dataset = current_dataset[~current_dataset['SMILES_spectra'].isin(super_test_smiles)]
    removed_count = original_count - len(current_dataset)
    print(f"Removed {removed_count} samples from super test set")
    print(f"Dataset shape after removal: {current_dataset.shape}")
    
    print(f"Config - Bin: {bin_size}, Threshold: {threshold}")

    # Fix data types - be more selective about which columns to convert
    exclude_cols = ['SMILES_spectra', 'index_id', 'Group', 'Response', 'log_response']
    numeric_cols = [col for col in current_dataset.columns if col not in exclude_cols]

    # Apply train/test split with improved filtering
    counts = current_dataset['SMILES_spectra'].value_counts()
    valid_smiles = counts[counts >= 4].index  # Increased from 4 to 8
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
    
    print(f"Starting training for {epochs} epochs with learning rate {lr}...")
    
    # Train encoder with adaptive config
    trained_encoder = fd.train_model_chemnet_encoder(
        model=encoder_current,
        train_data=train_loader_enc,
        val_data=val_loader_enc,
        epochs=epochs,
        learning_rate=lr,
        criterion=criterion,
        device=device,
        config=chemnet_config  
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
    
    # Save to chemnet folder (overwriting existing file)
    chemnet_dataset_name = f"chemnet_emb_{dataset_name}"
    save_path = os.path.join(chemnet_folder, f"{chemnet_dataset_name}.pkl")
    full_chemnet_df.to_pickle(save_path)
    print(f"Saved to: {save_path}")


    # ====== GENERATE SUPER TEST SET EMBEDDINGS ======
    if len(super_test_df) > 0:
        print(f"\n=== GENERATING SUPER TEST SET EMBEDDINGS ===")
        
        # Add index column to super_test_df (required by create_dataset_tensors)
        super_test_df_with_index = super_test_df.reset_index(drop=True).copy()
        super_test_df_with_index['index'] = super_test_df_with_index.index
        
        # Create super test tensors
        y_super_test, x_super_test, super_test_indices_tensor = fd.create_dataset_tensors(
            super_test_df_with_index, name_smiles_embedding_df, device, start_idx=1, stop_idx=-3)
        
        print(f"Super test tensor shapes: x_super_test: {x_super_test.shape}, y_super_test: {y_super_test.shape}")
        
        # Generate embeddings for super test set
        with torch.no_grad():
            super_test_embeddings = encoder_current(x_super_test).cpu().numpy()
        
        print(f"Generated super test embeddings shape: {super_test_embeddings.shape}")
        
        # Create super test ChemNet dataset with embeddings
        super_test_chemnet_df = pd.DataFrame(super_test_embeddings, columns=[f'emb_{j}' for j in range(output_size)])
        super_test_chemnet_df['SMILES_spectra'] = super_test_df['SMILES_spectra'].values
        super_test_chemnet_df['Response'] = super_test_df['Response'].values
        super_test_chemnet_df['log_response'] = super_test_df['log_response'].values
        super_test_chemnet_df['index_id'] = super_test_df['index_id'].values
        
        # Save super test set embeddings
        super_test_save_name = f"super_test_chemnet_emb_{dataset_name}"
        super_test_save_path = os.path.join(super_test_folder, f"{super_test_save_name}.pkl")
        super_test_chemnet_df.to_pickle(super_test_save_path)
        print(f"Saved super test set embeddings to: {super_test_save_path}")
        print(f"Super test set embeddings shape: {super_test_chemnet_df.shape}")
    else:
        print(f"\n=== NO SUPER TEST SET SMILES FOUND IN DATASET ===")
    
    # Display results summary
    print(f"\n=== ENCODER TRAINING COMPLETED ===")
    print(f"Dataset: {dataset_name}")
    print(f"ChemNet Dataset: {chemnet_dataset_name}")
    print(f"Train Samples: {len(train_data_current)}")
    print(f"Test Samples: {len(test_data_current)}")
    print(f"Total Samples: {len(full_chemnet_df)}")
    print(f"Super Test Samples: {len(super_test_df) if len(super_test_df) > 0 else 0}")
    print(f"Embedding Dimension: {output_size}")
    print(f"Original Features: {x_train_enc.shape[1]}")
    print(f"Bin Size: {bin_size}")
    print(f"Threshold: {threshold}")
    
    print(f"\nSuccessfully created and saved ChemNet embedding dataset!")
    if len(super_test_df) > 0:
        print(f"Successfully created and saved super test set embeddings!")
    
except Exception as e:
    print(f"Error processing {dataset_name}: {str(e)}")
    import traceback
    traceback.print_exc()