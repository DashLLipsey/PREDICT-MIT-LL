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

##### ==================== GLOBAL REWRITE OF TRAINING PROCESS ==================== #####
# The big idea of this rewrite is to change the metric we evaulate by. Thus far we have stuck to MSE as our loss function and this
# swaps to mean OR median percent error as a written class and loss function to train with consistency to the metric of import to us.
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

# Storage for conditional encoder results
cond_encoder_results = [] 

# Conditional Encoder Architecture: Set the parameters and the loss function from the classes defined above.
output_size = 513
num_layers = 4
batch_size = 512
epochs=250
lr = 0.0003
lambda1 = 1
lambda2 = 5
# criterion=nn.MSELoss() # Still use MSELoss for the embedding criterion
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()

# Encoder architecture (With Validation Set)


# CONDITIONAL ENCODER TRAINING LOOP - Process all grid search datasets
print("=== CONDITIONAL ENCODER GRID SEARCH TRAINING ===")
print(f"Using toxicity criterion: {criterion2}")

# Set up device and load ChemNet reference
device = f.set_up_gpu()
name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_chemnet.parquet")

# Load the original dataset for response mapping
df5_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_subset.parquet")

# Define folders
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes"

# Get all dataset files from the grid search folder
dataset_files = [f for f in os.listdir(grid_search_folder) if f.endswith('.pkl') and 'df_spectra' in f]

# Define allowed bin sizes (exclude 0.01)
allowed_bin_prefixes = ['bin0_05_', 'bin0_1_', 'bin0_5_', 'bin1_', 'bin2_', 'bin5_', 'bin10_', 
                        'bin25_', 'bin50_', 'bin100_', 'bin200_', 'bin500_', 'bin1000_']

# Filter dataset files to only include allowed bin sizes
dataset_files = [f for f in dataset_files if any(f.startswith(prefix) for prefix in allowed_bin_prefixes)]

dataset_names = [f.replace('.pkl', '') for f in dataset_files]

print(f"Found {len(dataset_names)} datasets to process (excluding bin size 0.01)")

# Storage for conditional encoder results
cond_encoder_results = []

# Loop through each dataset and evaluate toxicity predictions from element 512 (0-indexed)
for i, dataset_name in enumerate(sorted(dataset_names), 1):
    print(f"\nProcessing {i}/{len(dataset_names)}: {dataset_name}")
    
    try:
        # Load dataset from pickle file
        dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.pkl")
        dataset = pd.read_pickle(dataset_path)

        # Convert to DataFrame if it's not already one
        if not isinstance(dataset, pd.DataFrame):
            print(f"Converting {type(dataset)} to DataFrame")
            dataset = pd.DataFrame(dataset)

        print(f"Loaded {dataset_name} - Shape: {dataset.shape}")
                
        # Apply same filtering and splitting logic
        counts = dataset['SMILES_spectra'].value_counts()
        valid_smiles = counts[counts >= 4].index
        filtered_dataset = dataset[dataset['SMILES_spectra'].isin(valid_smiles)].copy()
        
        print(f"After filtering (>=4 spectra per SMILES): {filtered_dataset.shape}")
        
        if len(filtered_dataset) < 20:
            print(f"Skipping {dataset_name}: Only {len(filtered_dataset)} samples after filtering")
            continue
        
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
        
        train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
        test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)
        
        # Add index column
        train_data['index'] = train_data.index
        test_data['index'] = test_data.index
        
        # Add response and log response
        train_data_copy = train_data.copy()
        test_data_copy = test_data.copy()
        
        train_data_processed = fd.add_response_and_log_response(train_data_copy, df5_subset, smiles_col='SMILES_spectra')
        test_data_processed = fd.add_response_and_log_response(test_data_copy, df5_subset, smiles_col='SMILES_spectra')
        
        # Create tensors
        y_train_emb, y_train_tox, x_train, train_indices_tensor = fd.create_dataset_tensors_12(
            train_data_processed, name_smiles_embedding_df, device, start_idx=1, stop_idx=-4)

        y_val_emb, y_val_tox, x_val, val_indices_tensor = fd.create_dataset_tensors_12(
            test_data_processed, name_smiles_embedding_df, device, start_idx=1, stop_idx=-4)
        
        # Create data loaders
        train_dataset = TensorDataset(x_train, y_train_emb, y_train_tox, train_indices_tensor)
        val_dataset = TensorDataset(x_val, y_val_emb, y_val_tox, val_indices_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create and train conditional encoder
        cond_encoder_current = fd.Cond_Encoder_12(input_size=x_train.shape[1], output_size=output_size, num_layers=num_layers).to(device)
        
        # Parse dataset parameters for wandb config
        bin_size, threshold = parse_dataset_name(dataset_name)
        
        # Create wandb config for this dataset
        chemnet_tox_config = {
            'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
            'wandb_project': 'MIT-Lincoln-Lab',
            'wandb_name': f"cond_enc_{dataset_name}",
            'gpu': True,
            'encoder_type': "Conditional Encoder ChemNet + Toxicity",
            # Model hyperparameters
            'batch_size': batch_size,
            'output_size': output_size,
            'num_layers': num_layers,
            'learning_rate': lr,
            'epochs': epochs,
            'lambda1': lambda1,
            'lambda2': lambda2,
            # Dataset-specific parameters
            'Bin': bin_size,
            'Threshold': threshold,
        }
        
        # Train conditional encoder (wandb init/log/finish handled inside this function)
        trained_cond_encoder = fd.train_model_condenc_12(
            model=cond_encoder_current,
            train_data=train_loader,
            val_data=val_loader,
            epochs=epochs,
            learning_rate=lr,
            lambda1=1,
            lambda2=5,
            criterion1=criterion1,
            criterion2=criterion2,
            device=device,
            config=chemnet_tox_config  # Pass the config to your training function
        )
        
        # Generate predictions from the 513th element (index 512) - toxicity regression output
        cond_encoder_current.eval()
        with torch.no_grad():
            # Train predictions - extract toxicity prediction from element 512
            train_predictions = cond_encoder_current(x_train).cpu().numpy()
            train_tox_predictions = train_predictions[:, 512]  # Element 512 (0-indexed for 513th element)
            
            # Test predictions - extract toxicity prediction from element 512
            test_predictions = cond_encoder_current(x_val).cpu().numpy()
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
        
        # Add conditional encoder predictions to the dataframes
        # Generate 513-dimensional outputs for ALL data (train + test combined)
        full_data_processed = pd.concat([train_data_processed, test_data_processed], ignore_index=True)
        
        # Create tensors for full dataset
        y_full_emb, y_full_tox, x_full, full_indices_tensor = fd.create_dataset_tensors_emb_tox(
            full_data_processed, name_smiles_embedding_df, device, start_idx=1, stop_idx=-4)
        
        # Generate conditional encoder outputs
        cond_encoder_current.eval()
        with torch.no_grad():
            full_cond_outputs = cond_encoder_current(x_full).cpu().numpy()
        
        # Create output DataFrame with 513 dimensions + metadata
        emb_cols = [f'cond_emb_{j}' for j in range(512)]  # ChemNet embedding dimensions
        
        output_df = pd.DataFrame(full_cond_outputs[:, :512], columns=emb_cols)
        output_df['cond_tox_pred'] = full_cond_outputs[:, 512]  # Toxicity prediction
        
        # Add metadata
        output_df['SMILES_spectra'] = full_data_processed['SMILES_spectra'].values
        output_df['Response'] = full_data_processed['Response'].values
        output_df['log_response'] = full_data_processed['log_response'].values
        output_df['index_id'] = full_data_processed['index'].values
        
        # Store results for heatmap analysis (only percent errors)
        cond_encoder_results.append({
            'Dataset': dataset_name,
            'Train_Median_Percent_Error': train_median_percent_error,
            'Test_Median_Percent_Error': test_median_percent_error,
            'Train_Mean_Percent_Error': train_mean_percent_error,
            'Test_Mean_Percent_Error': test_mean_percent_error,
            'Samples': len(filtered_dataset),
            'Train_Samples': len(train_data_processed),
            'Test_Samples': len(test_data_processed)
        })
        
        # Parse bin size and threshold from dataset name for filename
        if 'thresh_zero' in dataset_name:
            bin_part = dataset_name.split('_thresh_zero')[0]  # Keep bin0_1 format
            threshold_part = "thresh_zero"
        else:
            parts = dataset_name.split('_thresh')
            bin_part = parts[0]  # Keep bin0_1 format
            
            thresh_part = parts[1].split('_df_spectra')[0]
            threshold_part = f"thresh{thresh_part}"  # Keep thresh0_001 format
        
        # Save conditional encoder outputs (513 dimensions + 4 metadata = 517 columns total)
        output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/cond_enc_outputs"
        os.makedirs(output_folder, exist_ok=True)

        predictions_filename = f"cond_enc_{bin_part}_{threshold_part}_df_spectra.pkl"
        predictions_path = os.path.join(output_folder, predictions_filename)
        output_df.to_pickle(predictions_path)

        print(f"Toxicity Prediction Performance (from 513th encoder output):")
        print(f"Test Median % Error: {test_median_percent_error:.1f}%")
        print(f"Test Mean % Error: {test_mean_percent_error:.1f}%")
        print(f"Saved prediction dataframe to {predictions_filename}")

        # Clear GPU memory after each dataset
        del x_train, x_val, y_train_emb, y_train_tox, y_val_emb, y_val_tox
        del train_indices_tensor, val_indices_tensor
        del cond_encoder_current, trained_cond_encoder
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        # Clear memory on error too
        torch.cuda.empty_cache()
        continue

print(f"\n=== CONDITIONAL ENCODER EVALUATION COMPLETED ===")
print(f"Successfully processed {len(cond_encoder_results)} datasets")

# Convert results to DataFrame for heatmap analysis
df_cond_percent_error_results = pd.DataFrame(cond_encoder_results)

print("\nConditional Encoder Results Summary:")
print(f"Mean Test Median % Error: {df_cond_percent_error_results['Test_Median_Percent_Error'].mean():.2f}%")
print(f"Mean Test Mean % Error: {df_cond_percent_error_results['Test_Mean_Percent_Error'].mean():.2f}%")
