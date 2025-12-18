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

# ============================= HYPERPARAMETER TUNING GRID SEARCH ============================= #
import itertools
from sklearn.model_selection import ParameterGrid

# Define hyperparameter grids
hyperparameter_grid = {
    'lambda1': [1, 3, 5, 10],
    'lambda2': [5, 10, 15, 20, 30],
    'lambda3': [0.5, 1, 2, 5],
    'lambda4': [0.1, 0.5, 1, 2, 5],  # For filtered Morgan fingerprints
    'batch_size': [64, 128, 256, 512],
    'epochs': [100, 200, 300, 400],
    'learning_rate': [0.0001, 0.0005, 0.001],
    'num_layers': [4, 6, 8, 10]
}

# Evaluation weights for combined metric
delta1 = 0.7  # Weight for super test set performance
delta2 = 0.3  # Weight for regular test set performance

# Select single bin/threshold combination for hyperparameter tuning
SELECTED_DATASET = "bin1_thresh0_05_df_spectra"  # Choose your preferred combination
print(f"Running hyperparameter tuning on: {SELECTED_DATASET}")

# Generate all hyperparameter combinations
param_combinations = list(ParameterGrid(hyperparameter_grid))
print(f"Total hyperparameter combinations to test: {len(param_combinations)}")

# Storage for hyperparameter tuning results
hyperparam_results = []
best_score = float('inf')
best_params = None
best_model_state = None

# Load and prepare the selected dataset once
print(f"Loading and preparing {SELECTED_DATASET}...")
dataset_path = os.path.join(grid_search_folder, f"{SELECTED_DATASET}.parquet")
dataset = pd.read_parquet(dataset_path)

# Prepare dataset (same preprocessing as in your original loop)
original_count = len(dataset)
dataset_no_super_test = dataset[~dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
removed_count = original_count - len(dataset_no_super_test)

# Add Group and CE_clean columns
if 'Group' not in dataset_no_super_test.columns:
    dataset_no_super_test['Group'] = dataset_no_super_test['index_id'].map(id_to_group).fillna('Unknown')
if 'CE_clean' not in dataset_no_super_test.columns:
    dataset_no_super_test['CE_clean'] = dataset_no_super_test['index_id'].map(id_to_ce_clean).fillna('Unknown')

# Apply filtering (>=4 spectra per SMILES)
counts = dataset_no_super_test['SMILES_spectra'].value_counts()
valid_smiles = counts[counts >= 4].index
filtered_dataset = dataset_no_super_test[dataset_no_super_test['SMILES_spectra'].isin(valid_smiles)].copy()

# Create train/test split
smiles_groups = filtered_dataset.groupby('SMILES_spectra')
train_indices = []
test_indices = []

np.random.seed(42)  # Fixed seed for reproducible splits
for smiles, group in smiles_groups:
    idx = group.index.values
    n = len(idx)
    np.random.shuffle(idx)
    split = n // 2
    test_indices.extend(idx[:split])
    train_indices.extend(idx[split:])

train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)

# Add index columns
train_data['index'] = range(len(train_data))
test_data['index'] = range(len(test_data))

# Process data
train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')

# Prepare super test set
super_test_df = dataset[dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
if len(super_test_df) > 0:
    if 'Group' not in super_test_df.columns:
        super_test_df['Group'] = super_test_df['index_id'].map(id_to_group).fillna('Unknown')
    if 'CE_clean' not in super_test_df.columns:
        super_test_df['CE_clean'] = super_test_df['index_id'].map(id_to_ce_clean).fillna('Unknown')
    
    super_test_df['index'] = range(len(super_test_df))
    super_test_processed = fd.add_response_and_log_response(super_test_df.copy(), df6_subset, smiles_col='SMILES_spectra')

print(f"Dataset prepared: Train={len(train_data_processed)}, Test={len(test_data_processed)}, Super Test={len(super_test_df) if len(super_test_df) > 0 else 0}")

# Load additional data once
name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_chemnet.parquet")
morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_morganfp.parquet")

# Load filtered Morgan fingerprint data for 1234e1e2
filtered_morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_filtered_morganfp.parquet")

# Create tensors once (they don't change across hyperparameter combinations)
print("Creating tensors...")
x_train_with_ext, y_train_emb, y_train_tox, y_train_morgan, y_train_filtered_morgan, train_indices_tensor = fd.create_dataset_tensors_condenc_1234e1e2(
    train_data_processed, name_smiles_embedding_df, morgan_df, filtered_morgan_df, device, start_idx=1, stop_idx=-6)

x_val_with_ext, y_val_emb, y_val_tox, y_val_morgan, y_val_filtered_morgan, val_indices_tensor = fd.create_dataset_tensors_condenc_1234e1e2(
    test_data_processed, name_smiles_embedding_df, morgan_df, filtered_morgan_df, device, start_idx=1, stop_idx=-6)

if len(super_test_df) > 0:
    x_super_test_with_ext, y_super_test_emb, y_super_test_tox, y_super_test_morgan, y_super_test_filtered_morgan, super_test_indices_tensor = fd.create_dataset_tensors_condenc_1234e1e2(
        super_test_processed, name_smiles_embedding_df, morgan_df, filtered_morgan_df, device, start_idx=1, stop_idx=-6)

actual_input_size = x_train_with_ext.shape[1]
output_size = 2561  # ChemNet (512) + Toxicity (1) + Morgan (2048) for 1234e1e2

print(f"Starting hyperparameter grid search...")

# HYPERPARAMETER GRID SEARCH LOOP
for i, params in enumerate(param_combinations, 1):
    print(f"\n=== Hyperparameter Combination {i}/{len(param_combinations)} ===")
    print(f"Parameters: {params}")
    
    try:
        # Extract hyperparameters
        lambda1 = params['lambda1']
        lambda2 = params['lambda2']
        lambda3 = params['lambda3']
        lambda4 = params['lambda4']
        batch_size = params['batch_size']
        epochs = params['epochs']
        lr = params['learning_rate']
        num_layers = params['num_layers']
        
        # Create model with current hyperparameters
        model = fd.Cond_Encoder_1234(input_size=actual_input_size, 
                                     output_size=output_size, 
                                     num_layers=num_layers).to(device)
        
        # Create DataLoaders
        train_dataset = TensorDataset(x_train_with_ext, y_train_emb, y_train_tox, y_train_morgan, y_train_filtered_morgan, train_indices_tensor)
        val_dataset = TensorDataset(x_val_with_ext, y_val_emb, y_val_tox, y_val_morgan, y_val_filtered_morgan, val_indices_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
        
        # Create wandb config for tracking
        hyperparam_config = {
            'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
            'wandb_project': 'MIT-Lincoln-Lab',
            'wandb_name': f"hyperparam_tune_{SELECTED_DATASET}_{i}",
            'gpu': True,
            'encoder_type': "Conditional Encoder 1234e1e2 Hyperparameter Tuning",
            'dataset': SELECTED_DATASET,
            'hyperparam_combination': i,
            **params  # Include all hyperparameters
        }
        
        # Define loss functions
        criterion1 = nn.MSELoss()  # ChemNet embeddings
        criterion2 = nn.MSELoss()  # Toxicity
        criterion3 = nn.MSELoss()  # Morgan fingerprints
        criterion4 = nn.MSELoss()  # Filtered Morgan fingerprints
        
        # Train model
        print(f"Training model with {epochs} epochs...")
        trained_model = fd.train_model_condenc_1234e1e2(
            model=model,
            train_data=train_loader,
            val_data=val_loader,
            epochs=epochs,
            learning_rate=lr,
            criterion1=criterion1,
            criterion2=criterion2,
            criterion3=criterion3,
            criterion4=criterion4,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            lambda4=lambda4,
            device=device,
            config=hyperparam_config
        )
        
        # Evaluate model performance
        model.eval()
        with torch.no_grad():
            # Regular test set predictions
            test_predictions = model(x_val_with_ext).cpu().numpy()
            test_tox_predictions = test_predictions[:, 512]  # Toxicity prediction
            
            test_tox_true = y_val_tox.cpu().numpy().flatten()
            test_tox_pred = test_tox_predictions.flatten()
            
            # Calculate test set percent error
            test_response_true = np.exp(test_tox_true)
            test_response_pred = np.exp(test_tox_pred)
            test_median_percent_error = 100 * np.median(np.abs(test_response_pred - test_response_true) / test_response_true)
            
            # Super test set predictions (if available)
            if len(super_test_df) > 0:
                super_test_predictions = model(x_super_test_with_ext).cpu().numpy()
                super_test_tox_predictions = super_test_predictions[:, 512]
                
                super_test_tox_true = y_super_test_tox.cpu().numpy().flatten()
                super_test_tox_pred = super_test_tox_predictions.flatten()
                
                super_test_response_true = np.exp(super_test_tox_true)
                super_test_response_pred = np.exp(super_test_tox_pred)
                super_test_median_percent_error = 100 * np.median(np.abs(super_test_response_pred - super_test_response_true) / super_test_response_true)
            else:
                super_test_median_percent_error = float('inf')  # Penalty if no super test data
        
        # Calculate weighted combined score (lower is better)
        if len(super_test_df) > 0:
            combined_score = delta1 * super_test_median_percent_error + delta2 * test_median_percent_error
        else:
            combined_score = test_median_percent_error  # Fall back to test error only
        
        print(f"Test Median % Error: {test_median_percent_error:.2f}%")
        if len(super_test_df) > 0:
            print(f"Super Test Median % Error: {super_test_median_percent_error:.2f}%")
        print(f"Combined Score: {combined_score:.2f}")
        
        # Check if this is the best model so far
        if combined_score < best_score:
            best_score = combined_score
            best_params = params.copy()
            best_model_state = model.state_dict().copy()
            print(f"*** NEW BEST MODEL *** (Score: {best_score:.2f})")
        
        # Store results
        result_entry = {
            'combination': i,
            'test_median_percent_error': test_median_percent_error,
            'super_test_median_percent_error': super_test_median_percent_error if len(super_test_df) > 0 else None,
            'combined_score': combined_score,
            **params
        }
        hyperparam_results.append(result_entry)
        
        # Clean up GPU memory
        del model, trained_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error in combination {i}: {str(e)}")
        torch.cuda.empty_cache()
        continue

print(f"\n=== HYPERPARAMETER TUNING COMPLETED ===")
print(f"Tested {len(hyperparam_results)} combinations")
print(f"Best Score: {best_score:.2f}")
print(f"Best Parameters: {best_params}")

# Convert results to DataFrame and analyze
hyperparam_df = pd.DataFrame(hyperparam_results)

# Save results
hyperparam_results_path = f"/home/dlipsey/MITLincolnLabs/MIT_LL_data/hyperparameter_tuning_results_{SELECTED_DATASET}.parquet"
hyperparam_df.to_parquet(hyperparam_results_path, index=False)
print(f"Results saved to: {hyperparam_results_path}")

# Save best model
if best_model_state is not None:
    best_model_path = f"/home/dlipsey/MITLincolnLabs/MIT_LL_data/best_model_{SELECTED_DATASET}.pt"
    torch.save(best_model_state, best_model_path)
    print(f"Best model saved to: {best_model_path}")

# Analysis of hyperparameter importance
print("\n=== HYPERPARAMETER ANALYSIS ===")
print("Top 10 Best Combinations:")
top_10 = hyperparam_df.nsmallest(10, 'combined_score')
for idx, row in top_10.iterrows():
    print(f"Rank {row.name + 1}: Score={row['combined_score']:.2f}, "
          f"λ1={row['lambda1']}, λ2={row['lambda2']}, λ3={row['lambda3']}, λ4={row['lambda4']}, "
          f"LR={row['learning_rate']}, Epochs={row['epochs']}, Batch={row['batch_size']}, Layers={row['num_layers']}")

# Correlation analysis
correlation_cols = ['lambda1', 'lambda2', 'lambda3', 'lambda4', 'learning_rate', 'epochs', 'batch_size', 'num_layers']
correlations = hyperparam_df[correlation_cols + ['combined_score']].corr()['combined_score'].sort_values()
print("\nHyperparameter Correlations with Combined Score:")
for param, corr in correlations.items():
    if param != 'combined_score':
        print(f"{param}: {corr:.3f}")

print(f"\nEvaluation weights used: δ1={delta1} (super test), δ2={delta2} (regular test)")