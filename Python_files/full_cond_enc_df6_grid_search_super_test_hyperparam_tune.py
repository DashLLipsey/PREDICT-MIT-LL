# Basic Package Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import json

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

##### ==================== HYPERPARAMETER GRID DEFINITION ==================== #####
# Define hyperparameters to search over
hyperparameter_grid = {
    'num_layers': [6, 8],
    'batch_size': [64, 128, 256],
    'learning_rate': [0.0001],
    'lambda1': [1, 3, 5],  # ChemNet embedding weight
    'lambda2': [1, 10, 15],  # Toxicity weight  
    'lambda3': [1],  # Morgan fingerprint weight
    'epochs': [300, 500, 1000]
}

print("=== HYPERPARAMETER TUNING WITH SUPER TEST SET EVALUATION ===")
print("Hyperparameter Grid:")
for param, values in hyperparameter_grid.items():
    print(f"  {param}: {values}")

# Calculate total combinations
total_combinations = 1
for values in hyperparameter_grid.values():
    total_combinations *= len(values)
print(f"Total hyperparameter combinations to test: {total_combinations}")

##### ==================== SUPER TEST SET SMILES ==================== #####
# Define super test set SMILES to remove from training
super_test_smiles = [
    'COC(=O)C=C(C)OP(=O)(OC)OC',
    'COc1cc2c(c3oc(=O)c4c(c13)CCC4=O)[C@@H]1C=CO[C@@H]1O2',
    'CC(=O)OC1(C)CC(C)C(=O)C(C(O)CC2CC(=O)NC(=O)C2)C1',
    'C[C@H]1O[C@@H](O[C@H]2[C@@H](O)C[C@H](O[C@H]3[C@@H](O)C[C@H](O[C@H]4CC[C@@]5(C)[C@H](CC[C@@H]6[C@@H]5C[C@@H](O)[C@]5(C)[C@@H](C7=CC(=O)OC7)CC[C@]65O)C4)O[C@@H]3C)O[C@@H]2C)C[C@H](O)[C@@H]1O',
    'CNC(=O)Oc1cc(C)cc(C(C)C)c1',
    'CNC(=O)Oc1ccc(N(C)C)c(C)c1',
    'C[C@@H]1Cc2c(Cl)cc(C(=O)N[C@@H](Cc3ccccc3)C(=O)O)c(O)c2C(=O)O1',
    'COc1ccc2c(c1)c(CC(=O)OCC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1',
    'CC(C)(C)CC(C)(C)c1ccc(OCCOCC[N+](C)(C)Cc2ccccc2)cc1',
    'CC(=O)N1CCN(c2ccc(OC[C@H]3CO[C@](Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1',
    'c1ccc(C2CN3CCSC3=N2)cc1',
    'CN(C)CCC=C1c2ccccc2CCc2ccccc21',
    'CCOP(=S)(OCC)Oc1ccc2c(C)c(Cl)c(=O)oc2c1',
    'CC(C)NCC(O)COc1cccc2ccccc12',
    'CCOC(=O)C(C)(C)Oc1ccc(Cl)cc1',
    'CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC',
    'COc1ccc2c(c1OC)C(=O)OC2C1c2cc3c(cc2CCN1C)OCO3',
    'CN(C)c1ccc(SC#N)cc1',
    'CC(C)[C@H](N)C(=O)O',
    'CCCCOC(=O)COC(=O)c1ccccc1C(=O)OCCCC',
    'NC(C(=O)O)c1ccccc1'
]

##### ==================== UTILITY FUNCTIONS ==================== #####
def parse_dataset_name(dataset_name):
    """Extract bin size and threshold from dataset name"""
    if 'thresh_zero' in dataset_name:
        bin_part = dataset_name.split('_thresh_zero')[0].replace('bin', '')
        bin_size = float(bin_part.replace('_', '.'))
        threshold = 0.0
    else:
        parts = dataset_name.split('_thresh')
        bin_part = parts[0].replace('bin', '')
        bin_size = float(bin_part.replace('_', '.'))
        
        thresh_part = parts[1].split('_df_spectra')[0]
        threshold = float(thresh_part.replace('_', '.'))
    
    return bin_size, threshold

def train_and_evaluate_hyperparameters(hyperparams, dataset_name, dataset, device,
                                     name_smiles_embedding_df, morgan_df, df6_subset, df6_spectra,
                                     id_to_group, id_to_ce_clean):
    """Train model with given hyperparameters and return super test set median percent error"""
    
    try:
        print(f"\nTesting hyperparameters: {hyperparams}")
        
        # Extract hyperparameters
        num_layers = hyperparams['num_layers']
        batch_size = hyperparams['batch_size']
        lr = hyperparams['learning_rate']
        lambda1 = hyperparams['lambda1']
        lambda2 = hyperparams['lambda2']
        lambda3 = hyperparams['lambda3']
        epochs = hyperparams['epochs']
        
        # Fixed parameters
        output_size = 2561
        criterion1 = nn.MSELoss()
        criterion2 = nn.MSELoss()
        criterion3 = nn.MSELoss()
        
        # Remove super test SMILES from training data
        dataset_no_super_test = dataset[~dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
        
        # Add Group and CE_clean columns if not present
        if 'Group' not in dataset_no_super_test.columns:
            dataset_no_super_test['Group'] = dataset_no_super_test['index_id'].map(id_to_group).fillna('Unknown')
        
        if 'CE_clean' not in dataset_no_super_test.columns:
            dataset_no_super_test['CE_clean'] = dataset_no_super_test['index_id'].map(id_to_ce_clean).fillna('Unknown')
        
        # Apply filtering (>=4 spectra per SMILES)
        counts = dataset_no_super_test['SMILES_spectra'].value_counts()
        valid_smiles = counts[counts >= 4].index
        filtered_dataset = dataset_no_super_test[dataset_no_super_test['SMILES_spectra'].isin(valid_smiles)].copy()
        
        # Train/test split
        smiles_groups = filtered_dataset.groupby('SMILES_spectra')
        train_indices = []
        test_indices = []
        
        np.random.seed(42)
        for smiles, group in smiles_groups:
            idx = group.index.values
            n = len(idx)
            np.random.shuffle(idx)
            split = n // 2
            test_indices.extend(idx[:split])
            train_indices.extend(idx[split:])
        
        train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
        test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)
        
        train_data['index'] = range(len(train_data))
        test_data['index'] = range(len(test_data))
        
        train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
        test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')
        
        # Handle missing groups/CE_clean values for consistency
        all_possible_groups = set(df6_spectra['Group'].unique())
        all_possible_ce_clean = set(df6_spectra['CE_clean'].unique())
        
        training_groups = set(filtered_dataset['Group'].unique())
        training_ce_clean = set(filtered_dataset['CE_clean'].unique())
        
        missing_groups_in_training = all_possible_groups - training_groups
        missing_ce_clean_in_training = all_possible_ce_clean - training_ce_clean
        
        if missing_groups_in_training or missing_ce_clean_in_training:
            # Add dummy samples for missing groups/CE_clean values
            for missing_group in missing_groups_in_training:
                dummy_row = filtered_dataset.iloc[0:1].copy()
                dummy_row['Group'] = missing_group
                dummy_row['index_id'] = -999
                filtered_dataset = pd.concat([filtered_dataset, dummy_row], ignore_index=True)
            
            for missing_ce in missing_ce_clean_in_training:
                dummy_row = filtered_dataset.iloc[0:1].copy()
                dummy_row['CE_clean'] = missing_ce
                dummy_row['index_id'] = -998
                filtered_dataset = pd.concat([filtered_dataset, dummy_row], ignore_index=True)
            
            # Recreate train/test split
            smiles_groups = filtered_dataset.groupby('SMILES_spectra')
            train_indices = []
            test_indices = []
            
            np.random.seed(42)
            for smiles, group in smiles_groups:
                idx = group.index.values
                n = len(idx)
                np.random.shuffle(idx)
                split = n // 2
                test_indices.extend(idx[:split])
                train_indices.extend(idx[split:])
            
            train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
            test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)
            
            train_data['index'] = range(len(train_data))
            test_data['index'] = range(len(test_data))
            
            train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
            test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')
        
        # Create tensors
        x_train_with_ext, y_train_emb, y_train_tox, y_train_morgan, train_indices_tensor = fd.create_dataset_tensors_condenc_full2(
                train_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-6)

        x_val_with_ext, y_val_emb, y_val_tox, y_val_morgan, val_indices_tensor = fd.create_dataset_tensors_condenc_full2(
            test_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-6)
        
        # Create model
        actual_input_size = x_train_with_ext.shape[1]
        cond_encoder_current = fd.Cond_Encoder_full(input_size=actual_input_size,
                                                   output_size=output_size, 
                                                   num_layers=num_layers).to(device)
        
        # Create DataLoaders
        train_dataset = TensorDataset(x_train_with_ext, y_train_emb, y_train_tox, y_train_morgan, train_indices_tensor)
        val_dataset = TensorDataset(x_val_with_ext, y_val_emb, y_val_tox, y_val_morgan, val_indices_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
        
        # Parse dataset parameters for wandb config
        bin_size, threshold = parse_dataset_name(dataset_name)
        
        # Create wandb config
        config = {
            'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
            'wandb_project': 'MIT-Lincoln-Lab',
            'wandb_name': f"hyperparam_tuning_{dataset_name}_{num_layers}L_{batch_size}B_{lr}LR_{lambda1}_{lambda2}_{lambda3}",
            'gpu': True,
            'encoder_type': "Hyperparameter Tuning - Conditional Encoder (Super Test)",
            # Hyperparameters being tuned
            'batch_size': batch_size,
            'output_size': output_size,
            'num_layers': num_layers,
            'learning_rate': lr,
            'epochs': epochs,
            'lambda1': lambda1,
            'lambda2': lambda2,
            'lambda3': lambda3,
            # Dataset parameters
            'Bin': bin_size,
            'Threshold': threshold,
            'super_test_removed': True,
            'super_test_smiles_count': len(super_test_smiles)
        }
        
        # Train model
        trained_cond_encoder = fd.train_model_condenc_full2(
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
            config=config
        )
        
        # Evaluate on super test set
        super_test_df = dataset[dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
        
        if len(super_test_df) > 0:
            # Add Group and CE_clean columns to super test set
            if 'Group' not in super_test_df.columns:
                super_test_df['Group'] = super_test_df['index_id'].map(id_to_group).fillna('Unknown')
            
            if 'CE_clean' not in super_test_df.columns:
                super_test_df['CE_clean'] = super_test_df['index_id'].map(id_to_ce_clean).fillna('Unknown')
            
            # Handle missing groups/CE_clean in super test set
            super_test_groups = set(super_test_df['Group'].unique())
            super_test_ce_clean = set(super_test_df['CE_clean'].unique())
            
            missing_groups_in_super_test = all_possible_groups - super_test_groups
            missing_ce_clean_in_super_test = all_possible_ce_clean - super_test_ce_clean
            
            for missing_group in missing_groups_in_super_test:
                dummy_row = super_test_df.iloc[0:1].copy()
                dummy_row['Group'] = missing_group
                dummy_row['index_id'] = -999
                super_test_df = pd.concat([super_test_df, dummy_row], ignore_index=True)
            
            for missing_ce in missing_ce_clean_in_super_test:
                dummy_row = super_test_df.iloc[0:1].copy()
                dummy_row['CE_clean'] = missing_ce
                dummy_row['index_id'] = -998
                super_test_df = pd.concat([super_test_df, dummy_row], ignore_index=True)
            
            # Process super test set
            super_test_df['index'] = range(len(super_test_df))
            super_test_processed = fd.add_response_and_log_response(super_test_df.copy(), df6_subset, smiles_col='SMILES_spectra')
            
            # Create tensors for super test set
            x_super_test_with_ext, y_super_test_emb, y_super_test_tox, y_super_test_morgan, super_test_indices_tensor = fd.create_dataset_tensors_condenc_full2(
                super_test_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-6)
            
            # Generate predictions on super test set
            cond_encoder_current.eval()
            with torch.no_grad():
                super_test_predictions = cond_encoder_current(x_super_test_with_ext).cpu().numpy()
                super_test_tox_predictions = super_test_predictions[:, 512]
            
            # Calculate super test set median percent error
            super_test_tox_true = y_super_test_tox.cpu().numpy().flatten()
            super_test_tox_pred = super_test_tox_predictions.flatten()
            
            super_test_response_true = np.exp(super_test_tox_true)
            super_test_response_pred = np.exp(super_test_tox_pred)
            
            super_test_median_percent_error = 100 * np.median(np.abs(super_test_response_pred - super_test_response_true) / super_test_response_true)
            
            # Clean up memory
            del x_super_test_with_ext, y_super_test_emb, y_super_test_tox, y_super_test_morgan, super_test_indices_tensor
        else:
            super_test_median_percent_error = float('inf')  # Penalty if no super test samples
        
        # Clean up GPU memory
        del x_train_with_ext, x_val_with_ext, y_train_emb, y_train_tox, y_train_morgan, y_val_emb, y_val_tox, y_val_morgan
        del train_indices_tensor, val_indices_tensor, cond_encoder_current, trained_cond_encoder
        torch.cuda.empty_cache()
        
        return super_test_median_percent_error
        
    except Exception as e:
        print(f"Error in hyperparameter evaluation: {str(e)}")
        torch.cuda.empty_cache()
        return float('inf')  # Return worst possible score on error

##### ==================== MAIN HYPERPARAMETER TUNING LOOP ==================== #####

# Set up device and load data
device = fd.set_up_gpu()
name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_chemnet.parquet")
morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_morganfp.parquet")
df6_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet")
df6_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet")

# Create mappings
id_to_group = dict(zip(df6_spectra['index_id'], df6_spectra['Group']))
id_to_ce_clean = dict(zip(df6_spectra['index_id'], df6_spectra['CE_clean']))

# Define folders and get datasets
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes_df6"
dataset_files = [f for f in os.listdir(grid_search_folder) if f.endswith('.parquet') and 'df_spectra' in f]

# Filter datasets (use a subset for faster hyperparameter tuning)
allowed_bin_prefixes = ['bin1_']  # Use only one bin size for faster tuning
allowed_threshold_suffixes = ['thresh0_05']  # Use only one threshold for faster tuning

dataset_files = [f for f in dataset_files if any(f.startswith(prefix) for prefix in allowed_bin_prefixes)]
dataset_files = [f for f in dataset_files if any(suffix in f for suffix in allowed_threshold_suffixes)]
dataset_names = [f.replace('.parquet', '') for f in dataset_files]

print(f"Using {len(dataset_names)} datasets for hyperparameter tuning: {dataset_names}")

# Storage for results
hyperparameter_results = []

# Generate all hyperparameter combinations
param_names = list(hyperparameter_grid.keys())
param_values = list(hyperparameter_grid.values())
all_combinations = list(product(*param_values))

print(f"Testing {len(all_combinations)} hyperparameter combinations on {len(dataset_names)} datasets")

# Test each hyperparameter combination
for combo_idx, combo in enumerate(all_combinations, 1):
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER COMBINATION {combo_idx}/{len(all_combinations)}")
    print(f"{'='*80}")
    
    # Create hyperparameter dictionary
    hyperparams = dict(zip(param_names, combo))
    print(f"Testing: {hyperparams}")
    
    # Test on each dataset and collect results
    combo_results = []
    
    for dataset_name in dataset_names:
        print(f"\nTesting on dataset: {dataset_name}")
        
        # Load dataset
        dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.parquet")
        dataset = pd.read_parquet(dataset_path)
        
        # Train and evaluate with current hyperparameters
        super_test_error = train_and_evaluate_hyperparameters(
            hyperparams, dataset_name, dataset, device,
            name_smiles_embedding_df, morgan_df, df6_subset, df6_spectra,
            id_to_group, id_to_ce_clean
        )
        
        combo_results.append(super_test_error)
        print(f"Super Test Median % Error: {super_test_error:.2f}%")
    
    # Calculate average performance across datasets
    avg_super_test_error = np.mean(combo_results)
    
    # Store results
    result_entry = hyperparams.copy()
    result_entry['avg_super_test_median_percent_error'] = avg_super_test_error
    result_entry['super_test_errors_by_dataset'] = combo_results.copy()
    hyperparameter_results.append(result_entry)
    
    print(f"\nAverage Super Test Median % Error: {avg_super_test_error:.2f}%")

##### ==================== ANALYZE RESULTS ==================== #####

print(f"\n{'='*80}")
print("HYPERPARAMETER TUNING RESULTS")
print(f"{'='*80}")

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(hyperparameter_results)

# Sort by average super test error (best first)
results_df = results_df.sort_values('avg_super_test_median_percent_error')

print("\nTop 10 Hyperparameter Combinations (by Super Test Set Performance):")
print("="*100)

for i in range(min(10, len(results_df))):
    row = results_df.iloc[i]
    print(f"\nRank {i+1}: Avg Super Test Error = {row['avg_super_test_median_percent_error']:.2f}%")
    print(f"  num_layers: {row['num_layers']}")
    print(f"  batch_size: {row['batch_size']}")
    print(f"  learning_rate: {row['learning_rate']}")
    print(f"  lambda1: {row['lambda1']}")
    print(f"  lambda2: {row['lambda2']}")
    print(f"  lambda3: {row['lambda3']}")
    print(f"  epochs: {row['epochs']}")

# Best hyperparameters
best_hyperparams = results_df.iloc[0].to_dict()
del best_hyperparams['avg_super_test_median_percent_error']
del best_hyperparams['super_test_errors_by_dataset']

print(f"\n{'='*80}")
print("BEST HYPERPARAMETERS (Lowest Super Test Set Error)")
print(f"{'='*80}")
print(f"Best Average Super Test Median % Error: {results_df.iloc[0]['avg_super_test_median_percent_error']:.2f}%")
print("\nBest Hyperparameters:")
for param, value in best_hyperparams.items():
    print(f"  {param}: {value}")

# Save results
output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/hyperparameter_tuning_results"
os.makedirs(output_folder, exist_ok=True)

# Save detailed results
results_path = os.path.join(output_folder, "hyperparameter_tuning_results.parquet")
results_df.to_parquet(results_path, index=False)
print(f"\nSaved detailed results to: {results_path}")

# Save best hyperparameters as JSON
best_params_path = os.path.join(output_folder, "best_hyperparameters.json")
with open(best_params_path, 'w') as f:
    json.dump(best_hyperparams, f, indent=2)
print(f"Saved best hyperparameters to: {best_params_path}")

