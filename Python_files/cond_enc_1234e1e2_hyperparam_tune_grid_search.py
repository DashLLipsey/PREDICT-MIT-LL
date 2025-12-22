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
# Updated output size to include filtered Morgan fingerprints
# Assuming filtered_morgan_df has same number of bits as regular morgan (2048), total would be:
# 512 (ChemNet) + 1 (Toxicity) + 2048 (Morgan) + filtered_morgan_bits = output_size
# We'll determine the exact size dynamically from the data
output_size = None  # Will be set dynamically based on data
num_layers = 5
batch_size = 256
epochs = 800
lr = 0.0001
lambda1 = 3
lambda2 = 15
lambda3 = 1  # For regular Morgan fingerprints
lambda4 = 1  # For filtered Morgan fingerprints
# Loss functions
criterion1 = nn.MSELoss()  # ChemNet embeddings
criterion2 = nn.MSELoss()  # Toxicity
criterion3 = nn.MSELoss()  # Morgan fingerprints
criterion4 = nn.MSELoss()  # Filtered Morgan fingerprints

# Encoder architecture (With Validation Set)

# CONDITIONAL ENCODER TRAINING LOOP - Process all grid search datasets
print("=== CONDITIONAL ENCODER (ChemNet + Toxicity + Morgan + Filtered Morgan + Group + CE_clean) SUPER TEST TRAINING ===")
print(f"Super test SMILES to remove from training: {len(super_test_smiles)}")

# Set up device and load all reference datasets
device = fd.set_up_gpu()
name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_chemnet.parquet")
morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_morganfp.parquet")
filtered_morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_filtered_morganfp.parquet")

# Load the original dataset for response mapping
df6_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet")
df6_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet")

# Create Group mapping once at the start
print("Creating Group mapping from df6_spectra...")
id_to_group = dict(zip(df6_spectra['index_id'], df6_spectra['Group']))
print(f"Group mapping created with {len(id_to_group)} entries")

# Create CE_clean mapping from df6_spectra
print("Creating CE_clean mapping from df6_spectra...")
id_to_ce_clean = dict(zip(df6_spectra['index_id'], df6_spectra['CE_clean']))
print(f"CE_clean mapping created with {len(id_to_ce_clean)} entries")

# Define folders
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes_df6"

# Get all dataset files from the grid search folder
dataset_files = [f for f in os.listdir(grid_search_folder) if f.endswith('.parquet') and 'df_spectra' in f]

# ============================= HYPERPARAMETER TUNING GRID SEARCH ============================= #
import itertools
from sklearn.model_selection import ParameterGrid

# Define hyperparameter grids
hyperparameter_grid = {
    'lambda1': [1, 3, 5, 10],
    'lambda2': [5, 10, 15, 20, 30],
    'lambda3': [0.5, 1, 2, 5],
    'lambda4': [0.1, 0.5, 1, 2, 5],  # For filtered Morgan fingerprints
    'alpha1': [0.1, 0.5, 1, 2],
    'alpha2': [0.1, 0.5, 1, 2],
    'alpha3': [0.1, 0.5, 1, 2],
    'alpha4': [0.1, 0.5, 1, 2],
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

# Add Group and CE_clean columns using the mappings created above
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
        alpha1 = params['alpha1']
        alpha2 = params['alpha2']
        alpha3 = params['alpha3']
        alpha4 = params['alpha4']
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
        trained_model = fd.train_model_condenc_1234e1e2_weightloss(
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
            alpha1=alpha1,
            alpha2=alpha2,
            alpha3=alpha3,
            alpha4=alpha4,
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
            
            # Calculate alpha metric for test set: proportion of low toxicity spectra with >1000% error
            test_low_tox_mask = test_response_true < 50
            if np.sum(test_low_tox_mask) > 0:
                test_percent_errors = 100 * np.abs(test_response_pred[test_low_tox_mask] - test_response_true[test_low_tox_mask]) / test_response_true[test_low_tox_mask]
                test_high_error_count = np.sum(test_percent_errors > 1000)
                test_alpha_metric = test_high_error_count / np.sum(test_low_tox_mask)
            else:
                test_alpha_metric = 0.0
            
            # Super test set predictions (if available)
            if len(super_test_df) > 0:
                super_test_predictions = model(x_super_test_with_ext).cpu().numpy()
                super_test_tox_predictions = super_test_predictions[:, 512]
                
                super_test_tox_true = y_super_test_tox.cpu().numpy().flatten()
                super_test_tox_pred = super_test_tox_predictions.flatten()
                
                super_test_response_true = np.exp(super_test_tox_true)
                super_test_response_pred = np.exp(super_test_tox_pred)
                super_test_median_percent_error = 100 * np.median(np.abs(super_test_response_pred - super_test_response_true) / super_test_response_true)
                
                # Calculate alpha metric for super test set
                super_test_low_tox_mask = super_test_response_true < 50
                if np.sum(super_test_low_tox_mask) > 0:
                    super_test_percent_errors = 100 * np.abs(super_test_response_pred[super_test_low_tox_mask] - super_test_response_true[super_test_low_tox_mask]) / super_test_response_true[super_test_low_tox_mask]
                    super_test_high_error_count = np.sum(super_test_percent_errors > 1000)
                    super_test_alpha_metric = super_test_high_error_count / np.sum(super_test_low_tox_mask)
                else:
                    super_test_alpha_metric = 0.0
            else:
                super_test_median_percent_error = float('inf')  # Penalty if no super test data
                super_test_alpha_metric = 1.0  # Penalty if no super test data
        
        # Calculate weighted combined score (lower is better)
        # Include alpha metrics in scoring with high weight since they need to be minimized
        alpha_weight = 1000  # High weight to prioritize minimizing alpha metrics
        if len(super_test_df) > 0:
            combined_score = (delta1 * super_test_median_percent_error + 
                            delta2 * test_median_percent_error + 
                            alpha_weight * (0.7 * super_test_alpha_metric + 0.3 * test_alpha_metric))
        else:
            combined_score = test_median_percent_error + alpha_weight * test_alpha_metric
        
        print(f"Test Median % Error: {test_median_percent_error:.2f}%")
        print(f"Test Alpha Metric (low tox >1000% error): {test_alpha_metric:.4f}")
        if len(super_test_df) > 0:
            print(f"Super Test Median % Error: {super_test_median_percent_error:.2f}%")
            print(f"Super Test Alpha Metric (low tox >1000% error): {super_test_alpha_metric:.4f}")
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
            'test_alpha_metric': test_alpha_metric,
            'super_test_median_percent_error': super_test_median_percent_error if len(super_test_df) > 0 else None,
            'super_test_alpha_metric': super_test_alpha_metric if len(super_test_df) > 0 else None,
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
          f"α1={row['alpha1']}, α2={row['alpha2']}, α3={row['alpha3']}, α4={row['alpha4']}, "
          f"LR={row['learning_rate']}, Epochs={row['epochs']}, Batch={row['batch_size']}, Layers={row['num_layers']}, "
          f"Test Alpha={row['test_alpha_metric']:.4f}")

# Correlation analysis
correlation_cols = ['lambda1', 'lambda2', 'lambda3', 'lambda4', 'alpha1', 'alpha2', 'alpha3', 'alpha4', 'learning_rate', 'epochs', 'batch_size', 'num_layers']
correlations = hyperparam_df[correlation_cols + ['combined_score']].corr()['combined_score'].sort_values()
print("\nHyperparameter Correlations with Combined Score:")
for param, corr in correlations.items():
    if param != 'combined_score':
        print(f"{param}: {corr:.3f}")

print(f"\nEvaluation weights used: δ1={delta1} (super test), δ2={delta2} (regular test)")