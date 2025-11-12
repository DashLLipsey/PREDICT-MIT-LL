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
bin_size = 1   # Change this to your desired bin size (e.g., 0.05, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000)
threshold = 0.1  # Change this to your desired threshold (e.g., 0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100)

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
batch_size = 256
epochs = 250
lr = 0.0001
lambda1 = 5
lambda2 = 10
lambda3 = 1  # For Morgan fingerprints
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()
criterion3 = nn.MSELoss()

print("=== CONDITIONAL ENCODER (ChemNet + Toxicity + Morgan + Group + CE_clean) SINGLE DATASET TRAINING ===")

# Set up device and load reference datasets
device = fd.set_up_gpu()
name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_chemnet.parquet")
morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_morganfp.parquet")
df5_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_subset.parquet")
df5_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_spectra.parquet")

# Define folders
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes"
output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/cond_enc_full2_outputs"
super_test_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/super_test_sets"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(super_test_folder, exist_ok=True)

# Create the dataset name
dataset_name = create_dataset_name(bin_size, threshold)
print(f"Processing single dataset: {dataset_name}")

# Create Group mapping once at the start
print("Creating Group mapping from df5_spectra...")
id_to_group = dict(zip(df5_spectra['index_id'], df5_spectra['Group']))
print(f"Group mapping created with {len(id_to_group)} entries")

# Create CE_clean mapping from df5exp_spectra
print("Creating CE_clean mapping from df5exp_spectra...")
df5exp_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5exp_spectra.parquet")
id_to_ce_clean = dict(zip(df5exp_spectra['index_id'], df5exp_spectra['CE_clean']))
print(f"CE_clean mapping created with {len(id_to_ce_clean)} entries")

try:
    # Load dataset from parquet file
    dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.parquet")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
    dataset = pd.read_parquet(dataset_path)

    # Convert to DataFrame if it's not already one
    if not isinstance(dataset, pd.DataFrame):
        dataset = pd.DataFrame(dataset)

    print(f"Loaded {dataset_name} - Shape: {dataset.shape}")
    
    # ====== REMOVE SUPER TEST SET FROM TRAINING DATA ======
    # Save super test set before removing
    super_test_df = dataset[dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
    print(f"Super test set size: {len(super_test_df)} samples")
    print(f"Super test SMILES found: {super_test_df['SMILES_spectra'].nunique()} unique SMILES")
    
    # Remove super test set from training data
    original_count = len(dataset)
    dataset = dataset[~dataset['SMILES_spectra'].isin(super_test_smiles)]
    removed_count = original_count - len(dataset)
    print(f"Removed {removed_count} samples from super test set")
    print(f"Dataset shape after removal: {dataset.shape}")
    
    # Add Group column if not present
    if 'Group' not in dataset.columns:
        dataset = dataset.copy()  # Defragment DataFrame
        dataset['Group'] = dataset['index_id'].map(id_to_group).fillna('Unknown')
        print(f"Added Group column. Unique groups: {dataset['Group'].nunique()}")
    
    # Add CE_clean column if not present
    if 'CE_clean' not in dataset.columns:
        dataset['CE_clean'] = dataset['index_id'].map(id_to_ce_clean).fillna('Unknown')
        print(f"Added CE_clean column. Unique CE_clean values: {dataset['CE_clean'].nunique()}")

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
    x_train_with_ext, y_train_emb, y_train_tox, y_train_morgan, train_indices_tensor = fd.create_dataset_tensors_condenc_full2(
            train_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-5)

    x_val_with_ext, y_val_emb, y_val_tox, y_val_morgan, val_indices_tensor = fd.create_dataset_tensors_condenc_full2(
        test_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-5)

    # Get the actual input size and create model accordingly
    actual_input_size = x_train_with_ext.shape[1]
    print(f"Creating model with input size: {actual_input_size} for {dataset_name}")

    # Create model with correct input size
    cond_encoder_current = fd.Cond_Encoder_full(input_size=actual_input_size,
                                                 output_size=output_size, 
                                                 num_layers=num_layers).to(device)

    # Create DataLoaders
    train_dataset = TensorDataset(x_train_with_ext, y_train_emb, y_train_tox, y_train_morgan, train_indices_tensor)
    val_dataset = TensorDataset(x_val_with_ext, y_val_emb, y_val_tox, y_val_morgan, val_indices_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
            
    # Parse dataset parameters for wandb config
    parsed_bin_size, parsed_threshold = parse_dataset_name(dataset_name)
    
    # Create wandb config for this dataset
    chemnet_tox_morgan_config = {
        'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
        'wandb_project': 'MIT-Lincoln-Lab',
        'wandb_name': f"cond_enc_full2_{dataset_name}",
        'gpu': True,
        'encoder_type': "Conditional Encoder ChemNet + Toxicity + Morgan + Group + CE_clean",
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
        'Super_test': True,
    }
    
    print(f"Starting training for {epochs} epochs with learning rate {lr}...")
    
    # Train conditional encoder using the group-conditioned training function
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
        config=chemnet_tox_morgan_config
    )
    
    # Generate predictions from the 513th element (index 512) - toxicity regression output
    cond_encoder_current.eval()
    with torch.no_grad():
        # Train predictions - extract toxicity prediction from element 512
        train_predictions = cond_encoder_current(x_train_with_ext).cpu().numpy()
        train_tox_predictions = train_predictions[:, 512]  # Element 512 (0-indexed for 513th element)
        
        # Test predictions - extract toxicity prediction from element 512
        test_predictions = cond_encoder_current(x_val_with_ext).cpu().numpy()
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
    x_full_with_ext, y_full_emb, y_full_tox, y_full_morgan, full_indices_tensor = fd.create_dataset_tensors_condenc_full2(
        full_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-5)

    # Generate conditional encoder outputs
    cond_encoder_current.eval()
    with torch.no_grad():
        full_cond_outputs = cond_encoder_current(x_full_with_ext).cpu().numpy()
    
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
    predictions_filename = f"cond_enc_full2_{bin_part}_{threshold_part}_df_spectra.parquet"
    predictions_path = os.path.join(output_folder, predictions_filename)
    output_df.to_parquet(predictions_path)

    # ====== GENERATE SUPER TEST SET CONDITIONAL ENCODER OUTPUTS ======
    if len(super_test_df) > 0:
        print(f"\n=== GENERATING SUPER TEST SET CONDITIONAL ENCODER OUTPUTS ===")
        
        # Add Group column to super test set if not present
        if 'Group' not in super_test_df.columns:
            super_test_df = super_test_df.copy()
            super_test_df['Group'] = super_test_df['index_id'].map(id_to_group).fillna('Unknown')
        
        # Add CE_clean column to super test set if not present
        if 'CE_clean' not in super_test_df.columns:
            super_test_df['CE_clean'] = super_test_df['index_id'].map(id_to_ce_clean).fillna('Unknown')

        # Add index column to super_test_df (required by create_dataset_tensors_condenc_full2)
        super_test_df_with_index = super_test_df.reset_index(drop=True).copy()
        super_test_df_with_index['index'] = super_test_df_with_index.index
        
        # Process super test set to add response and log_response
        super_test_df_processed = fd.add_response_and_log_response(super_test_df_with_index.copy(), df5_subset, smiles_col='SMILES_spectra')
        
        # FIX: Ensure super test set has same structure as training data
        # Get the unique groups and CE_clean values from training data
        training_groups = set(full_data_processed['Group'].unique())
        super_test_groups = set(super_test_df_processed['Group'].unique())
        training_ce_clean = set(full_data_processed['CE_clean'].unique())
        super_test_ce_clean = set(super_test_df_processed['CE_clean'].unique())
        
        print(f"Training data groups: {training_groups}")
        print(f"Super test data groups: {super_test_groups}")
        print(f"Training data CE_clean values: {training_ce_clean}")
        print(f"Super test data CE_clean values: {super_test_ce_clean}")
        
        # Check if super test has groups not in training data
        missing_groups = super_test_groups - training_groups
        if missing_groups:
            print(f"Warning: Super test set contains groups not in training data: {missing_groups}")
            # Map missing groups to the most common training group
            most_common_group = full_data_processed['Group'].mode()[0]
            print(f"Mapping missing groups to: {most_common_group}")
            super_test_df_processed['Group'] = super_test_df_processed['Group'].replace(
                list(missing_groups), most_common_group
            )
        
        # Check if super test has CE_clean values not in training data
        missing_ce_clean = super_test_ce_clean - training_ce_clean
        if missing_ce_clean:
            print(f"Warning: Super test set contains CE_clean values not in training data: {missing_ce_clean}")
            # Map missing CE_clean values to the most common training CE_clean value
            most_common_ce_clean = full_data_processed['CE_clean'].mode()[0]
            print(f"Mapping missing CE_clean values to: {most_common_ce_clean}")
            super_test_df_processed['CE_clean'] = super_test_df_processed['CE_clean'].replace(
                list(missing_ce_clean), most_common_ce_clean
            )
        
        # Modified tensor creation function that ensures consistent group and CE_clean encoding
        def create_dataset_tensors_condenc_full2_consistent(spectra_dataset, embedding_df, morgan_df, device, 
                                                           reference_groups, reference_ce_clean, start_idx=None, stop_idx=None):
            """
            Create tensors with consistent group and CE_clean encoding based on reference values.
            """
            # Extract spectral data
            spectra = spectra_dataset.iloc[:, start_idx:stop_idx]
            
            # One-hot encode the Group column using ALL groups from reference
            group_encoded = pd.get_dummies(spectra_dataset['Group'], prefix='group', dtype=int)
            
            # Ensure all reference groups are present (add missing columns with zeros)
            for group in reference_groups:
                col_name = f'group_{group}'
                if col_name not in group_encoded.columns:
                    group_encoded[col_name] = 0
            
            # Reorder columns to match reference order (alphabetical)
            reference_group_cols = [f'group_{group}' for group in sorted(reference_groups)]
            group_encoded = group_encoded.reindex(columns=reference_group_cols, fill_value=0)
            
            # One-hot encode the CE_clean column using ALL CE_clean values from reference
            ce_encoded = pd.get_dummies(spectra_dataset['CE_clean'], prefix='ce', dtype=int)
            
            # Ensure all reference CE_clean values are present (add missing columns with zeros)
            for ce in reference_ce_clean:
                col_name = f'ce_{ce}'
                if col_name not in ce_encoded.columns:
                    ce_encoded[col_name] = 0
            
            # Reorder columns to match reference order (alphabetical)
            reference_ce_cols = [f'ce_{ce}' for ce in sorted(reference_ce_clean)]
            ce_encoded = ce_encoded.reindex(columns=reference_ce_cols, fill_value=0)
            
            print(f"Group encoding shape: {group_encoded.shape}")
            print(f"Group columns: {group_encoded.columns.tolist()}")
            print(f"CE_clean encoding shape: {ce_encoded.shape}")
            print(f"CE_clean columns: {ce_encoded.columns.tolist()}")
            
            # Concatenate spectra with group and CE_clean encoding
            spectra_with_ext = pd.concat([spectra, group_encoded, ce_encoded], axis=1)
            
            # Create chemical labels list
            chem_labels = list(spectra_dataset['SMILES_spectra'])
            
            # Create tensors
            spectra_with_ext_tensor = torch.Tensor(spectra_with_ext.values).to(device)
            embeddings_tensor = torch.Tensor([embedding_df.loc[embedding_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
            log_tox_tensor = torch.Tensor(spectra_dataset["log_response"].values).unsqueeze(1).to(device)
            morgan_tensor = torch.Tensor([morgan_df.loc[morgan_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
            spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

            return spectra_with_ext_tensor, embeddings_tensor, log_tox_tensor, morgan_tensor, spectra_indices_tensor
        
        # Create super test tensors with consistent group and CE_clean encoding
        reference_groups = full_data_processed['Group'].unique()
        reference_ce_clean = full_data_processed['CE_clean'].unique()
        x_super_test_with_ext, y_super_test_emb, y_super_test_tox, y_super_test_morgan, super_test_indices_tensor = create_dataset_tensors_condenc_full2_consistent(
            super_test_df_processed, name_smiles_embedding_df, morgan_df, device, 
            reference_groups, reference_ce_clean, start_idx=1, stop_idx=-5)
        
        print(f"Super test tensor shapes: x_super_test: {x_super_test_with_ext.shape}")
        print(f"Expected input size: {actual_input_size}")
        print(f"Actual input size: {x_super_test_with_ext.shape[1]}")
        
        # Verify the shapes match before proceeding
        if x_super_test_with_ext.shape[1] != actual_input_size:
            print(f"ERROR: Shape mismatch detected!")
            print(f"Training input size: {actual_input_size}")
            print(f"Super test input size: {x_super_test_with_ext.shape[1]}")
            print(f"Difference: {x_super_test_with_ext.shape[1] - actual_input_size}")
            
            # Try to fix by padding or truncating
            if x_super_test_with_ext.shape[1] < actual_input_size:
                # Pad with zeros
                padding_size = actual_input_size - x_super_test_with_ext.shape[1]
                padding = torch.zeros(x_super_test_with_ext.shape[0], padding_size, device=device)
                x_super_test_with_ext = torch.cat([x_super_test_with_ext, padding], dim=1)
                print(f"Padded super test tensor to shape: {x_super_test_with_ext.shape}")
            elif x_super_test_with_ext.shape[1] > actual_input_size:
                # Truncate
                x_super_test_with_ext = x_super_test_with_ext[:, :actual_input_size]
                print(f"Truncated super test tensor to shape: {x_super_test_with_ext.shape}")
        
        # Generate conditional encoder outputs for super test set
        cond_encoder_current.eval()
        with torch.no_grad():
            super_test_cond_outputs = cond_encoder_current(x_super_test_with_ext).cpu().numpy()
        
        print(f"Generated super test conditional encoder outputs shape: {super_test_cond_outputs.shape}")
        
        # Create super test conditional encoder dataset with outputs
        super_test_output_df = pd.DataFrame(super_test_cond_outputs[:, :512], columns=emb_cols)
        super_test_output_df['cond_tox_pred'] = super_test_cond_outputs[:, 512]  # Toxicity prediction
        
        # Add Morgan fingerprint predictions
        super_test_morgan_pred_df = pd.DataFrame(super_test_cond_outputs[:, 513:], columns=morgan_cols)
        super_test_output_df = pd.concat([super_test_output_df, super_test_morgan_pred_df], axis=1)
        
        # Add metadata
        super_test_output_df['SMILES_spectra'] = super_test_df_processed['SMILES_spectra'].values
        super_test_output_df['Response'] = super_test_df_processed['Response'].values
        super_test_output_df['log_response'] = super_test_df_processed['log_response'].values
        super_test_output_df['index_id'] = super_test_df_processed['index'].values
        
        # Save super test set conditional encoder outputs
        super_test_save_name = f"super_test_cond_enc_full2_{bin_part}_{threshold_part}_df_spectra"
        super_test_save_path = os.path.join(super_test_folder, f"{super_test_save_name}.parquet")
        super_test_output_df.to_parquet(super_test_save_path)
        print(f"Saved super test set conditional encoder outputs to: {super_test_save_path}")
        print(f"Super test set outputs shape: {super_test_output_df.shape}")
    else:
        print(f"\n=== NO SUPER TEST SET SMILES FOUND IN DATASET ===")

    print(f"\n=== CONDITIONAL ENCODER TRAINING COMPLETED ===")
    print(f"Dataset: {dataset_name}")
    print(f"Bin Size: {bin_size}")
    print(f"Threshold: {threshold}")
    print(f"Train Samples: {len(train_data_processed)}")
    print(f"Test Samples: {len(test_data_processed)}")
    print(f"Total Samples: {len(full_data_processed)}")
    print(f"Super Test Samples: {len(super_test_df) if len(super_test_df) > 0 else 0}")
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

    if len(super_test_df) > 0:
        print(f"\nSuccessfully created and saved conditional encoder outputs!")
        print(f"Successfully created and saved super test set conditional encoder outputs!")
    else:
        print(f"\nSuccessfully created and saved conditional encoder outputs!")
        
except Exception as e:
    print(f"Error processing {dataset_name}: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # Clear GPU memory
    if 'x_train_with_ext' in locals():
        del x_train_with_ext, x_val_with_ext, y_train_emb, y_train_tox, y_train_morgan, y_val_emb, y_val_tox, y_val_morgan
    if 'train_indices_tensor' in locals():
        del train_indices_tensor, val_indices_tensor
    if 'cond_encoder_current' in locals():
        del cond_encoder_current, trained_cond_encoder
    if 'x_super_test_with_ext' in locals():
        del x_super_test_with_ext, y_super_test_emb, y_super_test_tox, y_super_test_morgan, super_test_indices_tensor
    torch.cuda.empty_cache()

























# # ====== SUPER TEST SET SMILES (COMMENT OUT TO INCLUDE IN TRAINING) ======
# super_test_smiles = [
#     'NC(=S)Nc1ccccc1',                                  # 6
#     'COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1',   # 12
#     'CCNc1nc(Cl)nc(NC(C)(C)C#N)n1',                     # 6 
#     'C#CCN(C)Cc1ccccc1',                                # 6
#     'COP(=S)(OC)Oc1ccc(SC)c(C)c1',                      # 6
#     'Nc1cccc2c(N)cccc12',                               # 6
#     'Cn1c(=O)c2c(ncn2CCO)n(C)c1=O',                     # 40
#     'CNC(=O)N(C)c1nnc(C(C)(C)C)s1',                     # 6
#     'Nc1ccc(Sc2ccc(N)cc2)cc1',                          # 15
#     'COc1ccc2ccc(=O)oc2c1CC=C(C)C',                     # 6
# ]

# # ====== SPECIFY BIN SIZE AND THRESHOLD ======
# bin_size = 1   # Change this to your desired bin size (e.g., 0.05, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000)
# threshold = 0.01  # Change this to your desired threshold (e.g., 0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100)

# # Function to create dataset name from bin and threshold
# def create_dataset_name(bin_size, threshold):
#     """Create dataset name from bin size and threshold"""
#     bin_str = str(bin_size).replace('.', '_')
    
#     if threshold == 0:
#         return f"bin{bin_str}_thresh_zero_df_spectra"
#     else:
#         thresh_str = str(threshold).replace('.', '_')
#         return f"bin{bin_str}_thresh{thresh_str}_df_spectra"

# # Function to extract bin size and threshold from dataset name
# def parse_dataset_name(dataset_name):
#     """Extract bin size and threshold from dataset name"""
#     if 'thresh_zero' in dataset_name:
#         # Extract bin size
#         bin_part = dataset_name.split('_thresh_zero')[0].replace('bin', '')
#         bin_size = float(bin_part.replace('_', '.'))
#         threshold = 0.0
#     else:
#         # Extract bin size and threshold
#         parts = dataset_name.split('_thresh')
#         bin_part = parts[0].replace('bin', '')
#         bin_size = float(bin_part.replace('_', '.'))
        
#         thresh_part = parts[1].split('_df_spectra')[0]
#         threshold = float(thresh_part.replace('_', '.'))
    
#     return bin_size, threshold

# # Conditional Encoder Architecture: Set the parameters and the loss function
# output_size = 2561  # ChemNet + Toxicity + Morgan
# num_layers = 4
# batch_size = 256
# epochs = 250
# lr = 0.0001
# lambda1 = 5
# lambda2 = 10
# lambda3 = 1  # For Morgan fingerprints
# criterion1 = nn.MSELoss()
# criterion2 = nn.MSELoss()
# criterion3 = nn.MSELoss()

# print("=== CONDITIONAL ENCODER (ChemNet + Toxicity + Morgan + Group) SINGLE DATASET TRAINING ===")

# # Set up device and load reference datasets
# device = fd.set_up_gpu()
# name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_chemnet.parquet")
# morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_morganfp.parquet")
# df5_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_subset.parquet")
# df5_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_spectra.parquet")

# # Define folders
# grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes"
# output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/cond_enc_full_outputs"
# super_test_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/super_test_sets"
# os.makedirs(output_folder, exist_ok=True)
# os.makedirs(super_test_folder, exist_ok=True)

# # Create the dataset name
# dataset_name = create_dataset_name(bin_size, threshold)
# print(f"Processing single dataset: {dataset_name}")

# # Create Group mapping once at the start
# print("Creating Group mapping from df5_spectra...")
# id_to_group = dict(zip(df5_spectra['index_id'], df5_spectra['Group']))
# print(f"Group mapping created with {len(id_to_group)} entries")

# try:
#     # Load dataset from parquet file
#     dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.parquet")
#     if not os.path.exists(dataset_path):
#         raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
#     dataset = pd.read_parquet(dataset_path)

#     # Convert to DataFrame if it's not already one
#     if not isinstance(dataset, pd.DataFrame):
#         dataset = pd.DataFrame(dataset)

#     print(f"Loaded {dataset_name} - Shape: {dataset.shape}")
    
#     # ====== REMOVE SUPER TEST SET FROM TRAINING DATA ======
#     # Save super test set before removing
#     super_test_df = dataset[dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
#     print(f"Super test set size: {len(super_test_df)} samples")
#     print(f"Super test SMILES found: {super_test_df['SMILES_spectra'].nunique()} unique SMILES")
    
#     # Remove super test set from training data
#     original_count = len(dataset)
#     dataset = dataset[~dataset['SMILES_spectra'].isin(super_test_smiles)]
#     removed_count = original_count - len(dataset)
#     print(f"Removed {removed_count} samples from super test set")
#     print(f"Dataset shape after removal: {dataset.shape}")
    
#     # Add Group column if not present
#     if 'Group' not in dataset.columns:
#         dataset = dataset.copy()  # Defragment DataFrame
#         dataset['Group'] = dataset['index_id'].map(id_to_group).fillna('Unknown')
#         print(f"Added Group column. Unique groups: {dataset['Group'].nunique()}")

#     # Apply filtering
#     counts = dataset['SMILES_spectra'].value_counts()
#     valid_smiles = counts[counts >= 4].index
#     filtered_dataset = dataset[dataset['SMILES_spectra'].isin(valid_smiles)].copy()
    
#     print(f"After filtering (>=4 spectra per SMILES): {filtered_dataset.shape}")
    
#     # Vectorized train/test split
#     smiles_groups = filtered_dataset.groupby('SMILES_spectra')
#     train_indices = []
#     test_indices = []
    
#     np.random.seed(42)
#     for smiles, group in smiles_groups:
#         idx = group.index.values  # Use .values for faster access
#         n = len(idx)
#         np.random.shuffle(idx)
#         split = n // 2
#         test_indices.extend(idx[:split])
#         train_indices.extend(idx[split:])
    
#     train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
#     test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)
    
#     # Add index column
#     train_data['index'] = range(len(train_data))
#     test_data['index'] = range(len(test_data))
    
#     print(f"Train data shape: {train_data.shape}")
#     print(f"Test data shape: {test_data.shape}")
    
#     # Process both datasets together to avoid duplication
#     train_data_processed = fd.add_response_and_log_response(train_data.copy(), df5_subset, smiles_col='SMILES_spectra')
#     test_data_processed = fd.add_response_and_log_response(test_data.copy(), df5_subset, smiles_col='SMILES_spectra')
    
#     # Create tensors
#     print("Creating tensors...")
#     x_train_with_group, y_train_emb, y_train_tox, y_train_morgan, train_indices_tensor = fd.create_dataset_tensors_condenc_full(
#             train_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-5)

#     x_val_with_group, y_val_emb, y_val_tox, y_val_morgan, val_indices_tensor = fd.create_dataset_tensors_condenc_full(
#         test_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-5)

#     # print(x_train_with_group)
#     # sys.exit() # Here, split this into two python files ()

#     # Get the actual input size and create model accordingly
#     actual_input_size = x_train_with_group.shape[1]
#     print(f"Creating model with input size: {actual_input_size} for {dataset_name}")

#     # Create model with correct input size
#     cond_encoder_current = fd.Cond_Encoder_full(input_size=actual_input_size,
#                                                 output_size=output_size, 
#                                                 num_layers=num_layers).to(device)

#     # Create DataLoaders
#     train_dataset = TensorDataset(x_train_with_group, y_train_emb, y_train_tox, y_train_morgan, train_indices_tensor)
#     val_dataset = TensorDataset(x_val_with_group, y_val_emb, y_val_tox, y_val_morgan, val_indices_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
            
#     # Parse dataset parameters for wandb config
#     parsed_bin_size, parsed_threshold = parse_dataset_name(dataset_name)
    
#     # Create wandb config for this dataset
#     chemnet_tox_morgan_config = {
#         'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
#         'wandb_project': 'MIT-Lincoln-Lab',
#         'wandb_name': f"cond_enc_full_{dataset_name}",
#         'gpu': True,
#         'encoder_type': "Conditional Encoder ChemNet + Toxicity + Morgan + Group",
#         # Model hyperparameters
#         'batch_size': batch_size,
#         'output_size': output_size,
#         'num_layers': num_layers,
#         'learning_rate': lr,
#         'epochs': epochs,
#         'lambda1': lambda1,
#         'lambda2': lambda2,
#         'lambda3': lambda3,
#         # Dataset-specific parameters
#         'Bin': parsed_bin_size,
#         'Threshold': parsed_threshold,
#         'Super_test': True,
#     }
    
#     print(f"Starting training for {epochs} epochs with learning rate {lr}...")
    
#     # Train conditional encoder using the group-conditioned training function
#     trained_cond_encoder = fd.train_model_condenc_full(
#         model=cond_encoder_current,
#         train_data=train_loader,
#         val_data=val_loader,
#         epochs=epochs,
#         learning_rate=lr,
#         lambda1=lambda1,
#         lambda2=lambda2,
#         lambda3=lambda3,
#         criterion1=criterion1,
#         criterion2=criterion2,
#         criterion3=criterion3,
#         device=device,
#         config=chemnet_tox_morgan_config
#     )
    
#     # Generate predictions from the 513th element (index 512) - toxicity regression output
#     cond_encoder_current.eval()
#     with torch.no_grad():
#         # Train predictions - extract toxicity prediction from element 512
#         train_predictions = cond_encoder_current(x_train_with_group).cpu().numpy()
#         train_tox_predictions = train_predictions[:, 512]  # Element 512 (0-indexed for 513th element)
        
#         # Test predictions - extract toxicity prediction from element 512
#         test_predictions = cond_encoder_current(x_val_with_group).cpu().numpy()
#         test_tox_predictions = test_predictions[:, 512]  # Element 512 (0-indexed for 513th element)
    
#     # Get true toxicity values
#     train_tox_true = y_train_tox.cpu().numpy().flatten()
#     test_tox_true = y_val_tox.cpu().numpy().flatten()
#     train_tox_pred = train_tox_predictions.flatten()
#     test_tox_pred = test_tox_predictions.flatten()
    
#     # Calculate percent errors (undo log transform to get back to original response scale)
#     train_response_true = np.exp(train_tox_true)
#     train_response_pred = np.exp(train_tox_pred)
#     test_response_true = np.exp(test_tox_true)
#     test_response_pred = np.exp(test_tox_pred)
    
#     # Calculate absolute percent errors
#     train_median_percent_error = 100 * np.median(np.abs(train_response_pred - train_response_true) / train_response_true)
#     test_median_percent_error = 100 * np.median(np.abs(test_response_pred - test_response_true) / test_response_true)
#     train_mean_percent_error = 100 * np.mean(np.abs(train_response_pred - train_response_true) / train_response_true)
#     test_mean_percent_error = 100 * np.mean(np.abs(test_response_pred - test_response_true) / test_response_true)
    
#     # Generate 2561-dimensional outputs for ALL data (train + test combined)
#     full_data_processed = pd.concat([train_data_processed, test_data_processed], ignore_index=True)
    
#     # Create tensors for full dataset
#     x_full_with_group, y_full_emb, y_full_tox, y_full_morgan, full_indices_tensor = fd.create_dataset_tensors_condenc_full(
#         full_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-5)
    
#     # Generate conditional encoder outputs
#     cond_encoder_current.eval()
#     with torch.no_grad():
#         full_cond_outputs = cond_encoder_current(x_full_with_group).cpu().numpy()
    
#     print(f"Generated full conditional encoder outputs shape: {full_cond_outputs.shape}")
    
#     # Create output DataFrame with 2561 dimensions + metadata
#     emb_cols = [f'cond_emb_{j}' for j in range(512)]  # ChemNet embedding dimensions
#     morgan_cols = [f'cond_morgan_{j}' for j in range(2048)]  # Morgan fingerprint dimensions
    
#     output_df = pd.DataFrame(full_cond_outputs[:, :512], columns=emb_cols)
#     output_df['cond_tox_pred'] = full_cond_outputs[:, 512]  # Toxicity prediction
    
#     # Add Morgan fingerprint predictions
#     morgan_pred_df = pd.DataFrame(full_cond_outputs[:, 513:], columns=morgan_cols)
#     output_df = pd.concat([output_df, morgan_pred_df], axis=1)
    
#     # Add metadata
#     output_df['SMILES_spectra'] = full_data_processed['SMILES_spectra'].values
#     output_df['Response'] = full_data_processed['Response'].values
#     output_df['log_response'] = full_data_processed['log_response'].values
#     output_df['index_id'] = full_data_processed['index'].values
    
#     # Parse bin size and threshold from dataset name for filename
#     if 'thresh_zero' in dataset_name:
#         bin_part = dataset_name.split('_thresh_zero')[0]  # Keep bin format
#         threshold_part = "thresh_zero"
#     else:
#         parts = dataset_name.split('_thresh')
#         bin_part = parts[0]  # Keep bin format
        
#         thresh_part = parts[1].split('_df_spectra')[0]
#         threshold_part = f"thresh{thresh_part}"  # Keep thresh format
    
#     # Save conditional encoder outputs (2561 dimensions + 4 metadata = 2565 columns total)
#     predictions_filename = f"cond_enc_full_{bin_part}_{threshold_part}_df_spectra.parquet"
#     predictions_path = os.path.join(output_folder, predictions_filename)
#     output_df.to_parquet(predictions_path)



#     # ====== GENERATE SUPER TEST SET CONDITIONAL ENCODER OUTPUTS ======
#     if len(super_test_df) > 0:
#         print(f"\n=== GENERATING SUPER TEST SET CONDITIONAL ENCODER OUTPUTS ===")
        
#         # Add Group column to super test set if not present
#         if 'Group' not in super_test_df.columns:
#             super_test_df = super_test_df.copy()
#             super_test_df['Group'] = super_test_df['index_id'].map(id_to_group).fillna('Unknown')
        
#         # Add index column to super_test_df (required by create_dataset_tensors_condenc_full)
#         super_test_df_with_index = super_test_df.reset_index(drop=True).copy()
#         super_test_df_with_index['index'] = super_test_df_with_index.index
        
#         # Process super test set to add response and log_response
#         super_test_df_processed = fd.add_response_and_log_response(super_test_df_with_index.copy(), df5_subset, smiles_col='SMILES_spectra')
        
#         # FIX: Ensure super test set has same group structure as training data
#         # Get the unique groups from training data
#         training_groups = set(full_data_processed['Group'].unique())
#         super_test_groups = set(super_test_df_processed['Group'].unique())
        
#         print(f"Training data groups: {training_groups}")
#         print(f"Super test data groups: {super_test_groups}")
        
#         # Check if super test has groups not in training data
#         missing_groups = super_test_groups - training_groups
#         if missing_groups:
#             print(f"Warning: Super test set contains groups not in training data: {missing_groups}")
#             # Map missing groups to the most common training group
#             most_common_group = full_data_processed['Group'].mode()[0]
#             print(f"Mapping missing groups to: {most_common_group}")
#             super_test_df_processed['Group'] = super_test_df_processed['Group'].replace(
#                 list(missing_groups), most_common_group
#             )
        
#         # Modified tensor creation function that ensures consistent group encoding
#         def create_dataset_tensors_condenc_full_consistent(spectra_dataset, embedding_df, morgan_df, device, 
#                                                           reference_groups, start_idx=None, stop_idx=None):
#             """
#             Create tensors with consistent group encoding based on reference groups.
#             """
#             # Extract spectral data
#             spectra = spectra_dataset.iloc[:, start_idx:stop_idx]
            
#             # One-hot encode the Group column using ALL groups from reference
#             group_encoded = pd.get_dummies(spectra_dataset['Group'], prefix='group', dtype=int)
            
#             # Ensure all reference groups are present (add missing columns with zeros)
#             for group in reference_groups:
#                 col_name = f'group_{group}'
#                 if col_name not in group_encoded.columns:
#                     group_encoded[col_name] = 0
            
#             # Reorder columns to match reference order (alphabetical)
#             reference_cols = [f'group_{group}' for group in sorted(reference_groups)]
#             group_encoded = group_encoded.reindex(columns=reference_cols, fill_value=0)
            
#             print(f"Group encoding shape: {group_encoded.shape}")
#             print(f"Group columns: {group_encoded.columns.tolist()}")
            
#             # Concatenate spectra with group encoding
#             spectra_with_group = pd.concat([spectra, group_encoded], axis=1)
            
#             # Create chemical labels list
#             chem_labels = list(spectra_dataset['SMILES_spectra'])
            
#             # Create tensors
#             spectra_with_group_tensor = torch.Tensor(spectra_with_group.values).to(device)
#             embeddings_tensor = torch.Tensor([embedding_df.loc[embedding_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
#             log_tox_tensor = torch.Tensor(spectra_dataset["log_response"].values).unsqueeze(1).to(device)
#             morgan_tensor = torch.Tensor([morgan_df.loc[morgan_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
#             spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

#             return spectra_with_group_tensor, embeddings_tensor, log_tox_tensor, morgan_tensor, spectra_indices_tensor
        
#         # Create super test tensors with consistent group encoding
#         reference_groups = full_data_processed['Group'].unique()
#         x_super_test_with_group, y_super_test_emb, y_super_test_tox, y_super_test_morgan, super_test_indices_tensor = create_dataset_tensors_condenc_full_consistent(
#             super_test_df_processed, name_smiles_embedding_df, morgan_df, device, 
#             reference_groups, start_idx=1, stop_idx=-5)
        
#         print(f"Super test tensor shapes: x_super_test: {x_super_test_with_group.shape}")
#         print(f"Expected input size: {actual_input_size}")
#         print(f"Actual input size: {x_super_test_with_group.shape[1]}")
        
#         # Verify the shapes match before proceeding
#         if x_super_test_with_group.shape[1] != actual_input_size:
#             print(f"ERROR: Shape mismatch detected!")
#             print(f"Training input size: {actual_input_size}")
#             print(f"Super test input size: {x_super_test_with_group.shape[1]}")
#             print(f"Difference: {x_super_test_with_group.shape[1] - actual_input_size}")
            
#             # Try to fix by padding or truncating
#             if x_super_test_with_group.shape[1] < actual_input_size:
#                 # Pad with zeros
#                 padding_size = actual_input_size - x_super_test_with_group.shape[1]
#                 padding = torch.zeros(x_super_test_with_group.shape[0], padding_size, device=device)
#                 x_super_test_with_group = torch.cat([x_super_test_with_group, padding], dim=1)
#                 print(f"Padded super test tensor to shape: {x_super_test_with_group.shape}")
#             elif x_super_test_with_group.shape[1] > actual_input_size:
#                 # Truncate
#                 x_super_test_with_group = x_super_test_with_group[:, :actual_input_size]
#                 print(f"Truncated super test tensor to shape: {x_super_test_with_group.shape}")
        
#         # Generate conditional encoder outputs for super test set
#         cond_encoder_current.eval()
#         with torch.no_grad():
#             super_test_cond_outputs = cond_encoder_current(x_super_test_with_group).cpu().numpy()
        
#         print(f"Generated super test conditional encoder outputs shape: {super_test_cond_outputs.shape}")
        
#         # Create super test conditional encoder dataset with outputs
#         super_test_output_df = pd.DataFrame(super_test_cond_outputs[:, :512], columns=emb_cols)
#         super_test_output_df['cond_tox_pred'] = super_test_cond_outputs[:, 512]  # Toxicity prediction
        
#         # Add Morgan fingerprint predictions
#         super_test_morgan_pred_df = pd.DataFrame(super_test_cond_outputs[:, 513:], columns=morgan_cols)
#         super_test_output_df = pd.concat([super_test_output_df, super_test_morgan_pred_df], axis=1)
        
#         # Add metadata
#         super_test_output_df['SMILES_spectra'] = super_test_df_processed['SMILES_spectra'].values
#         super_test_output_df['Response'] = super_test_df_processed['Response'].values
#         super_test_output_df['log_response'] = super_test_df_processed['log_response'].values
#         super_test_output_df['index_id'] = super_test_df_processed['index'].values
        
#         # Save super test set conditional encoder outputs
#         super_test_save_name = f"super_test_cond_enc_full_{bin_part}_{threshold_part}_df_spectra"
#         super_test_save_path = os.path.join(super_test_folder, f"{super_test_save_name}.parquet")
#         super_test_output_df.to_parquet(super_test_save_path)
#         print(f"Saved super test set conditional encoder outputs to: {super_test_save_path}")
#         print(f"Super test set outputs shape: {super_test_output_df.shape}")
#     else:
#         print(f"\n=== NO SUPER TEST SET SMILES FOUND IN DATASET ===")

#     print(f"\n=== CONDITIONAL ENCODER TRAINING COMPLETED ===")
#     print(f"Dataset: {dataset_name}")
#     print(f"Bin Size: {bin_size}")
#     print(f"Threshold: {threshold}")
#     print(f"Train Samples: {len(train_data_processed)}")
#     print(f"Test Samples: {len(test_data_processed)}")
#     print(f"Total Samples: {len(full_data_processed)}")
#     print(f"Super Test Samples: {len(super_test_df) if len(super_test_df) > 0 else 0}")
#     print(f"Input Features: {actual_input_size}")
#     print(f"Output Dimensions: {output_size}")
#     print(f"Final Output Shape: {output_df.shape}")

#     print(f"\nToxicity Prediction Performance (from 513th encoder output):")
#     print(f"Train Median % Error: {train_median_percent_error:.1f}%")
#     print(f"Train Mean % Error: {train_mean_percent_error:.1f}%")
#     print(f"Test Median % Error: {test_median_percent_error:.1f}%")
#     print(f"Test Mean % Error: {test_mean_percent_error:.1f}%")

#     print(f"\nSaved prediction dataframe to: {predictions_filename}")
#     print(f"Full path: {predictions_path}")

#     if len(super_test_df) > 0:
#         print(f"\nSuccessfully created and saved conditional encoder outputs!")
#         print(f"Successfully created and saved super test set conditional encoder outputs!")
#     else:
#         print(f"\nSuccessfully created and saved conditional encoder outputs!")
        
# except Exception as e:
#     print(f"Error processing {dataset_name}: {str(e)}")
#     import traceback
#     traceback.print_exc()
    
# finally:
#     # Clear GPU memory
#     if 'x_train_with_group' in locals():
#         del x_train_with_group, x_val_with_group, y_train_emb, y_train_tox, y_train_morgan, y_val_emb, y_val_tox, y_val_morgan
#     if 'train_indices_tensor' in locals():
#         del train_indices_tensor, val_indices_tensor
#     if 'cond_encoder_current' in locals():
#         del cond_encoder_current, trained_cond_encoder
#     if 'x_super_test_with_group' in locals():
#         del x_super_test_with_group, y_super_test_emb, y_super_test_tox, y_super_test_morgan, super_test_indices_tensor
#     torch.cuda.empty_cache()