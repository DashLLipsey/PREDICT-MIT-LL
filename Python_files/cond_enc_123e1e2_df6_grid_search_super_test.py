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
output_size = 2561  # Changed from 513 to 2561 for ChemNet + Toxicity + Morgan
num_layers = 8
batch_size = 256
epochs = 300
lr = 0.0001
lambda1 = 3
lambda2 = 15
lambda3 = 1  # Added lambda3 for Morgan fingerprints
# criterion=nn.MSELoss() # Still use MSELoss for the embedding criterion
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()
criterion3 = nn.MSELoss()  # Added criterion3 for Morgan fingerprints

# Encoder architecture (With Validation Set)

# CONDITIONAL ENCODER TRAINING LOOP - Process all grid search datasets
print("=== CONDITIONAL ENCODER (ChemNet + Toxicity + Morgan + Group + CE_clean) SUPER TEST TRAINING ===")
print(f"Super test SMILES to remove from training: {len(super_test_smiles)}")

# Set up device and load ChemNet reference
device = fd.set_up_gpu()
name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_chemnet.parquet")

# Load Morgan fingerprint dataset
morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_morganfp.parquet")

# Load the original dataset for response mapping
df6_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet")
df6_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet")

# Define folders
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes_df6"

# Get all dataset files from the grid search folder
dataset_files = [f for f in os.listdir(grid_search_folder) if f.endswith('.parquet') and 'df_spectra' in f]

# Allow all bin sizes and thresholds
# allowed_bin_prefixes = [ 'bin0_05_','bin0_1_', 'bin0_5_', 'bin1_', 'bin2_', 'bin5_', 'bin10_',
#                         'bin25_', 'bin50_', 'bin100_', 'bin200_', 'bin500_', 'bin1000_']
# allowed_threshold_suffixes = ['thresh_zero', 'thresh0_001', 'thresh0_005', 'thresh0_01', 'thresh0_05', 
#                              'thresh0_1', 'thresh0_5', 'thresh1', 'thresh2', 'thresh5', 'thresh10', 
#                              'thresh50', 'thresh100']

# # Allow only range of interest as given by Rod/Sasha
allowed_bin_prefixes = [ 'bin0_5_','bin1_', 'bin2_'] # 'bin0_5_',
allowed_threshold_suffixes = ['thresh0_01', 'thresh0_05','thresh0_1'] # 'thresh0_05',

# Filter dataset files to only include allowed bin sizes and thresholds
dataset_files = [f for f in dataset_files if any(f.startswith(prefix) for prefix in allowed_bin_prefixes)]
dataset_files = [f for f in dataset_files if any(suffix in f for suffix in allowed_threshold_suffixes)]

dataset_names = [f.replace('.parquet', '') for f in dataset_files]

print(f"Found {len(dataset_names)} datasets to process (filtered by bin sizes and thresholds)")

# Storage for conditional encoder results
cond_encoder_results = []

# Create Group mapping once at the start
print("Creating Group mapping from df6_spectra...")
id_to_group = dict(zip(df6_spectra['index_id'], df6_spectra['Group']))
print(f"Group mapping created with {len(id_to_group)} entries")

# Create CE_clean mapping from df6_spectra
print("Creating CE_clean mapping from df6_spectra...")
id_to_ce_clean = dict(zip(df6_spectra['index_id'], df6_spectra['CE_clean']))
print(f"CE_clean mapping created with {len(id_to_ce_clean)} entries")

# Loop through each dataset and evaluate toxicity predictions from element 512 (0-indexed)
for i, dataset_name in enumerate(sorted(dataset_names), 1):
    print(f"\nProcessing {i}/{len(dataset_names)}: {dataset_name}")
    
    try:
        # Load dataset from parquet file
        dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.parquet")
        dataset = pd.read_parquet(dataset_path)

        # Convert to DataFrame if it's not already one
        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)

        print(f"Loaded {dataset_name} - Shape: {dataset.shape}")
        
        # SUPER TEST SET REMOVAL: Remove super test SMILES from training data
        original_count = len(dataset)
        dataset_no_super_test = dataset[~dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
        removed_count = original_count - len(dataset_no_super_test)
        print(f"Removed {removed_count} samples from super test set")
        print(f"Dataset shape after super test removal: {dataset_no_super_test.shape}")
        
        # OPTIMIZATION 2: Efficient Group addition using copy() to avoid fragmentation
        if 'Group' not in dataset_no_super_test.columns:
            dataset_no_super_test['Group'] = dataset_no_super_test['index_id'].map(id_to_group).fillna('Unknown')
            print(f"Added Group column. Unique groups: {dataset_no_super_test['Group'].nunique()}")
        
        # Add CE_clean column if not present
        if 'CE_clean' not in dataset_no_super_test.columns:
            dataset_no_super_test['CE_clean'] = dataset_no_super_test['index_id'].map(id_to_ce_clean).fillna('Unknown')
            print(f"Added CE_clean column. Unique CE_clean values: {dataset_no_super_test['CE_clean'].nunique()}")
                
        # Apply filtering (>=4 spectra per SMILES)
        counts = dataset_no_super_test['SMILES_spectra'].value_counts()
        valid_smiles = counts[counts >= 4].index
        filtered_dataset = dataset_no_super_test[dataset_no_super_test['SMILES_spectra'].isin(valid_smiles)].copy()
        
        print(f"After filtering (>=4 spectra per SMILES): {filtered_dataset.shape}")
        
        # OPTIMIZATION 3: Vectorized train/test split
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
        
        # OPTIMIZATION 4: Process both datasets together to avoid duplication
        train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
        test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')
        

# =================================================================================================================================
        # # SAVE BEFORE TENSOR CREATION
        # print("Saving DataFrames before tensor creation...")
        # before_tensor_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/misc_data"
        # os.makedirs(before_tensor_folder, exist_ok=True)

        # train_before_path = os.path.join(before_tensor_folder, f"{dataset_name}_train_before_tensor.parquet")
        # test_before_path = os.path.join(before_tensor_folder, f"{dataset_name}_test_before_tensor.parquet")

        # train_data_processed.to_parquet(train_before_path, index=False)
        # test_data_processed.to_parquet(test_before_path, index=False)
        # print(f"Saved train data: {train_data_processed.shape} -> {train_before_path}")
        # print(f"Saved test data: {test_data_processed.shape} -> {test_before_path}")


        # NEW CODE - Move model creation AFTER tensor creation:
        # Create tensors first
        print("Creating tensors...")
        x_train_with_ext, y_train_emb, y_train_tox, y_train_morgan, train_indices_tensor = fd.create_dataset_tensors_condenc_123e1e2(
                train_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-6)

        x_val_with_ext, y_val_emb, y_val_tox, y_val_morgan, val_indices_tensor = fd.create_dataset_tensors_condenc_123e1e2(
            test_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-6)

# ==================================================================================================================================
        # Get the actual input size and create model accordingly
        actual_input_size = x_train_with_ext.shape[1]
        print(f"Creating model with input size: {actual_input_size} for {dataset_name}")

        # Create model with correct input size
        cond_encoder_current = fd.Cond_Encoder_full(input_size=actual_input_size,
                                                     output_size=output_size, 
                                                     num_layers=num_layers).to(device)
        
        # Continue with DataLoader creation...
        train_dataset = TensorDataset(x_train_with_ext, y_train_emb, y_train_tox, y_train_morgan, train_indices_tensor)
        val_dataset = TensorDataset(x_val_with_ext, y_val_emb, y_val_tox, y_val_morgan, val_indices_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
                
        # Parse dataset parameters for wandb config
        bin_size, threshold = parse_dataset_name(dataset_name)
        
        # Create wandb config for this dataset
        chemnet_tox_morgan_config = {
            'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
            'wandb_project': 'MIT-Lincoln-Lab',
            'wandb_name': f"cond_enc_super_test_df6_{dataset_name}",
            'gpu': True,
            'encoder_type': "Conditional Encoder ChemNet + Toxicity + Morgan + Group + CE_clean (Super Test)",
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
            'Bin': bin_size,
            'Threshold': threshold,
            'super_test_removed': True,
            'super_test_smiles_count': len(super_test_smiles)
        }
        
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
        
        # Get true toxicity values for regular test set
        train_tox_true = y_train_tox.cpu().numpy().flatten()
        test_tox_true = y_val_tox.cpu().numpy().flatten()
        train_tox_pred = train_tox_predictions.flatten()
        test_tox_pred = test_tox_predictions.flatten()
        
        # Calculate percent errors for regular test set (undo log transform to get back to original response scale)
        train_response_true = np.exp(train_tox_true)
        train_response_pred = np.exp(train_tox_pred)
        test_response_true = np.exp(test_tox_true)
        test_response_pred = np.exp(test_tox_pred)
        
        # Calculate absolute percent errors for regular test set
        train_median_percent_error = 100 * np.median(np.abs(train_response_pred - train_response_true) / train_response_true)
        test_median_percent_error = 100 * np.median(np.abs(test_response_pred - test_response_true) / test_response_true)
        train_mean_percent_error = 100 * np.mean(np.abs(train_response_pred - train_response_true) / train_response_true)
        test_mean_percent_error = 100 * np.mean(np.abs(test_response_pred - test_response_true) / test_response_true)
        
        # ==================== SUPER TEST SET EVALUATION ==================== #
        # Extract super test set from original dataset
        super_test_df = dataset[dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
        
        if len(super_test_df) > 0:
            print(f"Super test set size: {len(super_test_df)} samples")
            
            # Add Group column to super test set
            if 'Group' not in super_test_df.columns:
                super_test_df['Group'] = super_test_df['index_id'].map(id_to_group).fillna('Unknown')
            
            # Add CE_clean column to super test set
            if 'CE_clean' not in super_test_df.columns:
                super_test_df['CE_clean'] = super_test_df['index_id'].map(id_to_ce_clean).fillna('Unknown')
            
            # CRITICAL FIX: Ensure both training and super test have all groups and CE_clean values
            # Define all possible groups and CE_clean values
            all_possible_groups = set(df6_spectra['Group'].unique())  # Get all groups from the full dataset
            all_possible_ce_clean = set(df6_spectra['CE_clean'].unique())  # Get all CE_clean values
            print(f"All possible groups in full dataset: {sorted(all_possible_groups)}")
            print(f"All possible CE_clean values in full dataset: {sorted(all_possible_ce_clean)}")
            
            # Check what groups and CE_clean values are in training data
            training_groups = set(filtered_dataset['Group'].unique())
            super_test_groups = set(super_test_df['Group'].unique())
            training_ce_clean = set(filtered_dataset['CE_clean'].unique())
            super_test_ce_clean = set(super_test_df['CE_clean'].unique())
            
            print(f"Training groups: {sorted(training_groups)}")
            print(f"Super test groups: {sorted(super_test_groups)}")
            print(f"Training CE_clean values: {sorted(training_ce_clean)}")
            print(f"Super test CE_clean values: {sorted(super_test_ce_clean)}")
            
            # Find missing groups and CE_clean values in training data
            missing_groups_in_training = all_possible_groups - training_groups
            missing_groups_in_super_test = all_possible_groups - super_test_groups
            missing_ce_clean_in_training = all_possible_ce_clean - training_ce_clean
            missing_ce_clean_in_super_test = all_possible_ce_clean - super_test_ce_clean
            
            if missing_groups_in_training or missing_ce_clean_in_training:
                print(f"Groups missing in training data: {sorted(missing_groups_in_training)}")
                print(f"CE_clean values missing in training data: {sorted(missing_ce_clean_in_training)}")
                
                # Add dummy samples for missing groups in training data
                for missing_group in missing_groups_in_training:
                    dummy_row = filtered_dataset.iloc[0:1].copy()  # Copy structure from first row
                    dummy_row['Group'] = missing_group
                    dummy_row['index_id'] = -999  # Use dummy index_id
                    filtered_dataset = pd.concat([filtered_dataset, dummy_row], ignore_index=True)
                    print(f"Added dummy sample for missing group: {missing_group}")
                
                # Add dummy samples for missing CE_clean values in training data
                for missing_ce in missing_ce_clean_in_training:
                    dummy_row = filtered_dataset.iloc[0:1].copy()  # Copy structure from first row
                    dummy_row['CE_clean'] = missing_ce
                    dummy_row['index_id'] = -998  # Use dummy index_id
                    filtered_dataset = pd.concat([filtered_dataset, dummy_row], ignore_index=True)
                    print(f"Added dummy sample for missing CE_clean: {missing_ce}")
                
                # Recreate train/test split with dummy samples included
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
                
                # Add index column
                train_data['index'] = range(len(train_data))
                test_data['index'] = range(len(test_data))
                
                # Reprocess data
                train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
                test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')
                
                # Recreate tensors
                print("Recreating tensors with all groups and CE_clean values...")
                x_train_with_ext, y_train_emb, y_train_tox, y_train_morgan, train_indices_tensor = fd.create_dataset_tensors_condenc_123e1e2(
                        train_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-6)

                x_val_with_ext, y_val_emb, y_val_tox, y_val_morgan, val_indices_tensor = fd.create_dataset_tensors_condenc_123e1e2(
                    test_data_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-6)
                
                # Recreate model with updated input size
                actual_input_size = x_train_with_ext.shape[1]
                print(f"Updated model input size: {actual_input_size}")
                
                # Delete old model and create new one
                del cond_encoder_current
                torch.cuda.empty_cache()
                
                cond_encoder_current = fd.Cond_Encoder_123(input_size=actual_input_size,
                                                            output_size=output_size, 
                                                            num_layers=num_layers).to(device)
                
                # Recreate DataLoaders
                train_dataset = TensorDataset(x_train_with_ext, y_train_emb, y_train_tox, y_train_morgan, train_indices_tensor)
                val_dataset = TensorDataset(x_val_with_ext, y_val_emb, y_val_tox, y_val_morgan, val_indices_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
                
                # Retrain model
                print("Retraining model with all groups and CE_clean values...")
                trained_cond_encoder = fd.train_model_condenc_123e1e2(
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
            
            if missing_groups_in_super_test:
                print(f"Groups missing in super test data: {sorted(missing_groups_in_super_test)}")
                # Add dummy samples for missing groups in super test data
                for missing_group in missing_groups_in_super_test:
                    dummy_row = super_test_df.iloc[0:1].copy()
                    dummy_row['Group'] = missing_group
                    dummy_row['index_id'] = -999  # Use dummy index_id
                    super_test_df = pd.concat([super_test_df, dummy_row], ignore_index=True)
                    print(f"Added dummy sample for missing group in super test: {missing_group}")
            
            if missing_ce_clean_in_super_test:
                print(f"CE_clean values missing in super test data: {sorted(missing_ce_clean_in_super_test)}")
                # Add dummy samples for missing CE_clean values in super test data
                for missing_ce in missing_ce_clean_in_super_test:
                    dummy_row = super_test_df.iloc[0:1].copy()
                    dummy_row['CE_clean'] = missing_ce
                    dummy_row['index_id'] = -998  # Use dummy index_id
                    super_test_df = pd.concat([super_test_df, dummy_row], ignore_index=True)
                    print(f"Added dummy sample for missing CE_clean in super test: {missing_ce}")
            
            # Add index column and process super test set
            super_test_df['index'] = range(len(super_test_df))
            super_test_processed = fd.add_response_and_log_response(super_test_df.copy(), df6_subset, smiles_col='SMILES_spectra')
            
            # Create tensors for super test set
            x_super_test_with_ext, y_super_test_emb, y_super_test_tox, y_super_test_morgan, super_test_indices_tensor = fd.create_dataset_tensors_condenc_full2(
                super_test_processed, name_smiles_embedding_df, morgan_df, device, start_idx=1, stop_idx=-6)
            
            # DEBUG: Check tensor shapes
            print(f"Training tensor shape: {x_train_with_ext.shape}")
            print(f"Super test tensor shape: {x_super_test_with_ext.shape}")
            
            # Generate predictions on super test set
            with torch.no_grad():
                super_test_predictions = cond_encoder_current(x_super_test_with_ext).cpu().numpy()
                super_test_tox_predictions = super_test_predictions[:, 512]  # Element 512 for toxicity

            # Calculate super test set percent errors
            super_test_tox_true = y_super_test_tox.cpu().numpy().flatten()
            super_test_tox_pred = super_test_tox_predictions.flatten()
            
            super_test_response_true = np.exp(super_test_tox_true)
            super_test_response_pred = np.exp(super_test_tox_pred)
            
            super_test_median_percent_error = 100 * np.median(np.abs(super_test_response_pred - super_test_response_true) / super_test_response_true)
            super_test_mean_percent_error = 100 * np.mean(np.abs(super_test_response_pred - super_test_response_true) / super_test_response_true)
            
            # Create super test output DataFrame with 2561 dimensions + metadata
            emb_cols = [f'cond_emb_{j}' for j in range(512)]  # ChemNet embedding dimensions
            morgan_cols = [f'cond_morgan_{j}' for j in range(2048)]  # Morgan fingerprint dimensions
            
            super_test_output_df = pd.DataFrame(super_test_predictions[:, :512], columns=emb_cols)
            super_test_output_df['cond_tox_pred'] = super_test_predictions[:, 512]  # Toxicity prediction
            
            # Add Morgan fingerprint predictions
            super_test_morgan_pred_df = pd.DataFrame(super_test_predictions[:, 513:], columns=morgan_cols)
            super_test_output_df = pd.concat([super_test_output_df, super_test_morgan_pred_df], axis=1)
            
            # Add metadata
            super_test_output_df['SMILES_spectra'] = super_test_processed['SMILES_spectra'].values
            super_test_output_df['Response'] = super_test_processed['Response'].values
            super_test_output_df['log_response'] = super_test_processed['log_response'].values
            super_test_output_df['index_id'] = super_test_processed['index'].values
            
            print(f"Super Test Set Performance:")
            print(f"Super Test Median % Error: {super_test_median_percent_error:.1f}%")
            print(f"Super Test Mean % Error: {super_test_mean_percent_error:.1f}%")
        else:
            print("No super test samples found in this dataset")
            super_test_median_percent_error = None
            super_test_mean_percent_error = None
            super_test_output_df = None
        
        # ==================== SAVE OUTPUTS ==================== #
        # Parse bin size and threshold from dataset name for filename
        if 'thresh_zero' in dataset_name:
            bin_part = dataset_name.split('_thresh_zero')[0]  # Keep bin0_1 format
            threshold_part = "thresh_zero"
        else:
            parts = dataset_name.split('_thresh')
            bin_part = parts[0]  # Keep bin0_1 format
            
            thresh_part = parts[1].split('_df_spectra')[0]
            threshold_part = f"thresh{thresh_part}"  # Keep thresh0_001 format
        
        # Save super test set predictions if they exist
        if super_test_output_df is not None:
            super_test_output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/super_test_sets_df6"
            os.makedirs(super_test_output_folder, exist_ok=True)

            super_test_predictions_filename = f"super_test_cond_enc_123e1e2_{bin_part}_{threshold_part}_df_spectra.parquet"
            super_test_predictions_path = os.path.join(super_test_output_folder, super_test_predictions_filename)
            
            # Robust parquet-only saving with comprehensive error handling
            try:
                super_test_output_df.to_parquet(super_test_predictions_path, index=False)
                print(f"✓ Successfully saved super test predictions to {super_test_predictions_filename}")
                print(f"  File path: {super_test_predictions_path}")
                print(f"  DataFrame shape: {super_test_output_df.shape}")
                
                # Verify the file was actually created and check its size
                if os.path.exists(super_test_predictions_path):
                    file_size = os.path.getsize(super_test_predictions_path)
                    print(f"  ✓ File confirmed: {file_size:,} bytes")
                    
                    # Quick validation - try to read it back
                    try:
                        test_df = pd.read_parquet(super_test_predictions_path)
                        print(f"  ✓ File validation passed: {test_df.shape}")
                        del test_df  # Clean up
                    except Exception as read_error:
                        print(f"  ⚠️ Warning: File saved but cannot be read back: {str(read_error)}")
                        # Try to save again with different parameters
                        try:
                            super_test_output_df.to_parquet(super_test_predictions_path, index=False, engine='pyarrow')
                            print(f"  ✓ Re-saved with pyarrow engine")
                        except Exception as retry_error:
                            print(f"  ✗ Retry with pyarrow failed: {str(retry_error)}")
                else:
                    print(f"  ✗ ERROR: File was not created at expected location")
                    
            except Exception as save_error:
                print(f"✗ ERROR saving {super_test_predictions_filename}: {str(save_error)}")
                print(f"  Error type: {type(save_error).__name__}")
                
                # Try different parquet engines
                engines_to_try = ['pyarrow', 'fastparquet']
                saved_successfully = False
                
                for engine in engines_to_try:
                    if saved_successfully:
                        break
                    try:
                        print(f"  Attempting save with {engine} engine...")
                        super_test_output_df.to_parquet(super_test_predictions_path, index=False, engine=engine)
                        
                        # Verify this save worked
                        if os.path.exists(super_test_predictions_path):
                            file_size = os.path.getsize(super_test_predictions_path)
                            print(f"  ✓ Successfully saved with {engine}: {file_size:,} bytes")
                            saved_successfully = True
                        else:
                            print(f"  ✗ {engine} engine failed - file not created")
                            
                    except Exception as engine_error:
                        print(f"  ✗ {engine} engine failed: {str(engine_error)}")
                
                if not saved_successfully:
                    print(f"  ✗ CRITICAL: All parquet save attempts failed for {dataset_name}")
                    print(f"      DataFrame info: {super_test_output_df.dtypes}")
                    print(f"      Memory usage: {super_test_output_df.memory_usage(deep=True).sum():,} bytes")
        else:
            print("No super test samples found in this dataset - skipping save")
            
        # Store results for analysis
        cond_encoder_results.append({
            'Dataset': dataset_name,
            'Train_Median_Percent_Error': train_median_percent_error,
            'Test_Median_Percent_Error': test_median_percent_error,
            'Train_Mean_Percent_Error': train_mean_percent_error,
            'Test_Mean_Percent_Error': test_mean_percent_error,
            'Super_Test_Median_Percent_Error': super_test_median_percent_error,
            'Super_Test_Mean_Percent_Error': super_test_mean_percent_error,
            'Samples': len(filtered_dataset),
            'Train_Samples': len(train_data_processed),
            'Test_Samples': len(test_data_processed),
            'Super_Test_Samples': len(super_test_df) if len(super_test_df) > 0 else 0,
            'Super_Test_Removed_Count': removed_count
        })
        
        print(f"Regular Test Set Performance:")
        print(f"Test Median % Error: {test_median_percent_error:.1f}%")
        print(f"Test Mean % Error: {test_mean_percent_error:.1f}%")

        # Clear GPU memory after each dataset
        del x_train_with_ext, x_val_with_ext, y_train_emb, y_train_tox, y_train_morgan, y_val_emb, y_val_tox, y_val_morgan
        del train_indices_tensor, val_indices_tensor
        if 'x_super_test_with_ext' in locals():
            del x_super_test_with_ext, y_super_test_emb, y_super_test_tox, y_super_test_morgan, super_test_indices_tensor
        del cond_encoder_current, trained_cond_encoder
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        # Clear memory on error too
        torch.cuda.empty_cache()
        continue

print(f"\n=== CONDITIONAL ENCODER (ChemNet + Toxicity + Morgan + Group + CE_clean) SUPER TEST EVALUATION COMPLETED ===")
print(f"Successfully processed {len(cond_encoder_results)} datasets")

# Convert results to DataFrame for analysis
df_cond_super_test_results = pd.DataFrame(cond_encoder_results)

print("\nConditional Encoder Results Summary:")
print(f"Mean Regular Test Median % Error: {df_cond_super_test_results['Test_Median_Percent_Error'].mean():.2f}%")
print(f"Mean Regular Test Mean % Error: {df_cond_super_test_results['Test_Mean_Percent_Error'].mean():.2f}%")

# Calculate super test set performance (excluding None values)
super_test_results = df_cond_super_test_results.dropna(subset=['Super_Test_Median_Percent_Error'])
if len(super_test_results) > 0:
    print(f"Mean Super Test Median % Error: {super_test_results['Super_Test_Median_Percent_Error'].mean():.2f}%")
    print(f"Mean Super Test Mean % Error: {super_test_results['Super_Test_Mean_Percent_Error'].mean():.2f}%")
    print(f"Datasets with super test samples: {len(super_test_results)}")
else:
    print("No datasets contained super test samples")

print(f"Total samples removed across all datasets: {df_cond_super_test_results['Super_Test_Removed_Count'].sum()}")