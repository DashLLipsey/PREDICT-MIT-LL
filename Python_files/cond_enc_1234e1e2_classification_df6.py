# Basic Package Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Non-basic package imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
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

# Model parameters
output_size = None  # Will be set dynamically based on data
num_layers = 4
batch_size = 256
epochs = 100
lr = 0.0001
lambda1 = 1 # For ChemNet embeddings
lambda2 = 1 # For toxicity
lambda3 = 1 # For regular Morgan fingerprints
lambda4 = 1 # For filtered Morgan fingerprints

# Loss functions
criterion1 = nn.MSELoss()  # ChemNet embeddings
criterion2 = CrossEntropyLoss()  # Toxicity classification
criterion3 = nn.MSELoss()  # Morgan fingerprints
criterion4 = nn.MSELoss()  # Filtered Morgan fingerprints

# CONDITIONAL ENCODER TRAINING LOOP - Process all grid search datasets
print("=== CONDITIONAL ENCODER (ChemNet + Toxicity Classification + Morgan + Filtered Morgan + Group + CE_clean) TRAINING ===")
print(f"Super test SMILES to remove from training: {len(super_test_smiles)}")

# Set up device
device = fd.set_up_gpu()

# Load in internal condition true values
name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_chemnet.parquet")
morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_morganfp.parquet")
filtered_morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_filtered_morganfp.parquet")

# # Load in internal conditions with noise
# name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_chemnet_noise.parquet")
# morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_morganfp_noise.parquet")
# filtered_morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_filtered_morganfp_noise.parquet")

# Load the original dataset for response mapping
df6_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet")
df6_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet")

# Define folders
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes_df6"

# Get all dataset files from the grid search folder
dataset_files = [f for f in os.listdir(grid_search_folder) if f.endswith('.parquet') and 'df_spectra' in f]

# # Allow only range of interest as given by Rod/Sasha
# allowed_bin_prefixes = ['bin0_5_', 'bin1_', 'bin2_', 'bin5_']
# allowed_threshold_suffixes = ['thresh0_01', 'thresh0_05', 'thresh0_1']

# allowed_bin_prefixes = ['bin0_1_', 'bin10_', 'bin25_', 'bin50_']
# allowed_threshold_suffixes = ['thresh_zero', 'thresh0_01', 'thresh0_05', 'thresh0_1']

# allowed_bin_prefixes = ['bin0_5_', 'bin1_', 'bin2_', 'bin5_']
# allowed_threshold_suffixes = ['thresh_zero']
# allowed_bin_prefixes = ['bin100_', 'bin200_', 'bin500_']
# Allow all bin sizes and thresholds

# allowed_bin_prefixes = ['bin0_1_', 'bin0_5_', 'bin1_', 'bin2_', 'bin5_', 'bin10_',
#                         'bin25_', 'bin50_', 'bin100_', 'bin200_', 'bin500_'] # 'bin0_05' 
# allowed_threshold_suffixes = ['thresh_zero', 'thresh0_001', 'thresh0_005', 'thresh0_01', 'thresh0_05', 
#                              'thresh0_1', 'thresh0_5', 'thresh1', 'thresh2', 'thresh5', 'thresh10', 
#                              'thresh50', 'thresh100']
# Allow all bin sizes and thresholds
allowed_bin_prefixes = ['bin1_'] 
allowed_threshold_suffixes = ['thresh0_05']

# Filter dataset files to only include allowed bin sizes and thresholds
dataset_files = [f for f in dataset_files if any(f.startswith(prefix) for prefix in allowed_bin_prefixes)]
dataset_files = [f for f in dataset_files if any(suffix in f for suffix in allowed_threshold_suffixes)]

dataset_names = [f.replace('.parquet', '') for f in dataset_files]

print(f"Found {len(dataset_names)} datasets to process (filtered by bin sizes and thresholds)")

# Create Group and CE_clean mappings
print("Creating Group mapping from df6_spectra...")
id_to_group = dict(zip(df6_spectra['index_id'], df6_spectra['Group']))
print(f"Group mapping created with {len(id_to_group)} entries")

print("Creating CE_clean mapping from df6_spectra...")
id_to_ce_clean = dict(zip(df6_spectra['index_id'], df6_spectra['CE_clean']))
print(f"CE_clean mapping created with {len(id_to_ce_clean)} entries")

# Calculate output size from the fingerprint data
regular_morgan_bits = morgan_df.shape[1] - 1  # Subtract 1 for SMILES_spectra column
filtered_morgan_bits = filtered_morgan_df.shape[1] - 1  # Subtract 1 for SMILES_spectra column
output_size = 512 + 4 + regular_morgan_bits + filtered_morgan_bits  # ChemNet + 4-class Toxicity + Morgan + Filtered Morgan
print(f"Calculated output size: {output_size} (512 ChemNet + 4 Toxicity Classes + {regular_morgan_bits} Morgan + {filtered_morgan_bits} Filtered Morgan)")

# Loop through each dataset and train one model, then evaluate on both full validation and super test sets
for i, dataset_name in enumerate(sorted(dataset_names), 1):
    print(f"\nProcessing {i}/{len(dataset_names)}: {dataset_name}")
    
    try:
        # Load dataset from parquet file
        dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.parquet")
        dataset = pd.read_parquet(dataset_path)

        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)

        print(f"Loaded {dataset_name} - Shape: {dataset.shape}")
        
        # Remove super test SMILES from training data
        original_count = len(dataset)
        dataset_no_super_test = dataset[~dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
        removed_count = original_count - len(dataset_no_super_test)
        print(f"Removed {removed_count} samples from super test set")
        
        # Add Group and CE_clean columns
        if 'Group' not in dataset_no_super_test.columns:
            dataset_no_super_test['Group'] = dataset_no_super_test['index_id'].map(id_to_group).fillna('Unknown')
        
        if 'CE_clean' not in dataset_no_super_test.columns:
            dataset_no_super_test['CE_clean'] = dataset_no_super_test['index_id'].map(id_to_ce_clean).fillna('Unknown')
                
        # Apply filtering (>=4 spectra per SMILES)
        counts = dataset_no_super_test['SMILES_spectra'].value_counts()
        valid_smiles = counts[counts >= 4].index
        filtered_dataset = dataset_no_super_test[dataset_no_super_test['SMILES_spectra'].isin(valid_smiles)].copy()
        
        print(f"After filtering (>=4 spectra per SMILES): {filtered_dataset.shape}")
        
        # Train/test split for model training
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
        
        # Create set of training indices for later tracking
        train_indices_set = set(train_indices)
        
        # Add index column
        train_data['index'] = range(len(train_data))
        test_data['index'] = range(len(test_data))
                    
        # Process datasets - add response and EPA levels
        train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
        train_data_processed = fd.add_epa_levels(train_data_processed)
        
        test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')
        test_data_processed = fd.add_epa_levels(test_data_processed)

        # Create tensors for training
        print("Creating training tensors...")
        x_train_with_ext, y_train_emb, y_train_tox_class, y_train_morgan, y_train_filtered_morgan, train_indices_tensor = fd.create_dataset_tensors_condenc_1234e1e2_class(
                train_data_processed, name_smiles_embedding_df, morgan_df, filtered_morgan_df, device, start_idx=1, stop_idx=-10)

        x_val_with_ext, y_val_emb, y_val_tox_class, y_val_morgan, y_val_filtered_morgan, val_indices_tensor = fd.create_dataset_tensors_condenc_1234e1e2_class(
            test_data_processed, name_smiles_embedding_df, morgan_df, filtered_morgan_df, device, start_idx=1, stop_idx=-10)

        # Create model
        actual_input_size = x_train_with_ext.shape[1]
        print(f"Creating model with input size: {actual_input_size}")

        cond_encoder_current = fd.Cond_Encoder_1234_class_dropout(input_size=actual_input_size,
                                                             output_size=output_size, 
                                                             num_layers=num_layers, dropout_rate=0.3).to(device)
        # Create DataLoaders for training
        train_dataset = TensorDataset(x_train_with_ext, y_train_emb, y_train_tox_class, y_train_morgan, y_train_filtered_morgan, train_indices_tensor)
        val_dataset = TensorDataset(x_val_with_ext, y_val_emb, y_val_tox_class, y_val_morgan, y_val_filtered_morgan, val_indices_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
                
        # Parse dataset parameters for wandb config
        bin_size, threshold = parse_dataset_name(dataset_name)
        
        # Create wandb config
        chemnet_tox_morgan_config = {
            'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
            'wandb_project': 'MIT-Lincoln-Lab',
            'wandb_name': f"cond_enc_class_filtered_df6_{dataset_name}",
            'gpu': True,
            'encoder_type': "Conditional Encoder ChemNet + Toxicity Classification + Morgan + Filtered Morgan + Group + CE_clean",
            'batch_size': batch_size,
            'output_size': output_size,
            'num_layers': num_layers,
            'learning_rate': lr,
            'epochs': epochs,
            'lambda1': lambda1,
            'lambda2': lambda2,
            'lambda3': lambda3,
            'lambda4': lambda4,
            'Bin': bin_size,
            'Threshold': threshold,
            'super_test_removed': True,
        }

        # ==================== TRAIN MODEL ==================== #
        print("Training model...")
        trained_cond_encoder = fd.train_model_condenc_1234e1e2_class_iap(
            model=cond_encoder_current,
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
            config=chemnet_tox_morgan_config,
            recompute_interval=50
        )
        
        # ==================== EVALUATE ON FULL VALIDATION SET ==================== #
        print("Evaluating on full validation set...")
        # Prepare full filtered dataset
        filtered_dataset_full = filtered_dataset.copy()
        
        # Create train indicator mapping based on original index
        # 1 = training spectra, 0 = validation/test spectra
        train_indicator_map = {}
        for idx in filtered_dataset_full.index:
            train_indicator_map[idx] = 1 if idx in train_indices_set else 0
        
        # Reset index and add sequential index column
        filtered_dataset_full = filtered_dataset_full.reset_index(drop=False, names=['original_index'])
        filtered_dataset_full['index'] = range(len(filtered_dataset_full))
        
        # Process dataset
        filtered_dataset_full_processed = fd.add_response_and_log_response(filtered_dataset_full.copy(), df6_subset, smiles_col='SMILES_spectra')
        filtered_dataset_full_processed = fd.add_epa_levels(filtered_dataset_full_processed)
        
        # Add train indicator using the original index mapping
        filtered_dataset_full_processed['train'] = filtered_dataset_full_processed['original_index'].map(train_indicator_map).fillna(0).astype(int)
        
        # Create a copy for tensor creation (without metadata columns that shouldn't be tensors)
        filtered_dataset_for_tensors = filtered_dataset_full_processed.drop(columns=['original_index', 'train']).copy()
        
        # Create tensors for full validation set
        x_full_val_with_ext, y_full_val_emb, y_full_val_tox_class, y_full_val_morgan, y_full_val_filtered_morgan, full_val_indices_tensor = fd.create_dataset_tensors_condenc_1234e1e2_class(
            filtered_dataset_for_tensors, name_smiles_embedding_df, morgan_df, filtered_morgan_df, device, start_idx=1, stop_idx=-10)
        
        # Generate predictions on full validation set
        cond_encoder_current.eval()
        with torch.no_grad():
            full_val_predictions = cond_encoder_current(x_full_val_with_ext).cpu().numpy()

        # Create output DataFrame for full validation set
        emb_cols = [f'cond_emb_{j}' for j in range(512)]
        morgan_cols = [f'cond_morgan_{j}' for j in range(regular_morgan_bits)]
        filtered_morgan_cols = [f'cond_filtered_morgan_{j}' for j in range(filtered_morgan_bits)]
        
        full_val_output_df = pd.DataFrame(full_val_predictions[:, :512], columns=emb_cols)
        
        # Add toxicity classification predictions (4 logits)
        tox_logits_cols = [f'cond_tox_logit_{j}' for j in range(4)]
        full_val_output_df[tox_logits_cols] = full_val_predictions[:, 512:516]
        
        # Add predicted class (argmax of logits)
        full_val_output_df['cond_tox_pred_class'] = np.argmax(full_val_predictions[:, 512:516], axis=1)
        
        # Add Morgan fingerprint predictions
        morgan_start_idx = 516
        morgan_end_idx = morgan_start_idx + regular_morgan_bits
        full_val_morgan_pred_df = pd.DataFrame(full_val_predictions[:, morgan_start_idx:morgan_end_idx], columns=morgan_cols)
        full_val_output_df = pd.concat([full_val_output_df, full_val_morgan_pred_df], axis=1)
        
        # Add Filtered Morgan fingerprint predictions
        filtered_morgan_start_idx = morgan_end_idx
        full_val_filtered_morgan_pred_df = pd.DataFrame(full_val_predictions[:, filtered_morgan_start_idx:], columns=filtered_morgan_cols)
        full_val_output_df = pd.concat([full_val_output_df, full_val_filtered_morgan_pred_df], axis=1)
        
        # Add metadata
        full_val_output_df['SMILES_spectra'] = filtered_dataset_full_processed['SMILES_spectra'].values
        full_val_output_df['Response'] = filtered_dataset_full_processed['Response'].values
        full_val_output_df['log_response'] = filtered_dataset_full_processed['log_response'].values
        full_val_output_df['index_id'] = filtered_dataset_full_processed['index'].values
        full_val_output_df['train'] = filtered_dataset_full_processed['train'].values
        
        # Verify train column
        train_count = full_val_output_df['train'].sum()
        val_count = len(full_val_output_df) - train_count
        print(f"Train column added: {train_count} training samples, {val_count} validation samples")
        
        # Save full validation set predictions
        if 'thresh_zero' in dataset_name:
            bin_part = dataset_name.split('_thresh_zero')[0]
            threshold_part = "thresh_zero"
        else:
            parts = dataset_name.split('_thresh')
            bin_part = parts[0]
            thresh_part = parts[1].split('_df_spectra')[0]
            threshold_part = f"thresh{thresh_part}"
        
        full_val_output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/cond_enc_1234e1e2_classification_df6"
        os.makedirs(full_val_output_folder, exist_ok=True)

        full_val_predictions_filename = f"cond_enc_{bin_part}_{threshold_part}_df_spectra.parquet"
        full_val_predictions_path = os.path.join(full_val_output_folder, full_val_predictions_filename)
        
        try:
            full_val_output_df.to_parquet(full_val_predictions_path, index=False)
            print(f"✓ Successfully saved full validation predictions to {full_val_predictions_filename}")
        except Exception as save_error:
            print(f"✗ ERROR saving full validation predictions: {str(save_error)}")
        
        # ==================== EVALUATE ON SUPER TEST SET ==================== #
        print("Evaluating on super test set...")
        # Extract super test set from original dataset
        super_test_df = dataset[dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
        
        if len(super_test_df) > 0:
            print(f"Super test set size: {len(super_test_df)} samples")
            
            # Add Group and CE_clean columns to super test set
            if 'Group' not in super_test_df.columns:
                super_test_df['Group'] = super_test_df['index_id'].map(id_to_group).fillna('Unknown')
            
            if 'CE_clean' not in super_test_df.columns:
                super_test_df['CE_clean'] = super_test_df['index_id'].map(id_to_ce_clean).fillna('Unknown')
            
            # Process super test set
            super_test_df['index'] = range(len(super_test_df))
            super_test_processed = fd.add_response_and_log_response(super_test_df.copy(), df6_subset, smiles_col='SMILES_spectra')
            super_test_processed = fd.add_epa_levels(super_test_processed)
            
            # Create tensors for super test set
            x_super_test_with_ext, y_super_test_emb, y_super_test_tox_class, y_super_test_morgan, y_super_test_filtered_morgan, super_test_indices_tensor = fd.create_dataset_tensors_condenc_1234e1e2_class(
                super_test_processed, name_smiles_embedding_df, morgan_df, filtered_morgan_df, device, start_idx=1, stop_idx=-10)
            
            # Generate predictions on super test set
            with torch.no_grad():
                super_test_predictions = cond_encoder_current(x_super_test_with_ext).cpu().numpy()

            # Create super test output DataFrame
            super_test_output_df = pd.DataFrame(super_test_predictions[:, :512], columns=emb_cols)
            
            # Add toxicity classification predictions (4 logits)
            super_test_output_df[tox_logits_cols] = super_test_predictions[:, 512:516]
            
            # Add predicted class (argmax of logits)
            super_test_output_df['cond_tox_pred_class'] = np.argmax(super_test_predictions[:, 512:516], axis=1)
            
            # Add Morgan fingerprint predictions
            super_test_morgan_pred_df = pd.DataFrame(super_test_predictions[:, morgan_start_idx:morgan_end_idx], columns=morgan_cols)
            super_test_output_df = pd.concat([super_test_output_df, super_test_morgan_pred_df], axis=1)
            
            # Add Filtered Morgan fingerprint predictions
            super_test_filtered_morgan_pred_df = pd.DataFrame(super_test_predictions[:, filtered_morgan_start_idx:], columns=filtered_morgan_cols)
            super_test_output_df = pd.concat([super_test_output_df, super_test_filtered_morgan_pred_df], axis=1)
            
            # Add metadata
            super_test_output_df['SMILES_spectra'] = super_test_processed['SMILES_spectra'].values
            super_test_output_df['Response'] = super_test_processed['Response'].values
            super_test_output_df['log_response'] = super_test_processed['log_response'].values
            super_test_output_df['index_id'] = super_test_processed['index'].values
            super_test_output_df['train'] = 0  # Super test samples were not in training set
            
            # Save super test set predictions
            super_test_output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/cond_enc_1234e1e2_classification_df6_super_test"
            os.makedirs(super_test_output_folder, exist_ok=True)

            super_test_predictions_filename = f"super_test_cond_enc_{bin_part}_{threshold_part}_df_spectra.parquet"
            super_test_predictions_path = os.path.join(super_test_output_folder, super_test_predictions_filename)
            
            try:
                super_test_output_df.to_parquet(super_test_predictions_path, index=False)
                print(f"✓ Successfully saved super test predictions to {super_test_predictions_filename}")
            except Exception as save_error:
                print(f"✗ ERROR saving super test predictions: {str(save_error)}")
            
            print(f"Super test evaluation completed for {len(super_test_df)} samples")
        else:
            print("No super test samples found in this dataset")
            
        print(f"Completed processing {dataset_name}")

        # Clear GPU memory after each dataset
        del x_train_with_ext, x_val_with_ext, y_train_emb, y_train_tox_class, y_train_morgan, y_train_filtered_morgan
        del y_val_emb, y_val_tox_class, y_val_morgan, y_val_filtered_morgan, train_indices_tensor, val_indices_tensor
        del x_full_val_with_ext, y_full_val_emb, y_full_val_tox_class, y_full_val_morgan, y_full_val_filtered_morgan, full_val_indices_tensor
        if 'x_super_test_with_ext' in locals():
            del x_super_test_with_ext, y_super_test_emb, y_super_test_tox_class, y_super_test_morgan, y_super_test_filtered_morgan, super_test_indices_tensor
        del cond_encoder_current, trained_cond_encoder
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        torch.cuda.empty_cache()
        continue

print(f"\n=== CONDITIONAL ENCODER CLASSIFICATION TRAINING AND EVALUATION COMPLETED ===")
print(f"Successfully processed datasets")






# # Basic Package Imports
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Non-basic package imports
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from torch.nn import CrossEntropyLoss
# import requests

# # Packages I don't understand
# from fcd_torch import FCD
# import rdkit
# from collections import Counter
# import gc
# import pickle
# import wandb

# # Add the Python_files directory to the Python path
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'Python_files'))

# # Now you can import your modules
# import functions_enc as f
# import function_depot as fd

# ##### ==================== SUPER TEST SET SMILES ==================== #####
# # Define super test set SMILES to remove from training
# super_test_smiles = [
#     'COC(=O)C=C(C)OP(=O)(OC)OC',
#     'COc1cc2c(c3oc(=O)c4c(c13)CCC4=O)[C@@H]1C=CO[C@@H]1O2',
#     'CC(=O)OC1(C)CC(C)C(=O)C(C(O)CC2CC(=O)NC(=O)C2)C1',
#     'C[C@H]1O[C@@H](O[C@H]2[C@@H](O)C[C@H](O[C@H]3[C@@H](O)C[C@H](O[C@H]4CC[C@@]5(C)[C@H](CC[C@@H]6[C@@H]5C[C@@H](O)[C@]5(C)[C@@H](C7=CC(=O)OC7)CC[C@]65O)C4)O[C@@H]3C)O[C@@H]2C)C[C@H](O)[C@@H]1O',
#     'CNC(=O)Oc1cc(C)cc(C(C)C)c1',
#     'CNC(=O)Oc1ccc(N(C)C)c(C)c1',
#     'C[C@@H]1Cc2c(Cl)cc(C(=O)N[C@@H](Cc3ccccc3)C(=O)O)c(O)c2C(=O)O1',
#     'COc1ccc2c(c1)c(CC(=O)OCC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1',
#     'CC(C)(C)CC(C)(C)c1ccc(OCCOCC[N+](C)(C)Cc2ccccc2)cc1',
#     'CC(=O)N1CCN(c2ccc(OC[C@H]3CO[C@](Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1',
#     'c1ccc(C2CN3CCSC3=N2)cc1',
#     'CN(C)CCC=C1c2ccccc2CCc2ccccc21',
#     'CCOP(=S)(OCC)Oc1ccc2c(C)c(Cl)c(=O)oc2c1',
#     'CC(C)NCC(O)COc1cccc2ccccc12',
#     'CCOC(=O)C(C)(C)Oc1ccc(Cl)cc1',
#     'CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC',
#     'COc1ccc2c(c1OC)C(=O)OC2C1c2cc3c(cc2CCN1C)OCO3',
#     'CN(C)c1ccc(SC#N)cc1',
#     'CC(C)[C@H](N)C(=O)O',
#     'CCCCOC(=O)COC(=O)c1ccccc1C(=O)OCCCC',
#     'NC(C(=O)O)c1ccccc1'
# ]

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

# # Storage for conditional encoder results
# cond_encoder_results = [] 

# # Model parameters
# output_size = None  # Will be set dynamically based on data
# num_layers = 5
# batch_size = 256
# epochs = 250
# lr = 0.0001
# lambda1 = 10
# lambda2 = 0.01
# lambda3 = 10  # For regular Morgan fingerprints
# lambda4 = 10 # For filtered Morgan fingerprints

# # Loss functions
# criterion1 = nn.MSELoss()  # ChemNet embeddings
# criterion2 = CrossEntropyLoss()  # Toxicity classification
# criterion3 = nn.MSELoss()  # Morgan fingerprints
# criterion4 = nn.MSELoss()  # Filtered Morgan fingerprints

# # CONDITIONAL ENCODER TRAINING LOOP - Process all grid search datasets
# print("=== CONDITIONAL ENCODER (ChemNet + Toxicity Classification + Morgan + Filtered Morgan + Group + CE_clean) TRAINING ===")
# print(f"Super test SMILES to remove from training: {len(super_test_smiles)}")

# # Set up device
# device = fd.set_up_gpu()

# # # Load in internal condition true values
# # name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_chemnet.parquet")
# # morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_morganfp.parquet")
# # filtered_morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_filtered_morganfp.parquet")

# # Load in internal conditions with noise
# name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_chemnet_noise.parquet")
# morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_morganfp_noise.parquet")
# filtered_morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_filtered_morganfp_noise.parquet")

# # Load the original dataset for response mapping
# df6_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet")
# df6_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet")

# # Define folders
# grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes_df6"

# # Get all dataset files from the grid search folder
# dataset_files = [f for f in os.listdir(grid_search_folder) if f.endswith('.parquet') and 'df_spectra' in f]

# # Allow only range of interest as given by Rod/Sasha
# # allowed_bin_prefixes = ['bin0_5_', 'bin1_', 'bin2_', 'bin5_']
# # allowed_threshold_suffixes = ['thresh0_01', 'thresh0_05', 'thresh0_1']

# # allowed_bin_prefixes = ['bin0_1_', 'bin10_', 'bin25_', 'bin50_']
# # allowed_threshold_suffixes = ['thresh_zero', 'thresh0_01', 'thresh0_05', 'thresh0_1']

# # allowed_bin_prefixes = ['bin0_5_', 'bin1_', 'bin2_', 'bin5_']
# # allowed_threshold_suffixes = ['thresh_zero']
# # allowed_bin_prefixes = ['bin100_', 'bin200_', 'bin500_']
# # Allow all bin sizes and thresholds

# # allowed_bin_prefixes = ['bin0_1_', 'bin0_5_', 'bin1_', 'bin2_', 'bin5_', 'bin10_',
# #                         'bin25_', 'bin50_', 'bin100_', 'bin200_', 'bin500_'] # 'bin0_05' 
# # allowed_threshold_suffixes = ['thresh_zero', 'thresh0_001', 'thresh0_005', 'thresh0_01', 'thresh0_05', 
# #                              'thresh0_1', 'thresh0_5', 'thresh1', 'thresh2', 'thresh5', 'thresh10', 
# #                              'thresh50', 'thresh100']
# # Allow all bin sizes and thresholds
# allowed_bin_prefixes = ['bin1_'] 
# allowed_threshold_suffixes = ['thresh0_05']

# # Filter dataset files to only include allowed bin sizes and thresholds
# dataset_files = [f for f in dataset_files if any(f.startswith(prefix) for prefix in allowed_bin_prefixes)]
# dataset_files = [f for f in dataset_files if any(suffix in f for suffix in allowed_threshold_suffixes)]

# dataset_names = [f.replace('.parquet', '') for f in dataset_files]

# print(f"Found {len(dataset_names)} datasets to process (filtered by bin sizes and thresholds)")

# # Create Group and CE_clean mappings
# print("Creating Group mapping from df6_spectra...")
# id_to_group = dict(zip(df6_spectra['index_id'], df6_spectra['Group']))
# print(f"Group mapping created with {len(id_to_group)} entries")

# print("Creating CE_clean mapping from df6_spectra...")
# id_to_ce_clean = dict(zip(df6_spectra['index_id'], df6_spectra['CE_clean']))
# print(f"CE_clean mapping created with {len(id_to_ce_clean)} entries")

# # Calculate output size from the fingerprint data
# regular_morgan_bits = morgan_df.shape[1] - 1  # Subtract 1 for SMILES_spectra column
# filtered_morgan_bits = filtered_morgan_df.shape[1] - 1  # Subtract 1 for SMILES_spectra column
# output_size = 512 + 4 + regular_morgan_bits + filtered_morgan_bits  # ChemNet + 4-class Toxicity + Morgan + Filtered Morgan
# print(f"Calculated output size: {output_size} (512 ChemNet + 4 Toxicity Classes + {regular_morgan_bits} Morgan + {filtered_morgan_bits} Filtered Morgan)")

# # Loop through each dataset and train one model, then evaluate on both full validation and super test sets
# for i, dataset_name in enumerate(sorted(dataset_names), 1):
#     print(f"\nProcessing {i}/{len(dataset_names)}: {dataset_name}")
    
#     try:
#         # Load dataset from parquet file
#         dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.parquet")
#         dataset = pd.read_parquet(dataset_path)

#         if not isinstance(dataset, pd.DataFrame):
#             dataset = pd.DataFrame(dataset)

#         print(f"Loaded {dataset_name} - Shape: {dataset.shape}")
        
#         # Remove super test SMILES from training data
#         original_count = len(dataset)
#         dataset_no_super_test = dataset[~dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
#         removed_count = original_count - len(dataset_no_super_test)
#         print(f"Removed {removed_count} samples from super test set")
        
#         # Add Group and CE_clean columns
#         if 'Group' not in dataset_no_super_test.columns:
#             dataset_no_super_test['Group'] = dataset_no_super_test['index_id'].map(id_to_group).fillna('Unknown')
        
#         if 'CE_clean' not in dataset_no_super_test.columns:
#             dataset_no_super_test['CE_clean'] = dataset_no_super_test['index_id'].map(id_to_ce_clean).fillna('Unknown')
                
#         # Apply filtering (>=4 spectra per SMILES)
#         counts = dataset_no_super_test['SMILES_spectra'].value_counts()
#         valid_smiles = counts[counts >= 4].index
#         filtered_dataset = dataset_no_super_test[dataset_no_super_test['SMILES_spectra'].isin(valid_smiles)].copy()
        
#         print(f"After filtering (>=4 spectra per SMILES): {filtered_dataset.shape}")
        
#         # Train/test split for model training
#         smiles_groups = filtered_dataset.groupby('SMILES_spectra')
#         train_indices = []
#         test_indices = []
        
#         np.random.seed(42)
#         for smiles, group in smiles_groups:
#             idx = group.index.values
#             n = len(idx)
#             np.random.shuffle(idx)
#             split = n // 2
#             test_indices.extend(idx[:split])
#             train_indices.extend(idx[split:])
        
#         train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
#         test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)
        
#         # Add index column
#         train_data['index'] = range(len(train_data))
#         test_data['index'] = range(len(test_data))
                    
#         # Process datasets - add response and EPA levels
#         train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
#         train_data_processed = fd.add_epa_levels(train_data_processed)
        
#         test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')
#         test_data_processed = fd.add_epa_levels(test_data_processed)

#         # Create tensors for training
#         print("Creating training tensors...")
#         x_train_with_ext, y_train_emb, y_train_tox_class, y_train_morgan, y_train_filtered_morgan, train_indices_tensor = fd.create_dataset_tensors_condenc_1234e1e2_class(
#                 train_data_processed, name_smiles_embedding_df, morgan_df, filtered_morgan_df, device, start_idx=1, stop_idx=-10)

#         x_val_with_ext, y_val_emb, y_val_tox_class, y_val_morgan, y_val_filtered_morgan, val_indices_tensor = fd.create_dataset_tensors_condenc_1234e1e2_class(
#             test_data_processed, name_smiles_embedding_df, morgan_df, filtered_morgan_df, device, start_idx=1, stop_idx=-10)

#         # Create model
#         actual_input_size = x_train_with_ext.shape[1]
#         print(f"Creating model with input size: {actual_input_size}")

#         cond_encoder_current = fd.Cond_Encoder_1234(input_size=actual_input_size,
#                                                              output_size=output_size, 
#                                                              num_layers=num_layers).to(device)
        
#         # Create DataLoaders for training
#         train_dataset = TensorDataset(x_train_with_ext, y_train_emb, y_train_tox_class, y_train_morgan, y_train_filtered_morgan, train_indices_tensor)
#         val_dataset = TensorDataset(x_val_with_ext, y_val_emb, y_val_tox_class, y_val_morgan, y_val_filtered_morgan, val_indices_tensor)
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
                
#         # Parse dataset parameters for wandb config
#         bin_size, threshold = parse_dataset_name(dataset_name)
        
#         # Create wandb config
#         chemnet_tox_morgan_config = {
#             'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
#             'wandb_project': 'MIT-Lincoln-Lab',
#             'wandb_name': f"cond_enc_class_filtered_df6_{dataset_name}",
#             'gpu': True,
#             'encoder_type': "Conditional Encoder ChemNet + Toxicity Classification + Morgan + Filtered Morgan + Group + CE_clean",
#             'batch_size': batch_size,
#             'output_size': output_size,
#             'num_layers': num_layers,
#             'learning_rate': lr,
#             'epochs': epochs,
#             'lambda1': lambda1,
#             'lambda2': lambda2,
#             'lambda3': lambda3,
#             'lambda4': lambda4,
#             'Bin': bin_size,
#             'Threshold': threshold,
#             'super_test_removed': True,
#         }

#         # ==================== TRAIN MODEL ==================== #
#         print("Training model...")
#         trained_cond_encoder = fd.train_model_condenc_1234e1e2_class(
#             model=cond_encoder_current,
#             train_data=train_loader,
#             val_data=val_loader,
#             epochs=epochs,
#             learning_rate=lr,
#             criterion1=criterion1,
#             criterion2=criterion2,
#             criterion3=criterion3,
#             criterion4=criterion4,
#             lambda1=lambda1,
#             lambda2=lambda2,
#             lambda3=lambda3,
#             lambda4=lambda4,
#             device=device,
#             config=chemnet_tox_morgan_config
#         )
        
#         # ==================== EVALUATE ON FULL VALIDATION SET ==================== #
#         print("Evaluating on full validation set...")
#         # Prepare full filtered dataset
#         filtered_dataset_full = filtered_dataset.copy()
#         filtered_dataset_full['index'] = range(len(filtered_dataset_full))
#         filtered_dataset_full_processed = fd.add_response_and_log_response(filtered_dataset_full.copy(), df6_subset, smiles_col='SMILES_spectra')
#         filtered_dataset_full_processed = fd.add_epa_levels(filtered_dataset_full_processed)
        
#         # Create tensors for full validation set
#         x_full_val_with_ext, y_full_val_emb, y_full_val_tox_class, y_full_val_morgan, y_full_val_filtered_morgan, full_val_indices_tensor = fd.create_dataset_tensors_condenc_1234e1e2_class(
#             filtered_dataset_full_processed, name_smiles_embedding_df, morgan_df, filtered_morgan_df, device, start_idx=1, stop_idx=-10)
        
#         # Generate predictions on full validation set
#         cond_encoder_current.eval()
#         with torch.no_grad():
#             full_val_predictions = cond_encoder_current(x_full_val_with_ext).cpu().numpy()

#         # Create output DataFrame for full validation set
#         emb_cols = [f'cond_emb_{j}' for j in range(512)]
#         morgan_cols = [f'cond_morgan_{j}' for j in range(regular_morgan_bits)]
#         filtered_morgan_cols = [f'cond_filtered_morgan_{j}' for j in range(filtered_morgan_bits)]
        
#         full_val_output_df = pd.DataFrame(full_val_predictions[:, :512], columns=emb_cols)
        
#         # Add toxicity classification predictions (4 logits)
#         tox_logits_cols = [f'cond_tox_logit_{j}' for j in range(4)]
#         full_val_output_df[tox_logits_cols] = full_val_predictions[:, 512:516]
        
#         # Add predicted class (argmax of logits)
#         full_val_output_df['cond_tox_pred_class'] = np.argmax(full_val_predictions[:, 512:516], axis=1)
        
#         # Add Morgan fingerprint predictions
#         morgan_start_idx = 516
#         morgan_end_idx = morgan_start_idx + regular_morgan_bits
#         full_val_morgan_pred_df = pd.DataFrame(full_val_predictions[:, morgan_start_idx:morgan_end_idx], columns=morgan_cols)
#         full_val_output_df = pd.concat([full_val_output_df, full_val_morgan_pred_df], axis=1)
        
#         # Add Filtered Morgan fingerprint predictions
#         filtered_morgan_start_idx = morgan_end_idx
#         full_val_filtered_morgan_pred_df = pd.DataFrame(full_val_predictions[:, filtered_morgan_start_idx:], columns=filtered_morgan_cols)
#         full_val_output_df = pd.concat([full_val_output_df, full_val_filtered_morgan_pred_df], axis=1)
        
#         # Add metadata
#         full_val_output_df['SMILES_spectra'] = filtered_dataset_full_processed['SMILES_spectra'].values
#         full_val_output_df['Response'] = filtered_dataset_full_processed['Response'].values
#         full_val_output_df['log_response'] = filtered_dataset_full_processed['log_response'].values
#         full_val_output_df['index_id'] = filtered_dataset_full_processed['index'].values
        
#         # Save full validation set predictions
#         if 'thresh_zero' in dataset_name:
#             bin_part = dataset_name.split('_thresh_zero')[0]
#             threshold_part = "thresh_zero"
#         else:
#             parts = dataset_name.split('_thresh')
#             bin_part = parts[0]
#             thresh_part = parts[1].split('_df_spectra')[0]
#             threshold_part = f"thresh{thresh_part}"
        
#         full_val_output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/cond_enc_1234e1e2_classification_df6"
#         os.makedirs(full_val_output_folder, exist_ok=True)

#         full_val_predictions_filename = f"cond_enc_{bin_part}_{threshold_part}_df_spectra.parquet"
#         full_val_predictions_path = os.path.join(full_val_output_folder, full_val_predictions_filename)
        
#         try:
#             full_val_output_df.to_parquet(full_val_predictions_path, index=False)
#             print(f"✓ Successfully saved full validation predictions to {full_val_predictions_filename}")
#         except Exception as save_error:
#             print(f"✗ ERROR saving full validation predictions: {str(save_error)}")
        
#         # ==================== EVALUATE ON SUPER TEST SET ==================== #
#         print("Evaluating on super test set...")
#         # Extract super test set from original dataset
#         super_test_df = dataset[dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
        
#         if len(super_test_df) > 0:
#             print(f"Super test set size: {len(super_test_df)} samples")
            
#             # Add Group and CE_clean columns to super test set
#             if 'Group' not in super_test_df.columns:
#                 super_test_df['Group'] = super_test_df['index_id'].map(id_to_group).fillna('Unknown')
            
#             if 'CE_clean' not in super_test_df.columns:
#                 super_test_df['CE_clean'] = super_test_df['index_id'].map(id_to_ce_clean).fillna('Unknown')
            
#             # Process super test set
#             super_test_df['index'] = range(len(super_test_df))
#             super_test_processed = fd.add_response_and_log_response(super_test_df.copy(), df6_subset, smiles_col='SMILES_spectra')
#             super_test_processed = fd.add_epa_levels(super_test_processed)
            
#             # Create tensors for super test set
#             x_super_test_with_ext, y_super_test_emb, y_super_test_tox_class, y_super_test_morgan, y_super_test_filtered_morgan, super_test_indices_tensor = fd.create_dataset_tensors_condenc_1234e1e2_class(
#                 super_test_processed, name_smiles_embedding_df, morgan_df, filtered_morgan_df, device, start_idx=1, stop_idx=-10)
            
#             # Generate predictions on super test set
#             with torch.no_grad():
#                 super_test_predictions = cond_encoder_current(x_super_test_with_ext).cpu().numpy()

#             # Create super test output DataFrame
#             super_test_output_df = pd.DataFrame(super_test_predictions[:, :512], columns=emb_cols)
            
#             # Add toxicity classification predictions (4 logits)
#             super_test_output_df[tox_logits_cols] = super_test_predictions[:, 512:516]
            
#             # Add predicted class (argmax of logits)
#             super_test_output_df['cond_tox_pred_class'] = np.argmax(super_test_predictions[:, 512:516], axis=1)
            
#             # Add Morgan fingerprint predictions
#             super_test_morgan_pred_df = pd.DataFrame(super_test_predictions[:, morgan_start_idx:morgan_end_idx], columns=morgan_cols)
#             super_test_output_df = pd.concat([super_test_output_df, super_test_morgan_pred_df], axis=1)
            
#             # Add Filtered Morgan fingerprint predictions
#             super_test_filtered_morgan_pred_df = pd.DataFrame(super_test_predictions[:, filtered_morgan_start_idx:], columns=filtered_morgan_cols)
#             super_test_output_df = pd.concat([super_test_output_df, super_test_filtered_morgan_pred_df], axis=1)
            
#             # Add metadata
#             super_test_output_df['SMILES_spectra'] = super_test_processed['SMILES_spectra'].values
#             super_test_output_df['Response'] = super_test_processed['Response'].values
#             super_test_output_df['log_response'] = super_test_processed['log_response'].values
#             super_test_output_df['index_id'] = super_test_processed['index'].values
            
#             # Save super test set predictions
#             super_test_output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/cond_enc_1234e1e2_classification_df6_super_test"
#             os.makedirs(super_test_output_folder, exist_ok=True)

#             super_test_predictions_filename = f"super_test_cond_enc_{bin_part}_{threshold_part}_df_spectra.parquet"
#             super_test_predictions_path = os.path.join(super_test_output_folder, super_test_predictions_filename)
            
#             try:
#                 super_test_output_df.to_parquet(super_test_predictions_path, index=False)
#                 print(f"✓ Successfully saved super test predictions to {super_test_predictions_filename}")
#             except Exception as save_error:
#                 print(f"✗ ERROR saving super test predictions: {str(save_error)}")
            
#             print(f"Super test evaluation completed for {len(super_test_df)} samples")
#         else:
#             print("No super test samples found in this dataset")
            
#         print(f"Completed processing {dataset_name}")

#         # Clear GPU memory after each dataset
#         del x_train_with_ext, x_val_with_ext, y_train_emb, y_train_tox_class, y_train_morgan, y_train_filtered_morgan
#         del y_val_emb, y_val_tox_class, y_val_morgan, y_val_filtered_morgan, train_indices_tensor, val_indices_tensor
#         del x_full_val_with_ext, y_full_val_emb, y_full_val_tox_class, y_full_val_morgan, y_full_val_filtered_morgan, full_val_indices_tensor
#         if 'x_super_test_with_ext' in locals():
#             del x_super_test_with_ext, y_super_test_emb, y_super_test_tox_class, y_super_test_morgan, y_super_test_filtered_morgan, super_test_indices_tensor
#         del cond_encoder_current, trained_cond_encoder
#         torch.cuda.empty_cache()
        
#     except Exception as e:
#         print(f"Error processing {dataset_name}: {str(e)}")
#         torch.cuda.empty_cache()
#         continue

# print(f"\n=== CONDITIONAL ENCODER CLASSIFICATION TRAINING AND EVALUATION COMPLETED ===")
# print(f"Successfully processed datasets")