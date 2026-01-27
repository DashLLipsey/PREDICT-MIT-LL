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

# Storage for results
direct_toxicity_results = [] 

# Model parameters
num_classes = 4
num_layers = 8
batch_size = 256
epochs = 250
lr = 0.0001

# Loss function for toxicity classification
criterion = CrossEntropyLoss()

# DIRECT TOXICITY PREDICTION TRAINING LOOP - Process all grid search datasets
print("=== DIRECT TOXICITY PREDICTION (Spectra + Group + CE_clean -> Toxicity) ===")
print(f"Super test SMILES to remove from training: {len(super_test_smiles)}")

# Set up device
device = fd.set_up_gpu()

# Load the original dataset for response mapping
df6_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet")
df6_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet")

# Define folders
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes_df6"

# Get all dataset files from the grid search folder
dataset_files = [f for f in os.listdir(grid_search_folder) if f.endswith('.parquet') and 'df_spectra' in f]

# # Filter for allowed bin sizes and thresholds
# allowed_bin_prefixes = ['bin1_'] 
# allowed_threshold_suffixes = ['thresh0_05']

# Full set of bin and threshold values
allowed_bin_prefixes = ['bin0_1_', 'bin0_5_', 'bin1_', 'bin2_', 'bin5_', 'bin10_',
                        'bin25_', 'bin50_', 'bin100_', 'bin200_', 'bin500_'] # 'bin0_05' 
allowed_threshold_suffixes = ['thresh_zero', 'thresh0_001', 'thresh0_005', 'thresh0_01', 'thresh0_05', 
                             'thresh0_1', 'thresh0_5', 'thresh1', 'thresh2', 'thresh5', 'thresh10', 
                             'thresh50', 'thresh100']

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

# Loop through each dataset
for i, dataset_name in enumerate(sorted(dataset_names), 1):
    print(f"\n{'='*80}")
    print(f"Processing {i}/{len(dataset_names)}: {dataset_name}")
    print(f"{'='*80}")
    
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

        # # Inspect columns that will be used for tensor creation
        # tensor_columns = train_data_processed.columns[1:-10]
        # print(f"\nColumns to be used in tensor (start_idx=1, stop_idx=-10):")
        # print(f"Total: {len(tensor_columns)} columns")
        # print(f"First 10: {list(tensor_columns[:10])}")
        # print(f"Last 10: {list(tensor_columns[-10:])}")
        
        # Create tensors for training
        print("\nCreating training tensors for direct toxicity prediction...")
        x_train, y_train_tox, train_indices_tensor = fd.create_dataset_tensors_direct_toxicity_e1e2(
            train_data_processed, device, start_idx=1, stop_idx=-10)
        
        print(f"Tensor shape created: {x_train.shape}")

        x_val, y_val_tox, val_indices_tensor = fd.create_dataset_tensors_direct_toxicity_e1e2(
            test_data_processed, device, start_idx=1, stop_idx=-10)

        # Create model
        actual_input_size = x_train.shape[1]
        print(f"Creating direct toxicity model with input size: {actual_input_size}")

        # Direct toxicity encoder that uses external conditions
        direct_tox_model = fd.Direct_Toxicity_Encoder(
            input_size=actual_input_size,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout_rate=0.2
        ).to(device)
        
        # Create DataLoaders for training
        train_dataset = TensorDataset(x_train, y_train_tox, train_indices_tensor)
        val_dataset = TensorDataset(x_val, y_val_tox, val_indices_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
                
        # Parse dataset parameters for wandb config
        bin_size, threshold = parse_dataset_name(dataset_name)
        
        # Create wandb config
        direct_tox_config = {
            'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
            'wandb_project': 'MIT-Lincoln-Lab',
            'wandb_name': f"direct_toxicity_{dataset_name}",
            'gpu': True,
            'model_type': "Direct Toxicity Prediction (Spectra + Group + CE_clean -> Toxicity)",
            'batch_size': batch_size,
            'num_classes': num_classes,
            'num_layers': num_layers,
            'learning_rate': lr,
            'epochs': epochs,
            'Bin': bin_size,
            'Threshold': threshold,
            'super_test_removed': True,
        }

        # ==================== TRAIN MODEL ==================== #
        print("Training direct toxicity model...")
        # Direct prediction training function
        trained_direct_tox_model, train_losses, val_losses, train_accs, val_accs = fd.train_direct_toxicity_encoder_e1e2(
            model=direct_tox_model,
            train_data=train_loader,
            val_data=val_loader,
            epochs=epochs,
            learning_rate=lr,
            criterion=criterion,
            device=device,
            config=direct_tox_config
        )

        # ==================== EVALUATE ON FULL VALIDATION SET ==================== #
        print(f"\n{'='*80}")
        print("Evaluating on Full Validation Set")
        print(f"{'='*80}")
        
        # Prepare full filtered dataset
        filtered_dataset_full = filtered_dataset.copy()
        
        # Create train indicator mapping based on original index
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
        
        # Create a copy for tensor creation
        filtered_dataset_for_tensors = filtered_dataset_full_processed.drop(columns=['original_index', 'train']).copy()

        # Create tensors for full validation set
        x_full_val, y_full_val_tox, full_val_indices_tensor = fd.create_dataset_tensors_direct_toxicity_e1e2(
            filtered_dataset_for_tensors, device, start_idx=1, stop_idx=-10)
        
        # Generate predictions on full validation set
        direct_tox_model.eval()
        with torch.no_grad():
            full_val_tox_logits = direct_tox_model(x_full_val).cpu().numpy()
        
        # Create output DataFrame for full validation set
        tox_logits_cols = [f'direct_tox_logit_{j}' for j in range(num_classes)]
        full_val_output_df = pd.DataFrame(full_val_tox_logits, columns=tox_logits_cols)
        
        # Add predicted class (argmax of logits)
        full_val_output_df['direct_tox_pred_class'] = np.argmax(full_val_tox_logits, axis=1)
        
        # Add true toxicity class
        full_val_output_df['true_tox_class'] = y_full_val_tox.cpu().numpy()
        
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
        
        # Calculate and print accuracy for full validation set
        correct = (full_val_output_df['direct_tox_pred_class'] == full_val_output_df['true_tox_class']).sum()
        total = len(full_val_output_df)
        accuracy = 100 * correct / total
        print(f"Full validation set accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        # Save full validation set predictions
        if 'thresh_zero' in dataset_name:
            bin_part = dataset_name.split('_thresh_zero')[0]
            threshold_part = "thresh_zero"
        else:
            parts = dataset_name.split('_thresh')
            bin_part = parts[0]
            thresh_part = parts[1].split('_df_spectra')[0]
            threshold_part = f"thresh{thresh_part}"
        
        # Direct encoder predictions output folder
        full_val_output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/regular_classifier_df6"
        os.makedirs(full_val_output_folder, exist_ok=True)

        full_val_predictions_filename = f"direct_tox_{bin_part}_{threshold_part}_df_spectra.parquet"
        full_val_predictions_path = os.path.join(full_val_output_folder, full_val_predictions_filename)
        
        try:
            full_val_output_df.to_parquet(full_val_predictions_path, index=False)
            print(f"✓ Successfully saved full validation predictions to {full_val_predictions_filename}")
        except Exception as save_error:
            print(f"✗ ERROR saving full validation predictions: {str(save_error)}")
        
        # ==================== EVALUATE ON SUPER TEST SET ==================== #
        print(f"\n{'='*80}")
        print("Evaluating on Super Test Set")
        print(f"{'='*80}")
        
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
            x_super_test, y_super_test_tox, super_test_indices_tensor = fd.create_dataset_tensors_direct_toxicity_e1e2(
                super_test_processed, device, start_idx=1, stop_idx=-10)
            
            # Generate predictions on super test set
            with torch.no_grad():
                super_test_tox_logits = direct_tox_model(x_super_test).cpu().numpy()

            # Create super test output DataFrame
            super_test_output_df = pd.DataFrame(super_test_tox_logits, columns=tox_logits_cols)
            
            # Add predicted class (argmax of logits)
            super_test_output_df['direct_tox_pred_class'] = np.argmax(super_test_tox_logits, axis=1)
            
            # Add true toxicity class
            super_test_output_df['true_tox_class'] = y_super_test_tox.cpu().numpy()
            
            # Add metadata
            super_test_output_df['SMILES_spectra'] = super_test_processed['SMILES_spectra'].values
            super_test_output_df['Response'] = super_test_processed['Response'].values
            super_test_output_df['log_response'] = super_test_processed['log_response'].values
            super_test_output_df['index_id'] = super_test_processed['index'].values
            super_test_output_df['train'] = 0  # Super test samples were not in training set
            
            # Calculate and print accuracy for super test set
            correct_super = (super_test_output_df['direct_tox_pred_class'] == super_test_output_df['true_tox_class']).sum()
            total_super = len(super_test_output_df)
            accuracy_super = 100 * correct_super / total_super
            print(f"Super test set accuracy: {accuracy_super:.2f}% ({correct_super}/{total_super})")
            
            # Save super test set predictions
            # Direct encoder predictions output folder
            super_test_output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/regular_classifier_df6_super_test"
            os.makedirs(super_test_output_folder, exist_ok=True)

            super_test_predictions_filename = f"super_test_direct_tox_{bin_part}_{threshold_part}_df_spectra.parquet"
            super_test_predictions_path = os.path.join(super_test_output_folder, super_test_predictions_filename)
            
            try:
                super_test_output_df.to_parquet(super_test_predictions_path, index=False)
                print(f"✓ Successfully saved super test predictions to {super_test_predictions_filename}")
            except Exception as save_error:
                print(f"✗ ERROR saving super test predictions: {str(save_error)}")
            
            print(f"Super test evaluation completed for {len(super_test_df)} samples")
        else:
            print("No super test samples found in this dataset")
            
        print(f"\nCompleted processing {dataset_name}")

        # Clear GPU memory after each dataset
        del x_train, x_val, y_train_tox, y_val_tox, train_indices_tensor, val_indices_tensor
        del x_full_val, y_full_val_tox, full_val_indices_tensor
        if 'x_super_test' in locals():
            del x_super_test, y_super_test_tox, super_test_indices_tensor
        del direct_tox_model, trained_direct_tox_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ ERROR processing {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        continue

print(f"\n{'='*80}")
print("=== DIRECT TOXICITY PREDICTION TRAINING AND EVALUATION COMPLETED ===")
print(f"{'='*80}")
print(f"Successfully processed datasets")