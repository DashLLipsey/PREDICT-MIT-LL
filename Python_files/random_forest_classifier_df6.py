# Basic Package Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
import pickle

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

# Storage for random forest results
rf_results = [] 

# Model parameters
n_estimators = 100
random_state = 42
max_depth = None
min_samples_split = 2
min_samples_leaf = 1

# RANDOM FOREST CLASSIFICATION TRAINING LOOP - Process all grid search datasets
print("=== RANDOM FOREST EPA CLASSIFICATION TRAINING ===")
print(f"Super test SMILES to remove from training: {len(super_test_smiles)}")

# Load the original dataset for response mapping
df6_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet")
df6_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet")

# Define folders
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes_df6"

# Get all dataset files from the grid search folder
dataset_files = [f for f in os.listdir(grid_search_folder) if f.endswith('.parquet') and 'df_spectra' in f]

# Allow only range of interest as given by Rod/Sasha
allowed_threshold_suffixes = ['thresh_zero', 'thresh0_01', 'thresh0_05', 'thresh0_1']
allowed_bin_prefixes = ['bin100_', 'bin200_', 'bin500_']

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
        
        # Add index column
        train_data['index'] = range(len(train_data))
        test_data['index'] = range(len(test_data))
        
        # Process datasets - add response and EPA levels
        train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
        train_data_processed = fd.add_epa_levels(train_data_processed)
        
        test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')
        test_data_processed = fd.add_epa_levels(test_data_processed)

        # Prepare features and targets for Random Forest
        # Extract spectra features (columns 1 to -10, excluding metadata columns)
        feature_cols = train_data_processed.columns[1:-10]  # Spectra features
        X_train = train_data_processed[feature_cols].values
        y_train = train_data_processed['EPA_level'].values
        
        X_test = test_data_processed[feature_cols].values
        y_test = test_data_processed['EPA_level'].values

        print(f"Training set shape: {X_train.shape}, Target shape: {y_train.shape}")
        print(f"Test set shape: {X_test.shape}, Target shape: {y_test.shape}")

        # Parse dataset parameters
        bin_size, threshold = parse_dataset_name(dataset_name)
        
        # ==================== TRAIN RANDOM FOREST MODEL ==================== #
        print("Training Random Forest model...")
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1  # Use all available cores
        )
        
        rf_model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_test_pred = rf_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_macro_f1 = f1_score(y_test, y_test_pred, average='macro')
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Macro F1: {test_macro_f1:.4f}")
        
        # ==================== EVALUATE ON FULL VALIDATION SET ==================== #
        print("Evaluating on full validation set...")
        # Prepare full filtered dataset
        filtered_dataset_full = filtered_dataset.copy()
        filtered_dataset_full['index'] = range(len(filtered_dataset_full))
        filtered_dataset_full_processed = fd.add_response_and_log_response(filtered_dataset_full.copy(), df6_subset, smiles_col='SMILES_spectra')
        filtered_dataset_full_processed = fd.add_epa_levels(filtered_dataset_full_processed)
        
        # Extract features for full validation set
        X_full_val = filtered_dataset_full_processed[feature_cols].values
        y_full_val = filtered_dataset_full_processed['EPA_level'].values
        
        # Generate predictions on full validation set
        full_val_predictions = rf_model.predict(X_full_val)
        full_val_probabilities = rf_model.predict_proba(X_full_val)
        
        # Calculate metrics for full validation set
        full_val_accuracy = accuracy_score(y_full_val, full_val_predictions)
        full_val_macro_f1 = f1_score(y_full_val, full_val_predictions, average='macro')
        
        print(f"Full Validation Accuracy: {full_val_accuracy:.4f}")
        print(f"Full Validation Macro F1: {full_val_macro_f1:.4f}")

        # Create output DataFrame for full validation set
        full_val_output_df = pd.DataFrame()
        
        # Add predictions and probabilities
        full_val_output_df['rf_predicted_epa_class'] = full_val_predictions
        
        # Add class probabilities
        class_labels = rf_model.classes_
        for i, class_label in enumerate(class_labels):
            full_val_output_df[f'rf_prob_class_{class_label}'] = full_val_probabilities[:, i]
        
        # Add metadata
        full_val_output_df['SMILES_spectra'] = filtered_dataset_full_processed['SMILES_spectra'].values
        full_val_output_df['Response'] = filtered_dataset_full_processed['Response'].values
        full_val_output_df['log_response'] = filtered_dataset_full_processed['log_response'].values
        full_val_output_df['EPA_level'] = filtered_dataset_full_processed['EPA_level'].values
        full_val_output_df['index_id'] = filtered_dataset_full_processed['index'].values
        
        # Add performance metrics
        full_val_output_df['accuracy'] = full_val_accuracy
        full_val_output_df['macro_f1'] = full_val_macro_f1
        full_val_output_df['bin_size'] = bin_size
        full_val_output_df['threshold'] = threshold
        
        # Save full validation set predictions
        if 'thresh_zero' in dataset_name:
            bin_part = dataset_name.split('_thresh_zero')[0]
            threshold_part = "thresh_zero"
        else:
            parts = dataset_name.split('_thresh')
            bin_part = parts[0]
            thresh_part = parts[1].split('_df_spectra')[0]
            threshold_part = f"thresh{thresh_part}"
        
        full_val_output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/random_forest_df6"
        os.makedirs(full_val_output_folder, exist_ok=True)

        full_val_predictions_filename = f"rf_{bin_part}_{threshold_part}_df_spectra.parquet"
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
            
            # Extract features for super test set
            X_super_test = super_test_processed[feature_cols].values
            y_super_test = super_test_processed['EPA_level'].values
            
            # Generate predictions on super test set
            super_test_predictions = rf_model.predict(X_super_test)
            super_test_probabilities = rf_model.predict_proba(X_super_test)
            
            # Calculate metrics for super test set
            super_test_accuracy = accuracy_score(y_super_test, super_test_predictions)
            super_test_macro_f1 = f1_score(y_super_test, super_test_predictions, average='macro')
            
            print(f"Super Test Accuracy: {super_test_accuracy:.4f}")
            print(f"Super Test Macro F1: {super_test_macro_f1:.4f}")

            # Create super test output DataFrame
            super_test_output_df = pd.DataFrame()
            
            # Add predictions and probabilities
            super_test_output_df['rf_predicted_epa_class'] = super_test_predictions
            
            # Add class probabilities
            for i, class_label in enumerate(class_labels):
                super_test_output_df[f'rf_prob_class_{class_label}'] = super_test_probabilities[:, i]
            
            # Add metadata
            super_test_output_df['SMILES_spectra'] = super_test_processed['SMILES_spectra'].values
            super_test_output_df['Response'] = super_test_processed['Response'].values
            super_test_output_df['log_response'] = super_test_processed['log_response'].values
            super_test_output_df['EPA_level'] = super_test_processed['EPA_level'].values
            super_test_output_df['index_id'] = super_test_processed['index'].values
            
            # Add performance metrics
            super_test_output_df['accuracy'] = super_test_accuracy
            super_test_output_df['macro_f1'] = super_test_macro_f1
            super_test_output_df['bin_size'] = bin_size
            super_test_output_df['threshold'] = threshold
            
            # Save super test set predictions
            super_test_output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/random_forest_df6_super_test"
            os.makedirs(super_test_output_folder, exist_ok=True)

            super_test_predictions_filename = f"super_test_rf_{bin_part}_{threshold_part}_df_spectra.parquet"
            super_test_predictions_path = os.path.join(super_test_output_folder, super_test_predictions_filename)
            
            try:
                super_test_output_df.to_parquet(super_test_predictions_path, index=False)
                print(f"✓ Successfully saved super test predictions to {super_test_predictions_filename}")
            except Exception as save_error:
                print(f"✗ ERROR saving super test predictions: {str(save_error)}")
            
            print(f"Super test evaluation completed for {len(super_test_df)} samples")
        else:
            print("No super test samples found in this dataset")
            
        # Store results for summary
        rf_results.append({
            'dataset_name': dataset_name,
            'bin_size': bin_size,
            'threshold': threshold,
            'test_accuracy': test_accuracy,
            'test_macro_f1': test_macro_f1,
            'full_val_accuracy': full_val_accuracy,
            'full_val_macro_f1': full_val_macro_f1,
            'super_test_accuracy': super_test_accuracy if len(super_test_df) > 0 else None,
            'super_test_macro_f1': super_test_macro_f1 if len(super_test_df) > 0 else None,
            'super_test_samples': len(super_test_df) if len(super_test_df) > 0 else 0
        })
            
        print(f"Completed processing {dataset_name}")
        
        # Clear memory after each dataset
        del X_train, X_test, y_train, y_test, X_full_val, y_full_val
        if 'X_super_test' in locals():
            del X_super_test, y_super_test
        del rf_model
        gc.collect()
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        gc.collect()
        continue

print(f"\n=== RANDOM FOREST CLASSIFICATION TRAINING AND EVALUATION COMPLETED ===")
print(f"Successfully processed {len(rf_results)} datasets")

# Save summary results
if rf_results:
    results_df = pd.DataFrame(rf_results)
    results_summary_path = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/random_forest_df6/rf_results_summary.parquet"
    results_df.to_parquet(results_summary_path, index=False)
    print(f"✓ Saved results summary to {results_summary_path}")
    
    # Print summary statistics
    print("\n=== RESULTS SUMMARY ===")
    print(f"Average Test Macro F1: {results_df['test_macro_f1'].mean():.4f}")
    print(f"Average Full Validation Macro F1: {results_df['full_val_macro_f1'].mean():.4f}")
    if results_df['super_test_macro_f1'].notna().any():
        print(f"Average Super Test Macro F1: {results_df['super_test_macro_f1'].mean():.4f}")