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

# Sklearn imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

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

# New Super Test SMILES list
super_test_smiles = [
    'CCOP(=S)(OCC)Oc1ccc([N+](=O)[O-])cc1',
    'CCOP(=S)(OCC)Oc1ccc(S(C)=O)cc1',
    'CC1(C)O[C@@H]2C[C@H]3[C@@H]4C[C@H](F)C5=CC(=O)C=C[C@]5(C)[C@H]4[C@@H](O)C[C@]3(C)[C@]2(C(=O)CO)O1 CC(=O)OC1(C)CC(C)C(=O)C(C(O)CC2CC(=O)NC(=O)C2)C1',
    'NC(=S)Nc1ccccc1',
    'CC(=O)OC[C@]12C[C@H](OC(=O)CC(C)C)C(C)=C[C@H]1O[C@@H]1[C@H](O)[C@@H](OC(C)=O)[C@@]2(C)[C@]12CO2',
    # Optional (Level 0 that would get filtered out)
    'CC(C)OC(=O)CCCC=CCC1C(O)CC(O)C1CCC(O)CCc1ccccc1',
    'Cc1cc(C(C)(C)C)c(O)c(C)c1CC1=NCCN1.Cl',
    'CCOP(=O)(OCC)Oc1ccc([N+](=O)[O-])cc1',
    'CCN1CC2(COC)C(OC(C)=O)CC(OC)C34C5CC6(O)C(OC)C(O)C(OC(C)=O)(C5C6OC(=O)c5ccccc5)C(C(OC)C23)C14',
    # Level 1
    'CCOP(=O)(OCC)OC(=CCl)c1ccc(Cl)cc1Cl',
    'CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1',
    'CNC(=O)Oc1ccccc1C1OCCO1',
    'O=C1C=C2C(=CCOC2O)O1',
    'CC(=O)C1=C(O)[C@@H]2[C@H]3c4c[nH]c5cccc(c45)C[C@H]3C(C)(C)N2C1=O',
    'Cc1cc(OC(=O)N(C)C)nn1C(=O)N(C)C',
    'C[C@@H]1Cc2c(Cl)cc(C(=O)N[C@@H](Cc3ccccc3)C(=O)O)c(O)c2C(=O)O1',
    'CNC(=O)Oc1cccc2c1OC(C)(C)O2',
    'CC(N)Cc1ccccc1',
    'CC1OC(OC2C(O)CC(OC3C(O)CC(OC4CCC5(C)C(CCC6C5CCC5(C)C(C7=CC(=O)OC7)CCC65O)C4)OC3C)OC2C)CC(O)C1O',
    # Optional (Level 1 that would get filtered out)
    'CC(=O)C1(O)Cc2c(O)c3c(c(O)c2C(OC2CC(N)C(O)C(C)O2)C1)C(=O)c1ccccc1C3=O',
    'CN1C(C(=O)Nc2ccccn2)=C(O)c2sc(Cl)cc2S1(=O)=O',
    'C=C1CCC(O)CC1=CC=C1CCCC2(C)C1CCC2C(C)C=CC(C)C(C)C',
    'CC(=O)OCC(=O)[C@@]12OC(C)(C)O[C@@H]1C[C@H]1[C@@H]3C[C@H](F)C4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@@]12C',
    'C[C@H]1O[C@@H](O[C@H]2[C@@H](O)C[C@H](O[C@H]3[C@@H](O)C[C@H](O[C@H]4CC[C@@]5(C)[C@H](CC[C@@H]6[C@@H]5CC[C@]5(C)[C@@H](C7=CC(=O)OC7)CC[C@]65O)C4)O[C@@H]3C)O[C@@H]2C)C[C@H](O)[C@@H]1O',
    'COP(=S)(OC)Oc1ccc(S(=O)(=O)N(C)C)cc1',
    # Level 2
    'COP(=S)(OC)SCN1C(=O)c2ccccc2C1=O',
    'CCOC(=O)C1(c2ccccc2)CCN(C)CC1',
    'CCOP(=S)(OCC)Oc1ccc2c(C)c(Cl)c(=O)oc2c1',
    'CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1',
    'S=c1[nH]c2ccccc2s1',
    'CC(=O)N1CCN(c2ccc(OC[C@H]3CO[C@](Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1',
    'CN(N=O)c1ccccc1',
    'CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21',
    'COc1cc2ccc(=O)oc2cc1OC',
    'CN1CCCC(n2nc(Cc3ccc(Cl)cc3)c3ccccc3c2=O)CC1',
    'CNC[C@H](O)c1cccc(O)c1',
    'C1ccc2ncccc2c1',
    'CN1C(=O)CN=C(c2ccccc2F)c2cc([N+](=O)[O-])ccc21',
    'Cn1cnc([N+](=O)[O-])c1Sc1ncnc2nc[nH]c12',
    # Level 3
    'C=CCOC(Cn1ccnc1)c1ccc(Cl)cc1Cl',
    'COc1ccnc(CS(=O)c2nc3ccc(OC(F)F)cc3[nH]2)c1OC',
    'CCC(=O)Nc1ccc(Cl)c(Cl)c1',
    'C1CCN2C[C@@H]3C[C@@H](CN4CCCC[C@@H]34)[C@H]2C1',
    'CC(=O)CCCCn1c(=O)c2c(ncn2C)n(C)c1=O',
    'COc1ccc(N)cc1',
    'Cc1ccc(C(C)C)cc2c(C)ccc1-2',
    'Clc1ccc(C2(Cn3cncn3)CC(Br)CO2)c(Cl)c1',
    'CC(CCc1ccccc1)NCC(O)c1ccc(O)c(C(N)=O)c1',
    'CC1COC(Cn2cncn2)(c2ccc(Oc3ccc(Cl)cc3)cc2Cl)O1',
    'Cc1ccc(S(N)(=O)=O)cc1',
    'Cc1cc(=O)nc(C(C)C)[nH]1',
    'N[C@@H](CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F',
    'COc1cc(C=CC(=O)CC(=O)C=Cc2ccc(O)c(OC)c2)ccc1O',
    'Cc1cc(C)nc(Nc2ccccc2)n1',
    'COC(=O)Nc1nc2ccccc2[nH]1',
    'CCOC(=O)NCCOc1ccc(Oc2ccccc2)cc1',
    'COc1cc2ccc(=O)oc2cc1O',
    # Level 4
    'Cc1ncc(COP(=O)(O)O)c(C=O)c1O',
    'OCCN(CCO)CCO',
    'O=C(O)c1cccnc1',
    'C[C@@H]1CC[C@@]2(OC1)O[C@H]1C[C@H]3[C@@H]4CC=C5C[C@@H](O)CC[C@]5(C)[C@H]4CC[C@]3(C)[C@H]1[C@@H]2C',
    'Oc1cc(O)c2c(c1)O[C@H](c1ccc(O)c(O)c1)[C@@H](O)C2',
    'O=c1[nH]c2c(c(=O)n1C1CCCCC1)CCC2',
    'NC(CCC(=O)O)C(=O)O',
    'N[C@@H](Cc1cnc[nH]1)C(=O)O',
    'COc1ccc(Cl)cc1C(=O)NCCc1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1',
    'CCCCC(CC)COC(=O)c1ccccc1C(=O)OCC(CC)CCCC',
    'CCCCOC(=O)CC(CC(=O)OCCCC)(OC(C)=O)C(=O)OCCC',
    'c1ccc(Nc2ccc3ccccc3c2)cc1'
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


print("="*80)
print("SECTION 1: ENCODER + RANDOM FOREST PREDICTOR")
print("="*80)

# Model parameters for encoder
output_size = 512  # ChemNet embeddings
num_layers = 5
batch_size = 256
epochs = 250
lr = 0.0001


# Random Forest parameters
rf_n_estimators = 100
rf_max_depth = None
rf_random_state = 42

# Loss functions
criterion = nn.MSELoss()  # ChemNet embeddings

print("=== ENCODER + RF: ChemNet Encoder + Random Forest Toxicity Predictor ===")
print(f"Super test SMILES to remove from training: {len(super_test_smiles)}")

# Set up device and load all reference datasets
device = fd.set_up_gpu()

# Step 1 embedding inputs
name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_chemnet.parquet")
# Load the original dataset for response mapping
df6_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet")
# Load spectra metadata
df6_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet")

# Define folders
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes_df6"

# Get all dataset files from the grid search folder
dataset_files = [f for f in os.listdir(grid_search_folder) if f.endswith('.parquet') and 'df_spectra' in f]

# Allowed bin sizes and thresholds
allowed_bin_prefixes = ['bin0_1_', 'bin0_5_', 'bin1_', 'bin10_', 'bin100_', 'bin500_']
allowed_threshold_suffixes = ['thresh_zero', 'thresh0_01', 'thresh0_05', 'thresh0_1', 'thresh0_5', 
                              'thresh10', 'thresh50', 'thresh100']

# Filter dataset files to only include allowed bin sizes and thresholds
dataset_files = [f for f in dataset_files if any(f.startswith(prefix) for prefix in allowed_bin_prefixes)]
dataset_files = [f for f in dataset_files if any(suffix in f for suffix in allowed_threshold_suffixes)]

dataset_names = [f.replace('.parquet', '') for f in dataset_files]

print(f"Found {len(dataset_names)} datasets to process (filtered by bin sizes and thresholds)")

# Load synthetic flag from df6_spectra for all index_ids
id_to_synthetic = dict(zip(df6_spectra['index_id'], df6_spectra['synthetic'].fillna(0)))
synthetic_index_ids = set([idx for idx, syn in id_to_synthetic.items() if syn==1])
print(f"Identified {len(synthetic_index_ids)} synthetic spectra (by index_id)")

# Loop through each dataset
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
        print(f"Removed {removed_count} samples from super test SMILES")
                
        # Apply filtering (>=3 spectra per SMILES)
        counts = dataset_no_super_test['SMILES_spectra'].value_counts()
        valid_smiles = counts[counts >= 3].index
        filtered_dataset = dataset_no_super_test[dataset_no_super_test['SMILES_spectra'].isin(valid_smiles)].copy()
        
        print(f"After filtering (>=3 spectra per SMILES): {filtered_dataset.shape}")
        
        # Separate synthetic and real spectra
        synthetic_mask = filtered_dataset['index_id'].map(lambda idx: id_to_synthetic.get(idx, 0)==1)
        real_mask = ~synthetic_mask
        synthetic_data = filtered_dataset[synthetic_mask].copy()
        real_data = filtered_dataset[real_mask].copy()
        
        print(f"Split data: {len(real_data)} real spectra, {len(synthetic_data)} synthetic spectra")

        # Train/test split for real spectra only
        smiles_groups = real_data.groupby('SMILES_spectra')
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
        
        # Add ALL synthetic to training, NOT to test
        train_indices.extend(synthetic_data.index.values)
        
        print(f"Train/test split: {len(train_indices)} train (including {len(synthetic_data)} synthetic), {len(test_indices)} test (real only)")
        
        # Preserve original index_ids
        train_index_ids = filtered_dataset.loc[train_indices, 'index_id'].values
        test_index_ids = filtered_dataset.loc[test_indices, 'index_id'].values
        
        train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
        test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)
        
        # Verify index_id preservation
        assert (train_data['index_id'].values == train_index_ids).all(), "Train index_id mismatch!"
        assert (test_data['index_id'].values == test_index_ids).all(), "Test index_id mismatch!"
        
        # Create set of training indices for later tracking
        train_indices_set = set(train_indices)
        
        # Add 'index' column using index_id
        train_data['index'] = train_data['index_id']
        test_data['index'] = test_data['index_id']
        
        # Process datasets
        train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
        test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')
        
        # Create tensors for training encoder
        # Use stop_idx=-6 to exclude both Response and log_response columns from input
        y_train_emb, x_train_with_ext, train_indices_tensor = fd.create_dataset_tensors_1(
                train_data_processed, name_smiles_embedding_df, device, start_idx=1, stop_idx=-6)

        y_val_emb, x_val_with_ext, val_indices_tensor = fd.create_dataset_tensors_1(
            test_data_processed, name_smiles_embedding_df, device, start_idx=1, stop_idx=-6)
        
        # Create model
        actual_input_size = x_train_with_ext.shape[1]
        print(f"Creating encoder with input size: {actual_input_size}")

        base_encoder_current = fd.base_Encoder(input_size=actual_input_size,
                                                output_size=output_size, 
                                                num_layers=num_layers).to(device)
        
        # Create DataLoaders for training encoder
        train_dataset = TensorDataset(x_train_with_ext, y_train_emb, train_indices_tensor)
        val_dataset = TensorDataset(x_val_with_ext, y_val_emb, val_indices_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
                
        # Parse dataset parameters for wandb config
        bin_size, threshold = parse_dataset_name(dataset_name)
        
        # Create wandb config
        chemnet_encoder_config = {
            'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
            'wandb_project': 'MIT-Lincoln-Lab',
            'wandb_name': f"base_encoder_rf_{dataset_name}",
            'gpu': True,
            'encoder_type': "Base Encoder ChemNet",
            'batch_size': batch_size,
            'output_size': output_size,
            'num_layers': num_layers,
            'learning_rate': lr,
            'epochs': epochs,
            'Bin': bin_size,
            'Threshold': threshold,
            'super_test_removed': True,
        }

        # ==================== TRAIN ENCODER ==================== #
        print("Training encoder...")
        trained_encoder, train_losses, val_losses = fd.train_model_chemnet_encoder(
            model=base_encoder_current,
            train_data=train_loader,
            val_data=val_loader,
            epochs=epochs,
            learning_rate=lr,
            criterion=criterion,
            device=device,
            config=chemnet_encoder_config
        )
        
        # ==================== GENERATE EMBEDDINGS FOR RF TRAINING ==================== #
        print("Generating embeddings for Random Forest training...")
        trained_encoder.eval()
        
        with torch.no_grad():
            train_embeddings = trained_encoder(x_train_with_ext).cpu().numpy()
            val_embeddings = trained_encoder(x_val_with_ext).cpu().numpy()
        
        # Get toxicity targets (log_response)
        y_train_tox_rf = train_data_processed['log_response'].values
        y_val_tox_rf = test_data_processed['log_response'].values
        
        # ==================== TRAIN RANDOM FOREST ==================== #
        print(f"Training Random Forest on embeddings (n_estimators={rf_n_estimators})...")
        rf_model = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=rf_random_state,
            n_jobs=-1,
            verbose=0
        )
        
        rf_model.fit(train_embeddings, y_train_tox_rf)
        
        # Evaluate on validation set
        val_predictions_rf = rf_model.predict(val_embeddings)
        val_mse = mean_squared_error(y_val_tox_rf, val_predictions_rf)
        val_r2 = r2_score(y_val_tox_rf, val_predictions_rf)
        print(f"Validation MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
        
        # ==================== EVALUATE ON FULL VALIDATION SET ==================== #
        print("Evaluating on full validation set...")
        # Prepare full filtered dataset
        filtered_dataset_full = filtered_dataset.copy()
        
        # Create train indicator mapping
        train_indicator_map = {}
        for idx in filtered_dataset_full.index:
            train_indicator_map[idx] = 1 if idx in train_indices_set else 0
        
        # Reset index and add sequential index column
        filtered_dataset_full = filtered_dataset_full.reset_index(drop=False, names=['original_index'])
        filtered_dataset_full['index'] = filtered_dataset_full['index_id']
        
        # Process dataset
        filtered_dataset_full_processed = fd.add_response_and_log_response(filtered_dataset_full.copy(), df6_subset, smiles_col='SMILES_spectra')
        
        # Add train indicator
        filtered_dataset_full_processed['train'] = filtered_dataset_full_processed['original_index'].map(train_indicator_map).fillna(0).astype(int)
        
        # Create a copy for tensor creation
        filtered_dataset_for_tensors = filtered_dataset_full_processed.drop(columns=['original_index', 'train']).copy()
        
        # Create tensors for full validation set
        y_full_val_emb, x_full_val_with_ext, full_val_indices_tensor = fd.create_dataset_tensors_1(
            filtered_dataset_for_tensors, name_smiles_embedding_df, device, start_idx=1, stop_idx=-6)
        
        # Generate embeddings
        with torch.no_grad():
            full_val_embeddings = trained_encoder(x_full_val_with_ext).cpu().numpy()
        
        # Generate predictions with RF
        full_val_tox_predictions = rf_model.predict(full_val_embeddings)

        # Create output DataFrame
        emb_cols = [f'encoder_emb_{j}' for j in range(512)]
        full_val_output_df = pd.DataFrame(full_val_embeddings, columns=emb_cols)
        full_val_output_df['rf_tox_pred'] = full_val_tox_predictions
        
        # Add metadata
        full_val_output_df['SMILES_spectra'] = filtered_dataset_full_processed['SMILES_spectra'].values
        full_val_output_df['Response'] = filtered_dataset_full_processed['Response'].values
        full_val_output_df['log_response'] = filtered_dataset_full_processed['log_response'].values
        full_val_output_df['index_id'] = filtered_dataset_full_processed['index_id'].values
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
        
        full_val_output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/chemnet_enc_rf_predictor"
        os.makedirs(full_val_output_folder, exist_ok=True)

        full_val_predictions_filename = f"enc_rf_{bin_part}_{threshold_part}_df_spectra.parquet"
        full_val_predictions_path = os.path.join(full_val_output_folder, full_val_predictions_filename)
        
        try:
            full_val_output_df.to_parquet(full_val_predictions_path, index=False)
            print(f"✓ Successfully saved full validation predictions to {full_val_predictions_filename}")
        except Exception as save_error:
            print(f"✗ ERROR saving full validation predictions: {str(save_error)}")
        
        # Save RF model
        rf_model_filename = f"rf_model_{bin_part}_{threshold_part}.pkl"
        rf_model_path = os.path.join(full_val_output_folder, rf_model_filename)
        with open(rf_model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"✓ Saved RF model to {rf_model_filename}")
        
        # ==================== EVALUATE ON SUPER TEST SET ==================== #
        print("Evaluating on super test set...")
        # Extract super test set from original dataset, then remove synthetic spectra
        super_test_df_all = dataset[dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
        super_test_df = super_test_df_all[~super_test_df_all['index_id'].isin(synthetic_index_ids)].copy()
        print(f"Super test: {len(super_test_df_all)} total spectra, {len(super_test_df)} real spectra (removed {len(super_test_df_all) - len(super_test_df)} synthetic spectra)")
        
        if len(super_test_df) > 0:
            print(f"Super test set size: {len(super_test_df)} samples")
            
            super_test_df['index'] = super_test_df['index_id']
            super_test_processed = fd.add_response_and_log_response(super_test_df.copy(), df6_subset, smiles_col='SMILES_spectra')
            
            # Create tensors for super test set
            y_super_test_emb, x_super_test_with_ext, super_test_indices_tensor = fd.create_dataset_tensors_1(
                super_test_processed, name_smiles_embedding_df, device, start_idx=1, stop_idx=-6)
            
            # Generate embeddings
            with torch.no_grad():
                super_test_embeddings = trained_encoder(x_super_test_with_ext).cpu().numpy()
            
            # Generate predictions with RF
            super_test_tox_predictions = rf_model.predict(super_test_embeddings)

            # Create super test output DataFrame
            super_test_output_df = pd.DataFrame(super_test_embeddings, columns=emb_cols)
            super_test_output_df['rf_tox_pred'] = super_test_tox_predictions
            
            # Add metadata
            super_test_output_df['SMILES_spectra'] = super_test_processed['SMILES_spectra'].values
            super_test_output_df['log_response'] = super_test_processed['log_response'].values
            super_test_output_df['Response'] = super_test_processed['Response'].values
            super_test_output_df['index_id'] = super_test_processed['index_id'].values
            super_test_output_df['train'] = 0  # Super test samples were not in training set
            
            # Save super test set predictions
            super_test_output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/chemnet_enc_rf_predictor_super_test"
            os.makedirs(super_test_output_folder, exist_ok=True)

            super_test_predictions_filename = f"super_test_enc_rf_{bin_part}_{threshold_part}_df_spectra.parquet"
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
        del x_train_with_ext, x_val_with_ext, y_train_emb, y_val_emb, train_indices_tensor, val_indices_tensor
        del x_full_val_with_ext, y_full_val_emb, full_val_indices_tensor
        if 'x_super_test_with_ext' in locals():
            del x_super_test_with_ext, y_super_test_emb, super_test_indices_tensor
        del base_encoder_current, trained_encoder, rf_model
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        continue

print(f"\n=== ENCODER + RF TRAINING AND EVALUATION COMPLETED ===")


print("\n" + "="*80)
print("SECTION 2: DIRECT RANDOM FOREST ON SPECTRA")
print("="*80)

print("=== DIRECT RF: Random Forest directly on Spectra ===")
print(f"Super test SMILES to remove from training: {len(super_test_smiles)}")

# Loop through each dataset
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
        print(f"Removed {removed_count} samples from super test SMILES")
                
        # Apply filtering (>=3 spectra per SMILES)
        counts = dataset_no_super_test['SMILES_spectra'].value_counts()
        valid_smiles = counts[counts >= 3].index
        filtered_dataset = dataset_no_super_test[dataset_no_super_test['SMILES_spectra'].isin(valid_smiles)].copy()
        
        print(f"After filtering (>=3 spectra per SMILES): {filtered_dataset.shape}")
        
        # Separate synthetic and real spectra
        synthetic_mask = filtered_dataset['index_id'].map(lambda idx: id_to_synthetic.get(idx, 0)==1)
        real_mask = ~synthetic_mask
        synthetic_data = filtered_dataset[synthetic_mask].copy()
        real_data = filtered_dataset[real_mask].copy()
        
        print(f"Split data: {len(real_data)} real spectra, {len(synthetic_data)} synthetic spectra")

        # Train/test split for real spectra only
        smiles_groups = real_data.groupby('SMILES_spectra')
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
        
        # Add ALL synthetic to training, NOT to test
        train_indices.extend(synthetic_data.index.values)
        
        print(f"Train/test split: {len(train_indices)} train (including {len(synthetic_data)} synthetic), {len(test_indices)} test (real only)")
        
        # Preserve original index_ids
        train_index_ids = filtered_dataset.loc[train_indices, 'index_id'].values
        test_index_ids = filtered_dataset.loc[test_indices, 'index_id'].values
        
        train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
        test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)
        
        # Verify index_id preservation
        assert (train_data['index_id'].values == train_index_ids).all(), "Train index_id mismatch!"
        assert (test_data['index_id'].values == test_index_ids).all(), "Test index_id mismatch!"
        
        # Create set of training indices for later tracking
        train_indices_set = set(train_indices)
        
        # Add 'index' column using index_id
        train_data['index'] = train_data['index_id']
        test_data['index'] = test_data['index_id']
        
        # Process datasets to get Response and log_response
        train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
        test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')
        
        # Extract spectra features (columns from index 1 to -3: exclude metadata)
        # Structure: index_id, spectra (149), SMILES_spectra, Response, log_response
        # So features are from column 1 to -3 (exclude SMILES_spectra, Response, log_response at the end)
        feature_start = 1  # Skip index_id
        feature_end = -3  # Exclude SMILES_spectra, Response, log_response at the end
        
        X_train_rf = train_data_processed.iloc[:, feature_start:feature_end].values
        y_train_rf = train_data_processed['log_response'].values
        
        X_val_rf = test_data_processed.iloc[:, feature_start:feature_end].values
        y_val_rf = test_data_processed['log_response'].values
        
        print(f"Training RF on spectra features: X_train shape {X_train_rf.shape}, y_train shape {y_train_rf.shape}")
        
        # ==================== TRAIN RANDOM FOREST DIRECTLY ON SPECTRA ==================== #
        print(f"Training Random Forest directly on spectra (n_estimators={rf_n_estimators})...")
        rf_direct_model = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=rf_random_state,
            n_jobs=-1,
            verbose=0
        )
        
        rf_direct_model.fit(X_train_rf, y_train_rf)
        
        # Evaluate on validation set
        val_predictions_direct = rf_direct_model.predict(X_val_rf)
        val_mse_direct = mean_squared_error(y_val_rf, val_predictions_direct)
        val_r2_direct = r2_score(y_val_rf, val_predictions_direct)
        print(f"Validation MSE: {val_mse_direct:.4f}, R²: {val_r2_direct:.4f}")
        
        # ==================== EVALUATE ON FULL VALIDATION SET ==================== #
        print("Evaluating on full validation set...")
        # Prepare full filtered dataset
        filtered_dataset_full = filtered_dataset.copy()
        
        # Create train indicator mapping
        train_indicator_map = {}
        for idx in filtered_dataset_full.index:
            train_indicator_map[idx] = 1 if idx in train_indices_set else 0
        
        # Reset index and add sequential index column
        filtered_dataset_full = filtered_dataset_full.reset_index(drop=False, names=['original_index'])
        filtered_dataset_full['index'] = filtered_dataset_full['index_id']
        
        # Process dataset
        filtered_dataset_full_processed = fd.add_response_and_log_response(filtered_dataset_full.copy(), df6_subset, smiles_col='SMILES_spectra')
        
        # Add train indicator
        filtered_dataset_full_processed['train'] = filtered_dataset_full_processed['original_index'].map(train_indicator_map).fillna(0).astype(int)
        
        # Extract features - find spectra columns (numeric columns excluding metadata)
        # Get column names to identify spectra features
        numeric_cols = filtered_dataset_full_processed.select_dtypes(include=[np.number]).columns
        # Exclude metadata columns: original_index, index_id, index, Response, log_response, train
        exclude_cols = {'original_index', 'index_id', 'index', 'Response', 'log_response', 'train'}
        spectra_cols = [col for col in numeric_cols if col not in exclude_cols]
        X_full_val_rf = filtered_dataset_full_processed[spectra_cols].values
        
        # Generate predictions with RF
        full_val_direct_predictions = rf_direct_model.predict(X_full_val_rf)

        # Create output DataFrame
        full_val_direct_output_df = pd.DataFrame()
        full_val_direct_output_df['rf_direct_tox_pred'] = full_val_direct_predictions
        
        # Add metadata
        full_val_direct_output_df['SMILES_spectra'] = filtered_dataset_full_processed['SMILES_spectra'].values
        full_val_direct_output_df['Response'] = filtered_dataset_full_processed['Response'].values
        full_val_direct_output_df['log_response'] = filtered_dataset_full_processed['log_response'].values
        full_val_direct_output_df['index_id'] = filtered_dataset_full_processed['index_id'].values
        full_val_direct_output_df['train'] = filtered_dataset_full_processed['train'].values
        
        # Verify train column
        train_count = full_val_direct_output_df['train'].sum()
        val_count = len(full_val_direct_output_df) - train_count
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
        
        direct_rf_output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/random_forest_df6"
        os.makedirs(direct_rf_output_folder, exist_ok=True)

        direct_rf_predictions_filename = f"rf_direct_{bin_part}_{threshold_part}_df_spectra.parquet"
        direct_rf_predictions_path = os.path.join(direct_rf_output_folder, direct_rf_predictions_filename)
        
        try:
            full_val_direct_output_df.to_parquet(direct_rf_predictions_path, index=False)
            print(f"✓ Successfully saved full validation predictions to {direct_rf_predictions_filename}")
        except Exception as save_error:
            print(f"✗ ERROR saving full validation predictions: {str(save_error)}")
        
        # Save RF model
        rf_direct_model_filename = f"rf_direct_model_{bin_part}_{threshold_part}.pkl"
        rf_direct_model_path = os.path.join(direct_rf_output_folder, rf_direct_model_filename)
        with open(rf_direct_model_path, 'wb') as f:
            pickle.dump(rf_direct_model, f)
        print(f"✓ Saved direct RF model to {rf_direct_model_filename}")
        
        # ==================== EVALUATE ON SUPER TEST SET ==================== #
        print("Evaluating on super test set...")
        # Extract super test set from original dataset, then remove synthetic spectra
        super_test_df_all = dataset[dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
        super_test_df = super_test_df_all[~super_test_df_all['index_id'].isin(synthetic_index_ids)].copy()
        print(f"Super test: {len(super_test_df_all)} total spectra, {len(super_test_df)} real spectra (removed {len(super_test_df_all) - len(super_test_df)} synthetic spectra)")
        
        if len(super_test_df) > 0:
            print(f"Super test set size: {len(super_test_df)} samples")
            
            super_test_df['index'] = super_test_df['index_id']
            super_test_processed = fd.add_response_and_log_response(super_test_df.copy(), df6_subset, smiles_col='SMILES_spectra')
            
            # Extract features - find spectra columns (numeric columns excluding metadata)
            numeric_cols_st = super_test_processed.select_dtypes(include=[np.number]).columns
            exclude_cols_st = {'index_id', 'index', 'Response', 'log_response'}
            spectra_cols_st = [col for col in numeric_cols_st if col not in exclude_cols_st]
            X_super_test_rf = super_test_processed[spectra_cols_st].values
            
            # Generate predictions with RF
            super_test_direct_predictions = rf_direct_model.predict(X_super_test_rf)

            # Create super test output DataFrame
            super_test_direct_output_df = pd.DataFrame()
            super_test_direct_output_df['rf_direct_tox_pred'] = super_test_direct_predictions
            
            # Add metadata
            super_test_direct_output_df['SMILES_spectra'] = super_test_processed['SMILES_spectra'].values
            super_test_direct_output_df['log_response'] = super_test_processed['log_response'].values
            super_test_direct_output_df['Response'] = super_test_processed['Response'].values
            super_test_direct_output_df['index_id'] = super_test_processed['index_id'].values
            super_test_direct_output_df['train'] = 0  # Super test samples were not in training set
            
            # Save super test set predictions
            direct_rf_super_test_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/random_forest_df6_super_test"
            os.makedirs(direct_rf_super_test_folder, exist_ok=True)

            direct_rf_super_test_filename = f"super_test_rf_direct_{bin_part}_{threshold_part}_df_spectra.parquet"
            direct_rf_super_test_path = os.path.join(direct_rf_super_test_folder, direct_rf_super_test_filename)
            
            try:
                super_test_direct_output_df.to_parquet(direct_rf_super_test_path, index=False)
                print(f"✓ Successfully saved super test predictions to {direct_rf_super_test_filename}")
            except Exception as save_error:
                print(f"✗ ERROR saving super test predictions: {str(save_error)}")
            
            print(f"Super test evaluation completed for {len(super_test_df)} samples")
        else:
            print("No super test samples found in this dataset")
            
        print(f"Completed processing {dataset_name}")

        # Clear memory
        del rf_direct_model
        gc.collect()
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        gc.collect()
        continue

print(f"\n=== DIRECT RF TRAINING AND EVALUATION COMPLETED ===")
print(f"\nAll processing completed successfully!")

