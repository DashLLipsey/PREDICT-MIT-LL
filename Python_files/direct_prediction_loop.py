# Package Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
import requests
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
import functions_enc as f
import function_depot as fd

### USER SETTINGS
dataset_name = 'bin1_thresh0_05_df_spectra'  # 'bin1_thresh0_05_df_spectra'
num_loops = 10

# --- Toxicity filtering config (easy to comment out) ---
ENABLE_TOX_FILTERING = False  # Set to True to enable toxicity-based filtering
# Removal percentage for each toxicity level (0-100, set to 0 to skip)
tox_removal_percent_level_0 = 0
tox_removal_percent_level_1 = 0
tox_removal_percent_level_2 = 86.5
tox_removal_percent_level_3 = 94.3
tox_removal_percent_level_4 = 79.5

VAL_DIR  = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/regular_classifier_loop"
SUPER_DIR = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/regular_classifier_loop_super_test"
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(SUPER_DIR, exist_ok=True)

# New Super Test SMILES list
super_test_smiles = [
    'CCOP(=S)(OCC)Oc1ccc([N+](=O)[O-])cc1',
    'CCOP(=S)(OCC)Oc1ccc(S(C)=O)cc1',
    'CC1(C)O[C@@H]2C[C@H]3[C@@H]4C[C@H](F)C5=CC(=O)C=C[C@]5(C)[C@H]4[C@@H](O)C[C@]3(C)[C@]2(C(=O)CO)O1 CC(=O)OC1(C)CC(C)C(=O)C(C(O)CC2CC(=O)NC(=O)C2)C1',
    'NC(=S)Nc1ccccc1',
    'CC(=O)OC[C@]12C[C@H](OC(=O)CC(C)C)C(C)=C[C@H]1O[C@@H]1[C@H](O)[C@@H](OC(C)=O)[C@@]2(C)[C@]12CO2',
    #(Level 0 that would get filtered out)
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
    # (Level 1 that would get filtered out)
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

num_classes = 5
num_layers = 1
batch_size = 256
epochs = 250
lr = 0.0001
dropout = 0.35
layer1_size = 32
layer2_size = 250
layer3_size = 50
layer4_size = 20
criterion = CrossEntropyLoss()

print("=== DIRECT TOXICITY PREDICTION (Repeat Loops) ===")
print(f"Super test SMILES to remove from training: {len(super_test_smiles)}")

device = fd.set_up_gpu()

df6_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet")
df6_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet")
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes_df6"
dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.parquet")
orig_dataset = pd.read_parquet(dataset_path)
orig_dataset = pd.DataFrame(orig_dataset) if not isinstance(orig_dataset, pd.DataFrame) else orig_dataset

id_to_group = dict(zip(df6_spectra['index_id'], df6_spectra['Group']))
id_to_ce_clean = dict(zip(df6_spectra['index_id'], df6_spectra['CE_clean']))

# --- Build synthetic and SMILES lookup ---
id_to_synthetic = dict(zip(df6_spectra['index_id'], df6_spectra['synthetic'].fillna(0)))
# --- Identify synthetic spectra (individual index_ids) ---
synthetic_index_ids = set(idx for idx, syn in id_to_synthetic.items() if syn == 1)
print(f"Identified {len(synthetic_index_ids)} synthetic spectra (by index_id)")
# Keep all SMILES - we'll filter synthetic spectra at the dataframe level

for loop_counter in range(num_loops):
    print(f"\n{'='*80}\nLOOP {loop_counter+1}/{num_loops}\n{'='*80}")

    dataset = orig_dataset.copy()
    # Exclude super test SMILES from training/validation
    dataset_no_super_test = dataset[~dataset['SMILES_spectra'].isin(super_test_smiles)].copy()

    if 'Group' not in dataset_no_super_test.columns:
        dataset_no_super_test['Group'] = dataset_no_super_test['index_id'].map(id_to_group).fillna('Unknown')
    if 'CE_clean' not in dataset_no_super_test.columns:
        dataset_no_super_test['CE_clean'] = dataset_no_super_test['index_id'].map(id_to_ce_clean).fillna('Unknown')
    counts = dataset_no_super_test['SMILES_spectra'].value_counts()
    valid_smiles = counts[counts >= 3].index
    filtered_dataset = dataset_no_super_test[dataset_no_super_test['SMILES_spectra'].isin(valid_smiles)].copy()

    # ============================================================
    # === TOXICITY LEVEL FILTERING ===
    # ============================================================
    if ENABLE_TOX_FILTERING:
        # Collect removal percentages for each level
        tox_removal_dict = {
            0: tox_removal_percent_level_0,
            1: tox_removal_percent_level_1,
            2: tox_removal_percent_level_2,
            3: tox_removal_percent_level_3,
            4: tox_removal_percent_level_4
        }
        active_levels = {k: v for k, v in tox_removal_dict.items() if v > 0}
        
        if active_levels:
            print(f"\n--- Toxicity Level Filtering ENABLED ---")
            print(f"Removal percentages: {active_levels}")
            
            # Temporarily add tox levels to identify which SMILES to remove
            temp_filtered = filtered_dataset.copy()
            temp_filtered = fd.add_response_and_log_response(temp_filtered, df6_subset, smiles_col='SMILES_spectra')
            temp_filtered = fd.add_tox_levels(temp_filtered)
            
            # If tox_level column doesn't exist, derive it from one-hot columns
            if 'tox_level' not in temp_filtered.columns:
                tox_level_cols = [f'tox_level_{i}' for i in range(5)]
                if all(col in temp_filtered.columns for col in tox_level_cols):
                    temp_filtered['tox_level'] = temp_filtered[tox_level_cols].values.argmax(axis=1)
            
            # Ensure tox_level is numeric for comparison
            if 'tox_level' in temp_filtered.columns:
                temp_filtered['tox_level'] = pd.to_numeric(temp_filtered['tox_level'], errors='coerce')
            
            # Remove SMILES for each toxicity level with its specific removal percentage
            # Strategy: Remove SMILES such that % spectra removed ≈ removal_percent
            all_smiles_to_remove = set()
            np.random.seed(loop_counter + 999)  # Reproducible randomization
            
            for tox_level, removal_percent in active_levels.items():
                level_data = temp_filtered[temp_filtered['tox_level'] == tox_level]
                total_spectra = len(level_data)
                target_spectra_to_remove = int(total_spectra * (removal_percent / 100))
                
                # Count spectra per SMILES at this level
                smiles_spectra_counts = level_data['SMILES_spectra'].value_counts()
                smiles_list = smiles_spectra_counts.index.tolist()
                
                # Special case: 100% removal means remove ALL SMILES at this level
                if removal_percent >= 100:
                    smiles_to_remove = smiles_list
                    cumulative_spectra = total_spectra
                else:
                    # Shuffle SMILES and accumulate until we reach target spectra count
                    np.random.shuffle(smiles_list)
                    smiles_to_remove = []
                    cumulative_spectra = 0
                    
                    for smiles in smiles_list:
                        if cumulative_spectra >= target_spectra_to_remove:
                            break
                        smiles_to_remove.append(smiles)
                        cumulative_spectra += smiles_spectra_counts[smiles]
                
                all_smiles_to_remove.update(smiles_to_remove)
                actual_percent = (cumulative_spectra / total_spectra * 100) if total_spectra > 0 else 0
                print(f"  Level {tox_level}: {len(smiles_to_remove)}/{len(smiles_list)} SMILES removed "
                      f"({cumulative_spectra}/{total_spectra} spectra = {actual_percent:.1f}%)")
            
            # Apply the filtering
            original_size = len(filtered_dataset)
            filtered_dataset = filtered_dataset[~filtered_dataset['SMILES_spectra'].isin(all_smiles_to_remove)].copy()
            new_size = len(filtered_dataset)
            
            print(f"  Dataset size: {original_size} -> {new_size} (removed {original_size - new_size} spectra)")
            print(f"--- Toxicity Filtering Complete ---\n")
        else:
            print("\n--- Toxicity Level Filtering: No levels configured for removal ---\n")
    else:
        print("\n--- Toxicity Level Filtering DISABLED ---\n")
    # ============================================================
    # === END TOXICITY LEVEL FILTERING ===
    # ============================================================
    
    # Add Response and tox_levels to filtered_dataset for use in model training
    filtered_dataset = fd.add_response_and_log_response(filtered_dataset, df6_subset, smiles_col='SMILES_spectra')
    filtered_dataset = fd.add_tox_levels(filtered_dataset)
    
    # Ensure all toxicity level columns exist (even if empty due to filtering)
    for tox_col in ['tox_level_0', 'tox_level_1', 'tox_level_2', 'tox_level_3', 'tox_level_4']:
        if tox_col not in filtered_dataset.columns:
            filtered_dataset[tox_col] = 0

    # === Synthetic-awareness in splitting ===
    synthetic_mask = filtered_dataset['index_id'].map(lambda idx: id_to_synthetic.get(idx, 0)==1)
    real_mask = ~synthetic_mask

    synthetic_data = filtered_dataset[synthetic_mask].copy()
    real_data = filtered_dataset[real_mask].copy()

    # ===================================================================
    # TRAIN-TEST SPLIT: SMILES-based 50/50 split (CE balanced within each SMILES)
    # ===================================================================
    smiles_groups = real_data.groupby('SMILES_spectra')
    train_indices, test_indices = [], []
    np.random.seed(loop_counter + 42)
    for smiles, group in smiles_groups:
        # CHANGE: Group by CE_clean level within this SMILES 
        ce_groups = group.groupby('CE_clean', dropna=False) 
        smiles_train_idx = [] 
        smiles_test_idx = [] 
        
        for ce_level, ce_group in ce_groups: 
            idx = ce_group.index.values 
            n = len(idx) 
            np.random.shuffle(idx) 
            split = n // 2 #change
            # CHANGE: Distribute this CE_clean level evenly between train/test 
            smiles_test_idx.extend(idx[:split]) 
            smiles_train_idx.extend(idx[split:]) 
        
        # CHANGE: Add this SMILES' indices to global lists 
        test_indices.extend(smiles_test_idx) 
        train_indices.extend(smiles_train_idx) 
    
    # Add ALL synthetic to training, NOT to test
    train_indices.extend(synthetic_data.index.values)

    train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
    test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)
    train_indices_set = set(train_indices)

    # Add 'index' column for the tensor function, but use index_id values
    train_data_processed = train_data.copy()
    train_data_processed['index'] = train_data_processed['index_id']
    
    test_data_processed = test_data.copy()
    test_data_processed['index'] = test_data_processed['index_id']

    x_train, y_train_tox, train_indices_tensor = fd.create_dataset_tensors_direct_toxicity_e1e2(
        train_data_processed, device, start_idx=1, stop_idx=-11)
    x_val, y_val_tox, val_indices_tensor = fd.create_dataset_tensors_direct_toxicity_e1e2(
        test_data_processed, device, start_idx=1, stop_idx=-11)
    actual_input_size = x_train.shape[1]
    direct_tox_model = fd.Direct_Toxicity_Encoder(
            input_size=actual_input_size,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout_rate=dropout
    ).to(device)
# Alternative architectures 
    # direct_tox_model = fd.Direct_Toxicity_Encoder_custom2(
    #         input_size=actual_input_size,
    #         num_classes=num_classes,
    #         dropout_rate=dropout,
    #         layer_size=layer1_size,
    # ).to(device)

    # direct_tox_model = fd.Direct_Toxicity_Encoder_custom3(
    #         input_size=actual_input_size,
    #         num_classes=num_classes,
    #         dropout_rate=dropout,
    #         layer1_size=layer1_size,
    #         layer2_size=layer2_size
    # ).to(device)

    # direct_tox_model = fd.Direct_Toxicity_Encoder_custom4(
    #         input_size=actual_input_size,
    #         num_classes=num_classes,
    #         dropout_rate=dropout,
    #         layer1_size=layer1_size,
    #         layer2_size=layer2_size,
    #         layer3_size=layer3_size
    # ).to(device)

    train_loader = DataLoader(TensorDataset(x_train, y_train_tox, train_indices_tensor),
                              batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
    val_loader = DataLoader(TensorDataset(x_val, y_val_tox, val_indices_tensor),
                            batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
    bin_size, threshold = parse_dataset_name(dataset_name)
    direct_tox_config = {
        'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
        'wandb_project': 'MIT-Lincoln-Lab',
        'wandb_name': f"direct_toxicity_{dataset_name}_loop{loop_counter}",
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
        'loop': loop_counter
    }

    print("Training loop", loop_counter)
    trained_direct_tox_model, *_ = fd.train_direct_toxicity_encoder_e1e2(
        model=direct_tox_model,
        train_data=train_loader,
        val_data=val_loader,
        epochs=epochs,
        learning_rate=lr,
        criterion=criterion,
        device=device,
        config=direct_tox_config
    )

    # ==== EVAL FULL VALID (same as your script)
    filtered_dataset_full = filtered_dataset.copy()
    train_indicator_map = {idx: 1 if idx in train_indices_set else 0 for idx in filtered_dataset_full.index}
    filtered_dataset_full = filtered_dataset_full.reset_index(drop=False, names=['original_index'])
    # filtered_dataset already has Response and tox_levels, no need to add again
    filtered_dataset_full_processed = filtered_dataset_full.copy()
    filtered_dataset_full_processed['train'] = filtered_dataset_full_processed['original_index'].map(train_indicator_map).fillna(0).astype(int)
    # Add 'index' column for tensor function using index_id
    filtered_dataset_full_processed['index'] = filtered_dataset_full_processed['index_id']
    filtered_dataset_for_tensors = filtered_dataset_full_processed.drop(columns=['original_index', 'train']).copy()
    x_full_val, y_full_val_tox, full_val_indices_tensor = fd.create_dataset_tensors_direct_toxicity_e1e2(
        filtered_dataset_for_tensors, device, start_idx=1, stop_idx=-11)
    direct_tox_model.eval()
    with torch.no_grad():
        full_val_tox_logits = direct_tox_model(x_full_val).cpu().numpy()
    tox_logits_cols = [f'direct_tox_logit_{j}' for j in range(num_classes)]
    full_val_output_df = pd.DataFrame(full_val_tox_logits, columns=tox_logits_cols)
    full_val_output_df['direct_tox_pred_class'] = np.argmax(full_val_tox_logits, axis=1)
    full_val_output_df['true_tox_class'] = y_full_val_tox.cpu().numpy()
    full_val_output_df['SMILES_spectra'] = filtered_dataset_full_processed['SMILES_spectra'].values
    full_val_output_df['Response'] = filtered_dataset_full_processed['Response'].values
    full_val_output_df['index_id'] = filtered_dataset_full_processed['index_id'].values  # Keep original index_id!
    full_val_output_df['Group'] = filtered_dataset_full_processed['Group'].values
    full_val_output_df['CE_clean'] = filtered_dataset_full_processed['CE_clean'].values
    full_val_output_df['train'] = filtered_dataset_full_processed['train'].values

    # Reorder columns: metadata, true class, logits/predictions, train
    metadata_cols = ['SMILES_spectra', 'index_id', 'Response', 'Group', 'CE_clean']
    prediction_cols = ['true_tox_class'] + tox_logits_cols + ['direct_tox_pred_class']
    column_order = metadata_cols + prediction_cols + ['train']
    full_val_output_df = full_val_output_df[column_order]

    # SAVE MAIN VALIDATION OUT
    val_out_fn = f"direct_tox_{dataset_name}_loop{loop_counter}.parquet"
    val_out_path = os.path.join(VAL_DIR, val_out_fn)
    full_val_output_df.to_parquet(val_out_path, index=False)
    print(f"✓ Saved validation set predictions: {val_out_fn}")

    # ==== SUPER TEST ====
    # Get all super test SMILES, then filter out only synthetic spectra (by index_id)
    super_test_df_all = dataset[dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
    super_test_df = super_test_df_all[~super_test_df_all['index_id'].isin(synthetic_index_ids)].copy()
    print(f"Super test: {len(super_test_df_all)} total spectra, {len(super_test_df)} non-synthetic spectra kept")
    
    if len(super_test_df) > 0:
        if 'Group' not in super_test_df.columns:
            super_test_df['Group'] = super_test_df['index_id'].map(id_to_group).fillna('Unknown')
        if 'CE_clean' not in super_test_df.columns:
            super_test_df['CE_clean'] = super_test_df['index_id'].map(id_to_ce_clean).fillna('Unknown')
        # Keep original index_id - DO NOT overwrite
        super_test_processed = fd.add_response_and_log_response(super_test_df.copy(), df6_subset, smiles_col='SMILES_spectra')
        super_test_processed = fd.add_tox_levels(super_test_processed)
        # Add 'index' column for the tensor function, but use index_id values
        super_test_processed['index'] = super_test_processed['index_id']
        super_test_processed = super_test_processed[train_data_processed.columns]
        x_super_test, y_super_test_tox, _ = fd.create_dataset_tensors_direct_toxicity_e1e2(
            super_test_processed, device, start_idx=1, stop_idx=-11)
        with torch.no_grad():
            super_test_tox_logits = direct_tox_model(x_super_test).cpu().numpy()
        super_test_output_df = pd.DataFrame(super_test_tox_logits, columns=tox_logits_cols)
        super_test_output_df['direct_tox_pred_class'] = np.argmax(super_test_tox_logits, axis=1)
        super_test_output_df['true_tox_class'] = y_super_test_tox.cpu().numpy()
        super_test_output_df['SMILES_spectra'] = super_test_processed['SMILES_spectra'].values
        super_test_output_df['Response'] = super_test_processed['Response'].values
        super_test_output_df['index_id'] = super_test_processed['index_id'].values  # Keep original index_id!
        super_test_output_df['Group'] = super_test_processed['Group'].values
        super_test_output_df['CE_clean'] = super_test_processed['CE_clean'].values
        super_test_output_df['train'] = 0
        
        # Reorder columns: metadata, true class, logits/predictions, train
        metadata_cols = ['SMILES_spectra', 'index_id', 'Response', 'Group', 'CE_clean']
        prediction_cols = ['true_tox_class'] + tox_logits_cols + ['direct_tox_pred_class']
        column_order = metadata_cols + prediction_cols + ['train']
        super_test_output_df = super_test_output_df[column_order]
        
        super_fn = f"super_test_direct_tox_{dataset_name}_loop{loop_counter}.parquet"
        super_path = os.path.join(SUPER_DIR, super_fn)
        super_test_output_df.to_parquet(super_path, index=False)
        print(f"✓ Saved super test predictions: {super_fn}")

    # MEMORY CLEANUP
    del x_train, x_val, y_train_tox, y_val_tox, train_indices_tensor, val_indices_tensor
    del x_full_val, y_full_val_tox, full_val_indices_tensor
    if 'x_super_test' in locals():
        del x_super_test, y_super_test_tox
    del direct_tox_model, trained_direct_tox_model
    torch.cuda.empty_cache()
print('\nAll loops completed and outputs saved.')


