import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'Python_files'))
import functions_enc as f
import function_depot as fd

### USER SETTINGS
dataset_name = 'bin1_thresh0_5_df_spectra'  # 'bin1_thresh0_05_df_spectra'
num_loops = 25

VAL_DIR  = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/regular_classifier_synth_abl_loop"
SUPER_DIR = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/regular_classifier_synth_abl_loop_super_test"
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(SUPER_DIR, exist_ok=True)

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
num_layers = 6
batch_size = 256
epochs = 350
lr = 0.0001
dropout = 0.35
layer1_size = 1000
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

for loop_counter in range(num_loops):
    print(f"\n{'='*80}\nLOOP {loop_counter+1}/{num_loops}\n{'='*80}")

    # ============================================================
    # === TOXICITY LEVEL FILTERING CONTROL ===
    # Set to True to enable removal, False to disable
    ENABLE_TOX_FILTERING = True
    # Removal percentage for each toxicity level (0-100, set to 0 to skip)
    tox_removal_percent_level_0 = 0
    tox_removal_percent_level_1 = 0
    tox_removal_percent_level_2 = 0
    tox_removal_percent_level_3 = 70
    tox_removal_percent_level_4 = 0
    # ============================================================

    dataset = orig_dataset.copy()
    dataset_no_super_test = dataset[~dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
    
    # ===================================================================
    # SYNTHETIC ABLATION: Remove all synthetic spectra from dataset
    # ===================================================================
    synthetic_ids = set(df6_spectra.loc[df6_spectra['synthetic'] == 1, 'index_id'])
    before_synth = len(dataset_no_super_test)
    dataset_no_super_test = dataset_no_super_test[~dataset_no_super_test['index_id'].isin(synthetic_ids)].copy()
    after_synth = len(dataset_no_super_test)
    print(f"Removed {before_synth - after_synth} samples with synthetic==1")
    
    if 'Group' not in dataset_no_super_test.columns:
        dataset_no_super_test['Group'] = dataset_no_super_test['index_id'].map(id_to_group).fillna('Unknown')
    if 'CE_clean' not in dataset_no_super_test.columns:
        dataset_no_super_test['CE_clean'] = dataset_no_super_test['index_id'].map(id_to_ce_clean).fillna('Unknown')
    
    # Filter for SMILES with at least 4 spectra
    counts = dataset_no_super_test['SMILES_spectra'].value_counts()
    valid_smiles = counts[counts >= 3].index
    filtered_dataset = dataset_no_super_test[dataset_no_super_test['SMILES_spectra'].isin(valid_smiles)].copy()

    # ============================================================
    # === TOXICITY LEVEL FILTERING (EASY TO COMMENT OUT) ===
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
            all_smiles_to_remove = set()
            np.random.seed(loop_counter + 999)  # Reproducible randomization
            
            for tox_level, removal_percent in active_levels.items():
                level_smiles = temp_filtered[temp_filtered['tox_level'] == tox_level]['SMILES_spectra'].unique()
                n_remove = int(len(level_smiles) * (removal_percent / 100))
                
                if n_remove > 0:
                    smiles_to_remove = np.random.choice(level_smiles, size=n_remove, replace=False)
                    all_smiles_to_remove.update(smiles_to_remove)
                    print(f"  Level {tox_level}: Removing {n_remove}/{len(level_smiles)} SMILES ({removal_percent}%)")
            
            # Apply the filtering
            original_size = len(filtered_dataset)
            filtered_dataset = filtered_dataset[~filtered_dataset['SMILES_spectra'].isin(all_smiles_to_remove)].copy()
            new_size = len(filtered_dataset)
            
            # Drop temporary tox_level columns so rest of code runs unchanged
            tox_cols_to_drop = [col for col in filtered_dataset.columns if col.startswith('tox_level')]
            if tox_cols_to_drop:
                filtered_dataset = filtered_dataset.drop(columns=tox_cols_to_drop)
            
            print(f"  Dataset size: {original_size} -> {new_size} (removed {original_size - new_size} spectra)")
            print(f"--- Toxicity Filtering Complete ---\n")
        else:
            print("\n--- Toxicity Level Filtering: No levels configured for removal ---\n")
    else:
        print("\n--- Toxicity Level Filtering DISABLED ---\n")
    # ============================================================
    # === END TOXICITY LEVEL FILTERING ===
    # ============================================================

    # ===================================================================
    # TRAIN-TEST SPLIT: SMILES-based 50/50 split (with extras to train)
    # Splits each SMILES group 50/50 between train and test
    # ensuring no data leakage between sets
    # Note: Synthetic data already removed in ablation study
    # CHANGE: Now balances CE_clean levels within each SMILES group #change
    # ===================================================================
    smiles_groups = filtered_dataset.groupby('SMILES_spectra')
    train_indices, test_indices = [], []
    np.random.seed(loop_counter + 42)
    for smiles, group in smiles_groups:
        # CHANGE: Group by CE_clean level within this SMILES #change
        ce_groups = group.groupby('CE_clean', dropna=False) #change
        smiles_train_idx = [] #change
        smiles_test_idx = [] #change
        
        for ce_level, ce_group in ce_groups: #change
            idx = ce_group.index.values #change
            n = len(idx) #change
            np.random.shuffle(idx) #change
            split = n // 2 #change
            # CHANGE: Distribute this CE_clean level evenly between train/test #change
            smiles_test_idx.extend(idx[:split]) #change
            smiles_train_idx.extend(idx[split:]) #change
        
        # CHANGE: Add this SMILES' indices to global lists #change
        test_indices.extend(smiles_test_idx) #change
        train_indices.extend(smiles_train_idx) #change
    
    train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
    test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)
    train_indices_set = set(train_indices)

    # Keep original index_id - DO NOT overwrite!
    train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
    train_data_processed = fd.add_tox_levels(train_data_processed)
    # Add 'index' column for the tensor function, but use index_id values
    train_data_processed['index'] = train_data_processed['index_id']
    
    test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')
    test_data_processed = fd.add_tox_levels(test_data_processed)
    # Add 'index' column for the tensor function, but use index_id values
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
    # Use bin/threshold for logging only
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
    # Keep original index_id - DO NOT overwrite!
    filtered_dataset_full_processed = fd.add_response_and_log_response(filtered_dataset_full.copy(), df6_subset, smiles_col='SMILES_spectra')
    filtered_dataset_full_processed = fd.add_tox_levels(filtered_dataset_full_processed)
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
    super_test_df = dataset[dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
    if len(super_test_df) > 0:
        # Remove synthetic data from super test set
        before_synth_super = len(super_test_df)
        super_test_df = super_test_df[~super_test_df['index_id'].isin(synthetic_ids)].copy()
        after_synth_super = len(super_test_df)
        print(f"Removed {before_synth_super - after_synth_super} synthetic samples from super test set")
        if 'Group' not in super_test_df.columns:
            super_test_df['Group'] = super_test_df['index_id'].map(id_to_group).fillna('Unknown')
        if 'CE_clean' not in super_test_df.columns:
            super_test_df['CE_clean'] = super_test_df['index_id'].map(id_to_ce_clean).fillna('Unknown')
        # Keep original index_id - DO NOT overwrite
        super_test_processed = fd.add_response_and_log_response(super_test_df.copy(), df6_subset, smiles_col='SMILES_spectra')
        super_test_processed = fd.add_tox_levels(super_test_processed)

        # Ensure one-hot columns for Group and CE_clean match training set
        # Get one-hot columns from train_data_processed
        train_group_cols = [col for col in train_data_processed.columns if col.startswith('group_')]
        train_ce_cols = [col for col in train_data_processed.columns if col.startswith('ce_')]
        # Add any missing one-hot columns to super_test_processed, fill with 0
        for col in train_group_cols:
            if col not in super_test_processed.columns:
                super_test_processed[col] = 0
        for col in train_ce_cols:
            if col not in super_test_processed.columns:
                super_test_processed[col] = 0
        # Add 'index' column for the tensor function, but use index_id values
        super_test_processed['index'] = super_test_processed['index_id']
        # Ensure column order matches
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