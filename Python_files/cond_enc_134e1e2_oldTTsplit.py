import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'Python_files'))
import functions_enc as f
import function_depot as fd

#### ==== USER-SETTINGS: CHOOSE DATASET AND REPEATS ==== ####
# --- Dataset config (set these!) ---
bin_size = 1.0  # 1.0 and 0.1     
threshold = 0.05  # 0.05 and 0.5
dataset_name = 'bin1_thresh0_05_df_spectra'  # <-- must match parquet file in grid_search_folder
num_loops = 10      # how many repeated train/val splits & models

# --- Toxicity filtering config (easy to comment out) ---
ENABLE_TOX_FILTERING = False  # Set to True to enable toxicity-based filtering
# Removal percentage for each toxicity level (0-100, set to 0 to skip)
tox_removal_percent_level_0 = 0
tox_removal_percent_level_1 = 0
tox_removal_percent_level_2 = 0
tox_removal_percent_level_3 = 0
tox_removal_percent_level_4 = 0

# --- Output folders ---
VAL_INT_DIR  = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/2_step_oldTTsplit_loop_intermediate"
VAL_FINAL_DIR = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/2_step_oldTTsplit_loop"
SUPER_INT_DIR  = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/2_step_oldTTsplit_intermediate_super_test_loop"
SUPER_FINAL_DIR = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/2_step_oldTTsplit_super_test_loop"
for d in [VAL_INT_DIR, VAL_FINAL_DIR, SUPER_INT_DIR, SUPER_FINAL_DIR]:
    os.makedirs(d, exist_ok=True)

# Step 1 embedding inputs
name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_chemnet.parquet")
morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_morganfp.parquet")
filtered_morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_filtered_morganfp.parquet")

# # Load in internal conditions with noise
# name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_chemnet_noise.parquet")
# morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_morganfp_noise.parquet")
# filtered_morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_filtered_morganfp_noise.parquet")

df6_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet")
df6_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet")
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes_df6"
dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.parquet")

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


#### ==== Model params ==== ####
embedding_num_layers = 6
embedding_batch_size = 256
embedding_epochs = 500
embedding_lr = 0.0001
lambda1 = 15
lambda3 = 5
lambda4 = 10
dropout1 = 0.35

input_length=4608
tox_num_layers = 4
tox_batch_size = 256
tox_epochs = 250
tox_lr = 0.0001
tox_num_classes = 5
dropout2 = 0.35

layer1_size = 1000
layer2_size = 250
layer3_size = 250

criterion1 = nn.MSELoss()
criterion3 = nn.MSELoss()
criterion4 = nn.MSELoss()
tox_criterion = CrossEntropyLoss()

device = fd.set_up_gpu()

regular_morgan_bits = morgan_df.shape[1] - 1
filtered_morgan_bits = filtered_morgan_df.shape[1] - 1
embedding_output_size = 512 + regular_morgan_bits + filtered_morgan_bits

id_to_group = dict(zip(df6_spectra['index_id'], df6_spectra['Group']))
id_to_ce_clean = dict(zip(df6_spectra['index_id'], df6_spectra['CE_clean']))

# Load synthetic flag from df6_spectra for all index_ids
# (If 'synthetic' contains NaNs, treat as 0 = not synthetic)
id_to_synthetic = dict(zip(df6_spectra['index_id'], df6_spectra['synthetic'].fillna(0)))

# Map SMILES_spectra -> index_id in the major dataset (for filtering super test below)
orig_dataset = pd.read_parquet(dataset_path)
orig_dataset = pd.DataFrame(orig_dataset) if not isinstance(orig_dataset, pd.DataFrame) else orig_dataset

# --- Identify synthetic spectra (individual index_ids) ---
synthetic_index_ids = set([idx for idx, syn in id_to_synthetic.items() if syn==1])
print(f"Identified {len(synthetic_index_ids)} synthetic spectra (by index_id)")

for loop_counter in range(num_loops):
    print(f'\n========== LOOP {loop_counter+1}/{num_loops} ==========')

    # SPLIT, FILTER, MAP GROUP/CLEAN (re-randomize every loop!)
    dataset = orig_dataset.copy()
    
    # Exclude super test SMILES from training/validation
    dataset_no_super_test = dataset[~dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
    
    # Map Group and CE_clean
    if 'Group' not in dataset_no_super_test.columns:
        dataset_no_super_test['Group'] = dataset_no_super_test['index_id'].map(id_to_group).fillna('Unknown')
    if 'CE_clean' not in dataset_no_super_test.columns:
        dataset_no_super_test['CE_clean'] = dataset_no_super_test['index_id'].map(id_to_ce_clean).fillna('Unknown')
    
    # Filter SMILES based on total count (real + synthetic)
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
            np.random.seed(loop_counter + 999)  # Reproducible randomization per loop
            
            for tox_level, removal_percent in active_levels.items():
                # Get all spectra at this toxicity level
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
            
            # Drop temporary tox_level columns so rest of code runs unchanged
            tox_cols_to_drop = [col for col in filtered_dataset.columns if col.startswith('tox_level')]
            if tox_cols_to_drop:
                filtered_dataset = filtered_dataset.drop(columns=tox_cols_to_drop)
            
            print(f"  Dataset size: {original_size} -> {new_size} (removed {original_size - new_size} spectra)")
            print(f"--- Toxicity Filtering Complete ---\n")
        else:
            print("\n--- Toxicity Level Filtering: No levels configured for removal ---\n")
    # ============================================================

    # === Synthetic-awareness in splitting ===
    # Create masks for synthetic and "real" index_ids:
    synthetic_mask = filtered_dataset['index_id'].map(lambda idx: id_to_synthetic.get(idx, 0)==1)
    real_mask = ~synthetic_mask

    synthetic_data = filtered_dataset[synthetic_mask].copy()
    real_data = filtered_dataset[real_mask].copy()

    # ===================================================================
    # TRAIN-TEST SPLIT: SMILES-based 50/50 split
    # Splits each SMILES group 50/50 between train and test
    # ensuring no data leakage between sets
    # Synthetic data is added to training set only
    # ===================================================================
    smiles_groups = real_data.groupby('SMILES_spectra')
    train_indices, test_indices = [], []
    np.random.seed(loop_counter + 42)
    for smiles, group in smiles_groups:
        idx = group.index.values
        n = len(idx)
        np.random.shuffle(idx)
        split = n // 2
        test_indices.extend(idx[:split])
        train_indices.extend(idx[split:])
    
    # Add ALL synthetic to training, NOT to test
    train_indices.extend(synthetic_data.index.values)

    train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
    test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)

    # Keep original index_id, add temp_index only for internal processing if needed
    # DO NOT overwrite index_id - it tracks specific spectra across the pipeline
    train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
    train_data_processed = fd.add_tox_levels(train_data_processed)
    # Ensure all toxicity level columns exist (even if empty due to filtering)
    for tox_col in ['tox_level_0', 'tox_level_1', 'tox_level_2', 'tox_level_3', 'tox_level_4']:
        if tox_col not in train_data_processed.columns:
            train_data_processed[tox_col] = 0
    # Add 'index' column for the tensor function, but use index_id values
    train_data_processed['index'] = train_data_processed['index_id']
    
    test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')
    test_data_processed = fd.add_tox_levels(test_data_processed)
    # Ensure all toxicity level columns exist (even if empty due to filtering)
    for tox_col in ['tox_level_0', 'tox_level_1', 'tox_level_2', 'tox_level_3', 'tox_level_4']:
        if tox_col not in test_data_processed.columns:
            test_data_processed[tox_col] = 0
    # Add 'index' column for the tensor function, but use index_id values
    test_data_processed['index'] = test_data_processed['index_id']


    # ==== STEP 1: EMBEDDING ====
    x_train_with_ext, y_train_emb, y_train_morgan, y_train_filtered_morgan, train_indices_tensor = fd.create_dataset_tensors_condenc_134e1e2(
        train_data_processed, name_smiles_embedding_df, morgan_df, filtered_morgan_df, device, start_idx=1, stop_idx=-11)
    x_val_with_ext, y_val_emb, y_val_morgan, y_val_filtered_morgan, val_indices_tensor = fd.create_dataset_tensors_condenc_134e1e2(
        test_data_processed, name_smiles_embedding_df, morgan_df, filtered_morgan_df, device, start_idx=1, stop_idx=-11)

    actual_input_size = x_train_with_ext.shape[1]
    embedding_model = fd.Cond_Encoder_134_dropout(input_size=actual_input_size, output_size=embedding_output_size,
                                                  num_layers=embedding_num_layers, dropout_rate=dropout1).to(device)

    train_loader_emb = DataLoader(TensorDataset(x_train_with_ext, y_train_emb, y_train_morgan, y_train_filtered_morgan, train_indices_tensor),
                                  batch_size=embedding_batch_size, shuffle=True, pin_memory=False, num_workers=0)
    val_loader_emb = DataLoader(TensorDataset(x_val_with_ext, y_val_emb, y_val_morgan, y_val_filtered_morgan, val_indices_tensor),
                                batch_size=embedding_batch_size, shuffle=False, pin_memory=False, num_workers=0)
    embedding_config = {
        'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
        'wandb_project': 'MIT-Lincoln-Lab',
        'wandb_name': f"step1_embedding_{dataset_name}_loop{loop_counter}",
        'gpu': True,
        'encoder_type': "Step 1: Embedding Model (ChemNet + Morgan + Filtered Morgan + Group + CE_clean)",
        'batch_size': embedding_batch_size,
        'output_size': embedding_output_size,
        'num_layers': embedding_num_layers,
        'learning_rate': embedding_lr,
        'epochs': embedding_epochs,
        'lambda1': lambda1,
        'lambda3': lambda3,
        'lambda4': lambda4,
        'Bin': bin_size,
        'Threshold': threshold,
        'super_test_removed': True,
        'loop_counter': loop_counter
    }

    trained_embedding_model = fd.train_model_condenc_134e1e2(
        model=embedding_model,
        train_data=train_loader_emb,
        val_data=val_loader_emb,
        epochs=embedding_epochs,
        learning_rate=embedding_lr,
        criterion1=criterion1,
        criterion3=criterion3,
        criterion4=criterion4,
        lambda1=lambda1,
        lambda3=lambda3,
        lambda4=lambda4,
        device=device,
        config=embedding_config
    )

    # ==== STEP 1: GENERATE EMBEDDINGS FOR ALL ====
    embedding_model.eval() # Enter evaluation mode to generate embeddings
    with torch.no_grad():
        train_embeddings_combined = embedding_model(x_train_with_ext).cpu()
        val_embeddings_combined = embedding_model(x_val_with_ext).cpu()
    train_pred_chemnet = train_embeddings_combined[:, :512]
    train_pred_morgan = train_embeddings_combined[:, 512:512+regular_morgan_bits]
    train_pred_filtered_morgan = train_embeddings_combined[:, 512+regular_morgan_bits:]
    val_pred_chemnet = val_embeddings_combined[:, :512]
    val_pred_morgan = val_embeddings_combined[:, 512:512+regular_morgan_bits]
    val_pred_filtered_morgan = val_embeddings_combined[:, 512+regular_morgan_bits:]

    all_pred_chemnet = torch.cat([train_pred_chemnet, val_pred_chemnet], dim=0)
    all_pred_morgan = torch.cat([train_pred_morgan, val_pred_morgan], dim=0)
    all_pred_filtered_morgan = torch.cat([train_pred_filtered_morgan, val_pred_filtered_morgan], dim=0)

    emb_cols = [f'cond_emb_{j}' for j in range(512)]
    morgan_cols = [f'cond_morgan_{j}' for j in range(regular_morgan_bits)]
    filtered_morgan_cols = [f'cond_filtered_morgan_{j}' for j in range(filtered_morgan_bits)]

    intermediate_df = pd.DataFrame(all_pred_chemnet.numpy(), columns=emb_cols)
    intermediate_df = pd.concat([intermediate_df, pd.DataFrame(all_pred_morgan.numpy(), columns=morgan_cols)], axis=1)
    intermediate_df = pd.concat([intermediate_df, pd.DataFrame(all_pred_filtered_morgan.numpy(), columns=filtered_morgan_cols)], axis=1)

    combined_processed = pd.concat([train_data_processed, test_data_processed], axis=0).reset_index(drop=True)
    intermediate_df['SMILES_spectra'] = combined_processed['SMILES_spectra'].values
    intermediate_df['index_id'] = combined_processed['index_id'].values  # Keep original index_id!
    intermediate_df['Response'] = combined_processed['Response'].values
    intermediate_df['Group'] = combined_processed['Group'].values
    intermediate_df['CE_clean'] = combined_processed['CE_clean'].values
    if 'tox_level' in combined_processed.columns:
        intermediate_df['tox_level'] = combined_processed['tox_level'].values
    for k in range(5):
        col = f'tox_level_{k}'
        if col in combined_processed.columns:
            intermediate_df[col] = combined_processed[col].values
    intermediate_df['train'] = [1]*len(train_data_processed) + [0]*len(test_data_processed)

    # Reorder columns: metadata, embeddings, train
    metadata_cols = ['SMILES_spectra', 'index_id', 'Response', 'Group', 'CE_clean']
    tox_cols = ['tox_level'] + [f'tox_level_{k}' for k in range(5)]
    metadata_cols.extend([col for col in tox_cols if col in intermediate_df.columns])
    embedding_cols = emb_cols + morgan_cols + filtered_morgan_cols
    column_order = metadata_cols + embedding_cols + ['train']
    intermediate_df = intermediate_df[column_order]

    interm_filename = f"intermediate_embeddings_{dataset_name}_loop{loop_counter}.parquet"
    interm_path = os.path.join(VAL_INT_DIR, interm_filename)
    intermediate_df.to_parquet(interm_path, index=False)
    print(f"✓ Saved intermediate embeddings: {interm_filename}")

    # ==== STEP 1: SUPER-TEST EMBEDDINGS ====
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
        # Ensure all toxicity level columns exist (even if empty due to filtering)
        for tox_col in ['tox_level_0', 'tox_level_1', 'tox_level_2', 'tox_level_3', 'tox_level_4']:
            if tox_col not in super_test_processed.columns:
                super_test_processed[tox_col] = 0
        # Add 'index' column for the tensor function, but use index_id values
        super_test_processed['index'] = super_test_processed['index_id']
        
        # Ensure super_test has same columns as training data (align column order)
        # This prevents shape mismatches in the model
        common_cols = [col for col in train_data_processed.columns if col in super_test_processed.columns]
        super_test_processed = super_test_processed[common_cols]

        x_super_test_with_ext, y_super_test_emb, y_super_test_morgan, y_super_test_filtered_morgan, _ = fd.create_dataset_tensors_condenc_134e1e2(
            super_test_processed, name_smiles_embedding_df, morgan_df, filtered_morgan_df, device, start_idx=1, stop_idx=-11)
        with torch.no_grad():
            super_test_embeddings_combined = embedding_model(x_super_test_with_ext).cpu()
        super_test_pred_chemnet = super_test_embeddings_combined[:, :512]
        super_test_pred_morgan = super_test_embeddings_combined[:, 512:512+regular_morgan_bits]
        super_test_pred_filtered_morgan = super_test_embeddings_combined[:, 512+regular_morgan_bits:]

        super_test_emb_df = pd.DataFrame(super_test_pred_chemnet.numpy(), columns=emb_cols)
        super_test_emb_df = pd.concat([super_test_emb_df, pd.DataFrame(super_test_pred_morgan.numpy(), columns=morgan_cols)], axis=1)
        super_test_emb_df = pd.concat([super_test_emb_df, pd.DataFrame(super_test_pred_filtered_morgan.numpy(), columns=filtered_morgan_cols)], axis=1)
        super_test_emb_df['SMILES_spectra'] = super_test_processed['SMILES_spectra'].values
        super_test_emb_df['index_id'] = super_test_processed['index_id'].values  # Keep original index_id!
        super_test_emb_df['Response'] = super_test_processed['Response'].values
        super_test_emb_df['Group'] = super_test_processed['Group'].values
        super_test_emb_df['CE_clean'] = super_test_processed['CE_clean'].values
        if 'tox_level' in super_test_processed.columns:
            super_test_emb_df['tox_level'] = super_test_processed['tox_level'].values
        for k in range(5):
            col = f'tox_level_{k}'
            if col in super_test_processed.columns:
                super_test_emb_df[col] = super_test_processed[col].values
        super_test_emb_df['train'] = 0
        
        # Reorder columns: metadata, embeddings, train
        metadata_cols = ['SMILES_spectra', 'index_id', 'Response', 'Group', 'CE_clean']
        tox_cols = ['tox_level'] + [f'tox_level_{k}' for k in range(5)]
        metadata_cols.extend([col for col in tox_cols if col in super_test_emb_df.columns])
        embedding_cols = emb_cols + morgan_cols + filtered_morgan_cols
        column_order = metadata_cols + embedding_cols + ['train']
        super_test_emb_df = super_test_emb_df[column_order]
        
        super_interm_filename = f"super_test_intermediate_embeddings_{dataset_name}_loop{loop_counter}.parquet"
        super_interm_path = os.path.join(SUPER_INT_DIR, super_interm_filename)
        super_test_emb_df.to_parquet(super_interm_path, index=False)
        print(f"✓ Saved super test embeddings: {super_interm_filename}")

    # ==== STEP 2: TOXICITY CLASSIFIER ====
    train_intermediate_df = intermediate_df[intermediate_df['train']==1].reset_index(drop=True)
    val_intermediate_df = intermediate_df[intermediate_df['train']==0].reset_index(drop=True)

    train_concat_emb, train_tox_labels = fd.create_dataset_tensors_toxicity_classifier_134(
        train_intermediate_df, device)
    val_concat_emb, val_tox_labels = fd.create_dataset_tensors_toxicity_classifier_134(
        val_intermediate_df, device)

    tox_classifier = fd.ToxicityClassifier_134(num_layers=tox_num_layers, 
                                               num_classes=tox_num_classes, 
                                               dropout_rate=dropout2).to(device)

    
    train_loader_tox = DataLoader(TensorDataset(train_concat_emb, train_tox_labels),
                             batch_size=tox_batch_size, shuffle=True, pin_memory=False, num_workers=0)
    val_loader_tox = DataLoader(TensorDataset(val_concat_emb, val_tox_labels),
                             batch_size=tox_batch_size, shuffle=False, pin_memory=False, num_workers=0)
    tox_config = {
        'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
        'wandb_project': 'MIT-Lincoln-Lab',
        'wandb_name': f"step2_toxicity_{dataset_name}_loop{loop_counter}",
        'gpu': True,
        'model_type': "Step 2: Toxicity Classifier",
        'batch_size': tox_batch_size,
        'num_layers': tox_num_layers,
        'num_classes': tox_num_classes,
        'learning_rate': tox_lr,
        'epochs': tox_epochs,
        'Bin': bin_size,
        'Threshold': threshold,
        'super_test_removed': True,
        'loop_counter': loop_counter
    }
    trained_tox_classifier, *_ = fd.train_toxicity_classifier_134(
        model=tox_classifier,
        train_data=train_loader_tox,
        val_data=val_loader_tox,
        epochs=tox_epochs,
        learning_rate=tox_lr,
        criterion=tox_criterion,
        device=device,
        config=tox_config
    )

    # Eval: produce logits/preds for all data sets
    tox_classifier.eval()
    with torch.no_grad():
        train_tox_logits = tox_classifier(train_concat_emb).cpu().numpy()
        val_tox_logits = tox_classifier(val_concat_emb).cpu().numpy()
    tox_logits_cols = [f'cond_tox_logit_{j}' for j in range(tox_num_classes)]
    for i, col in enumerate(tox_logits_cols):
        train_intermediate_df[col] = train_tox_logits[:, i]
        val_intermediate_df[col] = val_tox_logits[:, i]
    train_intermediate_df['cond_tox_pred_class'] = np.argmax(train_tox_logits, axis=1)
    val_intermediate_df['cond_tox_pred_class'] = np.argmax(val_tox_logits, axis=1)
    full_val_output_df = pd.concat([train_intermediate_df, val_intermediate_df], axis=0).reset_index(drop=True)
    for col in ['SMILES_spectra', 'index_id', 'Response', 'train', 'tox_level', 'tox_level_0', 'tox_level_1', 'tox_level_2', 'tox_level_3', 'tox_level_4', 'Group', 'CE_clean']:
        if col in intermediate_df.columns and col not in full_val_output_df.columns:
            full_val_output_df[col] = intermediate_df[col].values

    # Reorder columns: metadata, embeddings, logits/predictions, train
    metadata_cols = ['SMILES_spectra', 'index_id', 'Response', 'Group', 'CE_clean']
    tox_cols = ['tox_level'] + [f'tox_level_{k}' for k in range(5)]
    metadata_cols.extend([col for col in tox_cols if col in full_val_output_df.columns])
    embedding_cols = emb_cols + morgan_cols + filtered_morgan_cols
    prediction_cols = tox_logits_cols + ['cond_tox_pred_class']
    column_order = metadata_cols + embedding_cols + prediction_cols + ['train']
    column_order = [col for col in column_order if col in full_val_output_df.columns]
    full_val_output_df = full_val_output_df[column_order]

    # Save intermediate, then FINAL val
    val_final_fn = f"cond_enc_{dataset_name}_loop{loop_counter}.parquet"
    val_final_path = os.path.join(VAL_FINAL_DIR, val_final_fn)
    full_val_output_df.to_parquet(val_final_path, index=False)
    print(f"✓ Saved FINAL validation preds: {val_final_fn}")

    # ==== STEP 2: SUPER TEST ====
    if len(super_test_df) > 0:
        super_interm_fn = f"super_test_intermediate_embeddings_{dataset_name}_loop{loop_counter}.parquet"
        super_interm_path = os.path.join(SUPER_INT_DIR, super_interm_fn)
        if os.path.exists(super_interm_path):
            super_test_intermediate_df = pd.read_parquet(super_interm_path)
            super_test_concat_emb, super_test_tox_labels = fd.create_dataset_tensors_toxicity_classifier_134(
                super_test_intermediate_df, device)
            with torch.no_grad():
                super_test_tox_logits = tox_classifier(super_test_concat_emb).cpu().numpy()
            for i, col in enumerate(tox_logits_cols):
                super_test_intermediate_df[col] = super_test_tox_logits[:, i]
            super_test_intermediate_df['cond_tox_pred_class'] = np.argmax(super_test_tox_logits, axis=1)
            super_test_intermediate_df['train'] = 0 # All super test

            # Reorder columns: metadata, embeddings, logits/predictions, train
            metadata_cols = ['SMILES_spectra', 'index_id', 'Response', 'Group', 'CE_clean']
            tox_cols = ['tox_level'] + [f'tox_level_{k}' for k in range(5)]
            metadata_cols.extend([col for col in tox_cols if col in super_test_intermediate_df.columns])
            embedding_cols = emb_cols + morgan_cols + filtered_morgan_cols
            prediction_cols = tox_logits_cols + ['cond_tox_pred_class']
            column_order = metadata_cols + embedding_cols + prediction_cols + ['train']
            column_order = [col for col in column_order if col in super_test_intermediate_df.columns]
            super_test_intermediate_df = super_test_intermediate_df[column_order]

            super_final_fn = f"super_test_cond_enc_{dataset_name}_loop{loop_counter}.parquet"
            super_final_path = os.path.join(SUPER_FINAL_DIR, super_final_fn)
            super_test_intermediate_df.to_parquet(super_final_path, index=False)
            print(f"✓ Saved FINAL super test preds: {super_final_fn}")

    # ---- CLEANUP GPU memory (optional) ----
    del x_train_with_ext, x_val_with_ext
    del y_train_emb, y_train_morgan, y_train_filtered_morgan, train_indices_tensor
    del y_val_emb, y_val_morgan, y_val_filtered_morgan, val_indices_tensor
    del train_embeddings_combined, val_embeddings_combined
    del train_pred_chemnet, train_pred_morgan, train_pred_filtered_morgan
    del val_pred_chemnet, val_pred_morgan, val_pred_filtered_morgan
    del train_concat_emb, train_tox_labels, val_concat_emb, val_tox_labels
    del embedding_model, trained_embedding_model, tox_classifier, trained_tox_classifier
    torch.cuda.empty_cache()
print('-'*50)
print('All loops completed.')
