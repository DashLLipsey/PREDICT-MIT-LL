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
threshold = 0.05  # 0.5 and 0.05
dataset_name = 'bin1_thresh0_05_df_spectra'  # <-- must match parquet file in grid_search_folder
num_loops = 3       # how many repeated train/val splits & models

# --- Output folders (all must exist or will be made) ---
VAL_INT_DIR  = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/2step_synth_abl_134_loop_intermediate"
VAL_FINAL_DIR = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/2step_synth_abl_134_loop"
SUPER_INT_DIR  = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/2step_synth_abl_134_loop_super_test_intermediate"
SUPER_FINAL_DIR = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/2step_synth_abl_134_loop_super_test"
for d in [VAL_INT_DIR, VAL_FINAL_DIR, SUPER_INT_DIR, SUPER_FINAL_DIR]:
    os.makedirs(d, exist_ok=True)

# Step 1 embedding inputs
name_smiles_embedding_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_chemnet.parquet")
morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_morganfp.parquet")
filtered_morgan_df = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_filtered_morganfp.parquet")

df6_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet")
df6_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet")
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes_df6"
dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.parquet")

# Super test SMILES - (copy full list from your script)
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

#### ==== Model params ==== ####
embedding_num_layers = 6
embedding_batch_size = 256
embedding_epochs = 500
embedding_lr = 0.0001
lambda1 = 5
lambda3 = 1
lambda4 = 3
dropout1 = 0.35

input_length=4608
tox_num_layers = 4
tox_batch_size = 256
tox_epochs = 500
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

# LOAD original dataset just once
orig_dataset = pd.read_parquet(dataset_path)
orig_dataset = pd.DataFrame(orig_dataset) if not isinstance(orig_dataset, pd.DataFrame) else orig_dataset

for loop_counter in range(num_loops):
    print(f'\n========== LOOP {loop_counter+1}/{num_loops} ==========')

    # SPLIT, FILTER, MAP GROUP/CLEAN (re-randomize every loop!)
    dataset = orig_dataset.copy()
    dataset_no_super_test = dataset[~dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
    
    if 'Group' not in dataset_no_super_test.columns:
        dataset_no_super_test['Group'] = dataset_no_super_test['index_id'].map(id_to_group).fillna('Unknown')
    if 'CE_clean' not in dataset_no_super_test.columns:
        dataset_no_super_test['CE_clean'] = dataset_no_super_test['index_id'].map(id_to_ce_clean).fillna('Unknown')
    
    # === OPTION 1: Filter SMILES based on real spectra only (RECOMMENDED) ===
    # Filter for valid SMILES BEFORE removing synthetic, but count only real spectra
    synthetic_ids = set(df6_spectra.loc[df6_spectra['synthetic'] == 1, 'index_id'])
    real_data_temp = dataset_no_super_test[~dataset_no_super_test['index_id'].isin(synthetic_ids)].copy()
    # Count only REAL spectra per SMILES
    counts = real_data_temp['SMILES_spectra'].value_counts()
    valid_smiles = counts[counts >= 4].index
    # Filter full dataset to those SMILES, THEN remove synthetic
    filtered_dataset = dataset_no_super_test[dataset_no_super_test['SMILES_spectra'].isin(valid_smiles)].copy()
    before_synth = len(filtered_dataset)
    filtered_dataset = filtered_dataset[~filtered_dataset['index_id'].isin(synthetic_ids)].copy()
    after_synth = len(filtered_dataset)
    print(f"Removed {before_synth - after_synth} samples with synthetic==1")
    
    # === OPTION 2: Old method - remove synthetic first, then filter ===
    # synthetic_ids = set(df6_spectra.loc[df6_spectra['synthetic'] == 1, 'index_id'])
    # before_synth = len(dataset_no_super_test)
    # dataset_no_super_test = dataset_no_super_test[~dataset_no_super_test['index_id'].isin(synthetic_ids)].copy()
    # after_synth = len(dataset_no_super_test)
    # print(f"Removed {before_synth - after_synth} samples with synthetic==1")
    # counts = dataset_no_super_test['SMILES_spectra'].value_counts()
    # valid_smiles = counts[counts >= 4].index
    # filtered_dataset = dataset_no_super_test[dataset_no_super_test['SMILES_spectra'].isin(valid_smiles)].copy()

    # --- Simple 50-50 random split ---
    np.random.seed(loop_counter + 42)
    all_indices = filtered_dataset.index.values
    np.random.shuffle(all_indices)
    split_point = len(all_indices) // 2
    train_indices = all_indices[split_point:]
    test_indices = all_indices[:split_point]
    
    train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
    test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)

    # Keep original index_id, add temp_index only for internal processing if needed
    # DO NOT overwrite index_id - it tracks specific spectra across the pipeline
    train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
    train_data_processed = fd.add_tox_levels(train_data_processed)
    # Add 'index' column for the tensor function, but use index_id values
    train_data_processed['index'] = train_data_processed['index_id']
    
    test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')
    test_data_processed = fd.add_tox_levels(test_data_processed)
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

    trained_embedding_model, *_ = fd.train_model_condenc_134e1e2(
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
    embedding_model.eval() 
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
        # Add 'index' column for the tensor function, but use index_id values
        super_test_processed['index'] = super_test_processed['index_id']

        # Ensure super test processed columns match train_data_processed columns
        super_test_processed = super_test_processed[train_data_processed.columns]

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

    
    # tox_classifier = fd.ToxicityClassifier_134_2(num_classes=tox_num_classes,
    #                                              input_length=4608,
    #                                              dropout_rate=dropout2,
    #                                              layer1_size=layer1_size).to(device)

    # tox_classifier = fd.ToxicityClassifier_134_3(num_classes=tox_num_classes,
    #                                              input_length=4608,
    #                                              dropout_rate=dropout2,
    #                                              layer1_size=layer1_size,
    #                                              layer2_size=layer2_size).to(device)
    
    # tox_classifier = fd.ToxicityClassifier_134_4(num_classes=tox_num_classes,
    #                                              input_length=4608,
    #                                              dropout_rate=dropout2,
    #                                              layer1_size=layer1_size,
    #                                              layer2_size=layer2_size,
    #                                              layer3_size=layer3_size).to(device)  
    
    
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