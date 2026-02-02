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
dataset_name = 'bin0_1_thresh0_5_df_spectra'  # 'bin1_thresh0_05_df_spectra'
num_loops = 25

VAL_DIR  = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/regular_classifier_loop"
SUPER_DIR = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/regular_classifier_loop_super_test"
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(SUPER_DIR, exist_ok=True)

# Super test SMILES (copy full list from your code)
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
num_layers = 8
batch_size = 256
epochs = 500
lr = 0.0001
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

    dataset = orig_dataset.copy()
    dataset_no_super_test = dataset[~dataset['SMILES_spectra'].isin(super_test_smiles)].copy()

    if 'Group' not in dataset_no_super_test.columns:
        dataset_no_super_test['Group'] = dataset_no_super_test['index_id'].map(id_to_group).fillna('Unknown')
    if 'CE_clean' not in dataset_no_super_test.columns:
        dataset_no_super_test['CE_clean'] = dataset_no_super_test['index_id'].map(id_to_ce_clean).fillna('Unknown')
    counts = dataset_no_super_test['SMILES_spectra'].value_counts()
    valid_smiles = counts[counts >= 4].index
    filtered_dataset = dataset_no_super_test[dataset_no_super_test['SMILES_spectra'].isin(valid_smiles)].copy()

    # New random split per loop
    smiles_groups = filtered_dataset.groupby('SMILES_spectra')
    train_indices, test_indices = [], []
    np.random.seed(loop_counter + 42)
    for smiles, group in smiles_groups:
        idx = group.index.values
        n = len(idx)
        np.random.shuffle(idx)
        split = n // 2
        test_indices.extend(idx[:split])
        train_indices.extend(idx[split:])
    train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
    test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)
    train_indices_set = set(train_indices)
    train_data['index'] = range(len(train_data))
    test_data['index'] = range(len(test_data))

    # Preprocess, etc
    train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
    train_data_processed = fd.add_tox_levels(train_data_processed)
    test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')
    test_data_processed = fd.add_tox_levels(test_data_processed)

    x_train, y_train_tox, train_indices_tensor = fd.create_dataset_tensors_direct_toxicity_e1e2(
        train_data_processed, device, start_idx=1, stop_idx=-11)
    x_val, y_val_tox, val_indices_tensor = fd.create_dataset_tensors_direct_toxicity_e1e2(
        test_data_processed, device, start_idx=1, stop_idx=-11)
    actual_input_size = x_train.shape[1]
    direct_tox_model = fd.Direct_Toxicity_Encoder(
            input_size=actual_input_size,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout_rate=0.2
    ).to(device)
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
    filtered_dataset_full['index'] = range(len(filtered_dataset_full))
    filtered_dataset_full_processed = fd.add_response_and_log_response(filtered_dataset_full.copy(), df6_subset, smiles_col='SMILES_spectra')
    filtered_dataset_full_processed = fd.add_tox_levels(filtered_dataset_full_processed)
    filtered_dataset_full_processed['train'] = filtered_dataset_full_processed['original_index'].map(train_indicator_map).fillna(0).astype(int)
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
    full_val_output_df['log_response'] = filtered_dataset_full_processed['log_response'].values
    full_val_output_df['index_id'] = filtered_dataset_full_processed['index'].values
    full_val_output_df['train'] = filtered_dataset_full_processed['train'].values

    # SAVE MAIN VALIDATION OUT
    val_out_fn = f"direct_tox_{dataset_name}_loop{loop_counter}.parquet"
    val_out_path = os.path.join(VAL_DIR, val_out_fn)
    full_val_output_df.to_parquet(val_out_path, index=False)
    print(f"✓ Saved validation set predictions: {val_out_fn}")

    # ==== SUPER TEST ====
    super_test_df = dataset[dataset['SMILES_spectra'].isin(super_test_smiles)].copy()
    if len(super_test_df) > 0:
        if 'Group' not in super_test_df.columns:
            super_test_df['Group'] = super_test_df['index_id'].map(id_to_group).fillna('Unknown')
        if 'CE_clean' not in super_test_df.columns:
            super_test_df['CE_clean'] = super_test_df['index_id'].map(id_to_ce_clean).fillna('Unknown')
        super_test_df['index'] = range(len(super_test_df))
        super_test_processed = fd.add_response_and_log_response(super_test_df.copy(), df6_subset, smiles_col='SMILES_spectra')
        super_test_processed = fd.add_tox_levels(super_test_processed)
        # Ensure super test features match training features
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
        super_test_output_df['log_response'] = super_test_processed['log_response'].values
        super_test_output_df['index_id'] = super_test_processed['index'].values
        super_test_output_df['train'] = 0
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