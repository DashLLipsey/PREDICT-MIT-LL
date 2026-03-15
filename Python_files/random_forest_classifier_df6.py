import pandas as pd
import numpy as np
import os, sys
import gc

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'Python_files'))
import functions_enc as f
import function_depot as fd

###### ========== USER SETTINGS ========== #######
dataset_name = 'bin0_1_thresh0_05_df_spectra' 
num_loops = 25

##### ==================== SUPER TEST SET SMILES ==================== #####
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

def response_to_tox_class(response_value):
    """Convert Response value to EPA toxicity class (1-4)"""
    if response_value <= 5:
        return 0
    elif response_value <= 50:
        return 1
    elif response_value <= 500:
        return 2
    elif response_value <= 5000:
        return 3
    else:
        return 4

def add_tox_levels_for_rf(df, response_col='Response'):
    df = df.copy()
    df['Tox_level'] = df[response_col].apply(response_to_tox_class)
    return df

VAL_DIR = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/rf_classifier_loop"
SUPER_DIR = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/rf_classifier_loop_super_test"
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(SUPER_DIR, exist_ok=True)

print(f"--- Random Forest Repeat Training, {num_loops} Run(s) ---")

##### ====== Load data, mappings ====== #####
df6_subset = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet")
df6_spectra = pd.read_parquet("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet")
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes_df6"
dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.parquet")
orig_dataset = pd.read_parquet(dataset_path)
orig_dataset = pd.DataFrame(orig_dataset) if not isinstance(orig_dataset, pd.DataFrame) else orig_dataset

id_to_synthetic = dict(zip(df6_spectra['index_id'], df6_spectra['synthetic'].fillna(0)))
smiles_to_index = dict(zip(orig_dataset['SMILES_spectra'], orig_dataset['index_id']))

# Exclude synthetic from super test
synthetic_index_ids = set(idx for idx, syn in id_to_synthetic.items() if syn == 1)
super_test_smiles_non_synth = [
    smiles for smiles in super_test_smiles
    if smiles_to_index.get(smiles, None) not in synthetic_index_ids
]

for loop_counter in range(num_loops):
    print(f"\n{'='*80}\nLOOP {loop_counter+1}/{num_loops}\n{'='*80}")

    dataset = orig_dataset.copy()
    # Remove non-synth super test from training
    dataset_no_super_test = dataset[~dataset['SMILES_spectra'].isin(super_test_smiles_non_synth)].copy()

    counts = dataset_no_super_test['SMILES_spectra'].value_counts()
    valid_smiles = counts[counts >= 4].index
    filtered_dataset = dataset_no_super_test[dataset_no_super_test['SMILES_spectra'].isin(valid_smiles)].copy()

    # Splitting: all synthetic into train; real is stratified by smiles
    synthetic_mask = filtered_dataset['index_id'].map(lambda idx: id_to_synthetic.get(idx, 0)==1)
    real_mask = ~synthetic_mask
    synthetic_data = filtered_dataset[synthetic_mask].copy()
    real_data = filtered_dataset[real_mask].copy()

    # Split only real as before
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
    # Put all synthetic in train
    train_indices.extend(synthetic_data.index.values)

    train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
    test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)
    train_indices_set = set(train_indices)
    train_data['index'] = range(len(train_data))
    test_data['index'] = range(len(test_data))

    # Process as usual but using EPA classes for y
    train_data_processed = fd.add_response_and_log_response(train_data.copy(), df6_subset, smiles_col='SMILES_spectra')
    train_data_processed = add_tox_levels_for_rf(train_data_processed)
    test_data_processed = fd.add_response_and_log_response(test_data.copy(), df6_subset, smiles_col='SMILES_spectra')
    test_data_processed = add_tox_levels_for_rf(test_data_processed)

    # Feature selection
    # Extract spectra features (columns 1 to -11, as in your NN script)
    feature_cols = train_data_processed.columns[1:-11]
    X_train = train_data_processed[feature_cols].values
    y_train = train_data_processed['Tox_level'].values

    X_test = test_data_processed[feature_cols].values
    y_test = test_data_processed['Tox_level'].values
    # --- Fit RF ---
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=loop_counter + 42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_test_pred = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_macro_f1 = f1_score(y_test, y_test_pred, average='macro')

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")

    # ==== FULL VALIDATION ====
    filtered_dataset_full = filtered_dataset.copy()
    train_indicator_map = {idx: 1 if idx in train_indices_set else 0 for idx in filtered_dataset_full.index}
    filtered_dataset_full = filtered_dataset_full.reset_index(drop=False, names=['original_index'])
    filtered_dataset_full['index'] = range(len(filtered_dataset_full))
    filtered_dataset_full_processed = fd.add_response_and_log_response(filtered_dataset_full.copy(), df6_subset, smiles_col='SMILES_spectra')
    filtered_dataset_full_processed = add_tox_levels_for_rf(filtered_dataset_full_processed)
    filtered_dataset_full_processed['train'] = filtered_dataset_full_processed['original_index'].map(train_indicator_map).fillna(0).astype(int)

    X_full_val = filtered_dataset_full_processed[feature_cols].values
    y_full_val = filtered_dataset_full_processed['Tox_level'].values

    full_val_predictions = rf_model.predict(X_full_val)
    full_val_probabilities = rf_model.predict_proba(X_full_val)
    full_val_accuracy = accuracy_score(y_full_val, full_val_predictions)
    full_val_macro_f1 = f1_score(y_full_val, full_val_predictions, average='macro')
    class_labels = rf_model.classes_

    # Output DataFrame
    full_val_output_df = pd.DataFrame()
    full_val_output_df['rf_predicted_tox_class'] = full_val_predictions
    for i, class_label in enumerate(class_labels):
        full_val_output_df[f'rf_prob_class_{class_label}'] = full_val_probabilities[:, i]

    full_val_output_df['true_tox_class'] = y_full_val
    full_val_output_df['SMILES_spectra'] = filtered_dataset_full_processed['SMILES_spectra'].values
    full_val_output_df['Response'] = filtered_dataset_full_processed['Response'].values
    full_val_output_df['log_response'] = filtered_dataset_full_processed['log_response'].values
    full_val_output_df['index_id'] = filtered_dataset_full_processed['index_id'].values
    full_val_output_df['train'] = filtered_dataset_full_processed['train'].values

    val_out_fn = f"rf_{dataset_name}_loop{loop_counter}.parquet"
    val_out_path = os.path.join(VAL_DIR, val_out_fn)
    full_val_output_df.to_parquet(val_out_path, index=False)
    print(f"✓ Saved validation set predictions: {val_out_fn}")

    # ==== SUPER TEST ====
    super_test_df = dataset[dataset['SMILES_spectra'].isin(super_test_smiles_non_synth)].copy()
    if len(super_test_df) > 0:
        super_test_df['index'] = range(len(super_test_df))
        super_test_processed = fd.add_response_and_log_response(super_test_df.copy(), df6_subset, smiles_col='SMILES_spectra')
        super_test_processed = add_tox_levels_for_rf(super_test_processed)
        super_test_processed = super_test_processed[train_data_processed.columns]
        X_super_test = super_test_processed[feature_cols].values
        y_super_test = super_test_processed['Tox_level'].values
        super_test_predictions = rf_model.predict(X_super_test)
        super_test_probabilities = rf_model.predict_proba(X_super_test)
        super_test_output_df = pd.DataFrame()
        for i, class_label in enumerate(class_labels):
            super_test_output_df[f'rf_prob_class_{class_label}'] = super_test_probabilities[:, i]
        super_test_output_df['rf_predicted_tox_class'] = super_test_predictions
        super_test_output_df['true_tox_class'] = y_super_test
        super_test_output_df['SMILES_spectra'] = super_test_processed['SMILES_spectra'].values
        super_test_output_df['Response'] = super_test_processed['Response'].values
        super_test_output_df['log_response'] = super_test_processed['log_response'].values
        super_test_output_df['index_id'] = super_test_processed['index_id'].values
        super_test_output_df['train'] = 0
        super_fn = f"super_test_rf_{dataset_name}_loop{loop_counter}.parquet"
        super_path = os.path.join(SUPER_DIR, super_fn)
        super_test_output_df.to_parquet(super_path, index=False)
        print(f"✓ Saved super test predictions: {super_fn}")

    # MEMORY CLEANUP
    del X_train, X_test, y_train, y_test
    del X_full_val, y_full_val, full_val_probabilities, full_val_predictions
    if 'X_super_test' in locals():
        del X_super_test, y_super_test, super_test_probabilities, super_test_predictions
    del rf_model
    gc.collect()

print('\nAll loops completed and outputs saved.')

