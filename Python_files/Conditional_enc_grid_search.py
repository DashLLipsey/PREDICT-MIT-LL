
#%% 
# Package imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import seaborn as sns

# from fcd_torch import FCD
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

from fcd_torch import FCD
import rdkit

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import functions_enc as f

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

#%%
def create_dataset_tensors_emb_tox(spectra_dataset, embedding_df, device, start_idx=None, stop_idx=None):

    spectra = spectra_dataset.iloc[:,1:-3]

    # create tensors of spectra, true embeddings, true toxicity values, and chemical name encodings for train and val
    chem_labels = list(spectra_dataset['SMILES_spectra'])
    log_tox_tensor = torch.Tensor(spectra_dataset["log_response"].values).unsqueeze(1).to(device)
    embeddings_tensor = torch.Tensor([embedding_df.loc[embedding_df['SMILES'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
    spectra_tensor = torch.Tensor(spectra.values).to(device)
    spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

    return embeddings_tensor, log_tox_tensor, spectra_tensor, spectra_indices_tensor

# Add the 'Response' and 'log_response' columns 
def add_response_and_log_response(spectra_df, original_df, smiles_col='SMILES_spectra'):
    """
    Adds 'Response' and 'log_response' columns to spectra_df by mapping from original_df using the SMILES column.
    """
    smiles_to_response = original_df.drop_duplicates(subset=smiles_col).set_index(smiles_col)['Response']
    spectra_df['Response'] = spectra_df[smiles_col].map(smiles_to_response)
    spectra_df['log_response'] = np.log(spectra_df['Response'])
    return spectra_df
#$$
from encoder_essentials import Cond_Encoder, train_model_condenc
#%%

# CONDITIONAL ENCODER TRAINING LOOP - Process all grid search datasets
print("=== CONDITIONAL ENCODER GRID SEARCH TRAINING ===")

# Set up device and load ChemNet reference
device = f.set_up_gpu()
name_smiles_embedding_df = pd.read_csv("/home/dlipsey/Research/MITLincolnLabs/MIT_LL_data/ChemNet_of_df3_QQpos_no_repeats.csv")

# Define folders
grid_search_folder = "/home/dlipsey/Research/MITLincolnLabs/MIT_LL_data/grid_search_dataframes"
cond_encoder_folder = "/home/dlipsey/Research/MITLincolnLabs/MIT_LL_data/cond_encoder_grid_search_dataframes"

# Get all dataset files from the grid search folder
dataset_files = [f for f in os.listdir(grid_search_folder) if f.endswith('.pkl') and 'df3_QQpos_spectra' in f]
dataset_names = [f.replace('.pkl', '') for f in dataset_files]

print(f"Found {len(dataset_names)} datasets to process")

# Training parameters
batch_size = 64
epochs = 500
lr = 0.0001
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()
output_size = 513
num_layers = 8

# Storage for conditional encoder results
cond_encoder_results = []

# Loop through each dataset and evaluate toxicity predictions directly
for i, dataset_name in enumerate(sorted(dataset_names), 1):
    print(f"\nProcessing {i}/{len(dataset_names)}: {dataset_name}")
    
    try:
        # Load dataset from pickle file
        dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.pkl")
        dataset = pd.read_pickle(dataset_path)
        
        print(f"Loaded {dataset_name} - Shape: {dataset.shape}")
        
        # Apply same filtering and splitting logic
        counts = dataset['SMILES_spectra'].value_counts()
        valid_smiles = counts[counts >= 4].index
        filtered_dataset = dataset[dataset['SMILES_spectra'].isin(valid_smiles)].copy()
        
        print(f"After filtering (>=4 spectra per SMILES): {filtered_dataset.shape}")
        
        if len(filtered_dataset) < 20:
            print(f"Skipping {dataset_name}: Only {len(filtered_dataset)} samples after filtering")
            continue
        
        train_indices = []
        test_indices = []
        
        np.random.seed(42)
        for smiles, group in filtered_dataset.groupby('SMILES_spectra'):
            idx = group.index.tolist()
            n = len(idx)
            np.random.shuffle(idx)
            split = n // 2
            test_idx = idx[:split]
            train_idx = idx[split:]
            train_indices.extend(train_idx)
            test_indices.extend(test_idx)
        
        train_data = filtered_dataset.loc[train_indices].reset_index(drop=True)
        test_data = filtered_dataset.loc[test_indices].reset_index(drop=True)
        
        # Add index column
        train_data['index'] = train_data.index
        test_data['index'] = test_data.index
        
        # Add response and log response
        train_data_copy = train_data.copy()
        test_data_copy = test_data.copy()
        
        train_data_processed = add_response_and_log_response(train_data_copy, df3_QQpos, smiles_col='SMILES_spectra')
        test_data_processed = add_response_and_log_response(test_data_copy, df3_QQpos, smiles_col='SMILES_spectra')
        
        # Create tensors
        y_train_emb, y_train_tox, x_train, train_indices_tensor = create_dataset_tensors_emb_tox(
            train_data_processed, name_smiles_embedding_df, device, start_idx=1, stop_idx=-3)
        
        y_val_emb, y_val_tox, x_val, val_indices_tensor = create_dataset_tensors_emb_tox(
            test_data_processed, name_smiles_embedding_df, device, start_idx=1, stop_idx=-3)
        
        # Create data loaders
        train_dataset = TensorDataset(x_train, y_train_emb, y_train_tox, train_indices_tensor)
        val_dataset = TensorDataset(x_val, y_val_emb, y_val_tox, val_indices_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create and train conditional encoder
        cond_encoder_current = Cond_Encoder(input_size=x_train.shape[1], output_size=output_size, num_layers=num_layers).to(device)
        
        # Train conditional encoder
        trained_cond_encoder = train_model_condenc(
            model=cond_encoder_current,
            train_data=train_loader,
            val_data=val_loader,
            epochs=epochs,
            learning_rate=lr,
            criterion1=criterion1,
            criterion2=criterion2,
            device=device
        )
        
        # Generate predictions and evaluate toxicity prediction performance
        cond_encoder_current.eval()
        with torch.no_grad():
            # Train predictions
            train_predictions = cond_encoder_current(x_train).cpu().numpy()
            train_tox_predictions = train_predictions[:, 512:]  # Last column (toxicity)
            
            # Test predictions
            test_predictions = cond_encoder_current(x_val).cpu().numpy()
            test_tox_predictions = test_predictions[:, 512:]  # Last column (toxicity)
        
        # Evaluate toxicity prediction performance directly
        train_tox_true = y_train_tox.cpu().numpy().flatten()
        test_tox_true = y_val_tox.cpu().numpy().flatten()
        train_tox_pred = train_tox_predictions.flatten()
        test_tox_pred = test_tox_predictions.flatten()
        
        # Calculate R² for toxicity prediction
        train_r2 = r2_score(train_tox_true, train_tox_pred)
        test_r2 = r2_score(test_tox_true, test_tox_pred)
        
        # Calculate percent errors (undo log transform)
        train_response_true = np.exp(train_tox_true)
        train_response_pred = np.exp(train_tox_pred)
        test_response_true = np.exp(test_tox_true)
        test_response_pred = np.exp(test_tox_pred)
        
        train_median_percent_error = 100 * np.median(np.abs(train_response_pred - train_response_true) / train_response_true)
        test_median_percent_error = 100 * np.median(np.abs(test_response_pred - test_response_true) / test_response_true)
        train_mean_percent_error = 100 * np.mean(np.abs(train_response_pred - train_response_true) / train_response_true)
        test_mean_percent_error = 100 * np.mean(np.abs(test_response_pred - test_response_true) / test_response_true)
        
        # Store results for heatmap analysis
        cond_encoder_results.append({
            'Dataset': dataset_name,
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Train_Median_Percent_Error': train_median_percent_error,
            'Test_Median_Percent_Error': test_median_percent_error,
            'Train_Mean_Percent_Error': train_mean_percent_error,
            'Test_Mean_Percent_Error': test_mean_percent_error,
            'Samples': len(filtered_dataset),
            'Train_Samples': len(train_data_processed),
            'Test_Samples': len(test_data_processed)
        })
        
        print(f"✓ Toxicity Prediction Performance:")
        print(f"   Test R²: {test_r2:.4f}")
        print(f"   Test Median % Error: {test_median_percent_error:.1f}%")
        
    except Exception as e:
        print(f"✗ Error processing {dataset_name}: {str(e)}")
        continue

print(f"\n=== CONDITIONAL ENCODER EVALUATION COMPLETED ===")
print(f"Successfully processed {len(cond_encoder_results)} datasets")

# Convert results to DataFrames for heatmap analysis
df_cond_r2_results = pd.DataFrame(cond_encoder_results)
df_cond_percent_error_results = pd.DataFrame(cond_encoder_results)

print("\nConditional Encoder Results Summary:")
print(f"Mean Test R²: {df_cond_r2_results['Test_R2'].mean():.4f}")
print(f"Mean Test Median % Error: {df_cond_percent_error_results['Test_Median_Percent_Error'].mean():.2f}%")

#%%
# CREATE HEATMAPS FOR CONDITIONAL ENCODER RESULTS
# Parse dataset names to extract bin sizes and thresholds
def parse_cond_dataset_name(dataset_name):
    """Extract bin size and threshold from dataset name"""
    if 'thresh_zero' in dataset_name:
        bin_part = dataset_name.split('_thresh_zero')[0].replace('bin', '')
        bin_size = float(bin_part.replace('_', '.'))
        threshold = 0.0
    else:
        parts = dataset_name.split('_thresh')
        bin_part = parts[0].replace('bin', '')
        bin_size = float(bin_part.replace('_', '.'))
        
        thresh_part = parts[1].split('_df3_QQpos_spectra')[0]
        threshold = float(thresh_part.replace('_', '.'))
    
    return bin_size, threshold

# Add bin size and threshold columns
bin_sizes = []
thresholds = []

for dataset_name in df_cond_r2_results['Dataset']:
    bin_size, threshold = parse_cond_dataset_name(dataset_name)
    bin_sizes.append(bin_size)
    thresholds.append(threshold)

df_cond_r2_results['BinSize'] = bin_sizes
df_cond_r2_results['Threshold'] = thresholds
df_cond_percent_error_results['BinSize'] = bin_sizes
df_cond_percent_error_results['Threshold'] = thresholds

# Create pivot tables for heatmaps
thresholds_subset = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
bins_subset = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]

cond_r2_pivot = df_cond_r2_results.pivot(index='BinSize', columns='Threshold', values='Test_R2')
cond_median_error_pivot = df_cond_percent_error_results.pivot(index='BinSize', columns='Threshold', values='Test_Median_Percent_Error')

# Reindex to show all combinations
cond_r2_pivot = cond_r2_pivot.reindex(columns=thresholds_subset, index=bins_subset)
cond_median_error_pivot = cond_median_error_pivot.reindex(columns=thresholds_subset, index=bins_subset)

# Create heatmaps
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# R² Heatmap
sns.heatmap(cond_r2_pivot, annot=True, fmt='.3f', cmap='Blues', ax=axes[0],
            cbar_kws={'label': 'Test R²'})
axes[0].set_title('Conditional Encoder: Test R² by Bin Size and Threshold', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Threshold Value', fontsize=12)
axes[0].set_ylabel('Bin Size', fontsize=12)
axes[0].invert_yaxis()

# Median Percent Error Heatmap
sns.heatmap(cond_median_error_pivot, annot=True, fmt='.1f', cmap='Reds', ax=axes[1],
            cbar_kws={'label': 'Median % Error'})
axes[1].set_title('Conditional Encoder: Median Absolute Percent Error', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Threshold Value', fontsize=12)
axes[1].set_ylabel('Bin Size', fontsize=12)
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

print("\n=== CONDITIONAL ENCODER BEST PERFORMANCE ===")
best_r2_idx = df_cond_r2_results['Test_R2'].idxmax()
best_r2_dataset = df_cond_r2_results.loc[best_r2_idx, 'Dataset']
best_r2_value = df_cond_r2_results.loc[best_r2_idx, 'Test_R2']
print(f"Best R²: {best_r2_value:.4f} from {best_r2_dataset}")

best_error_idx = df_cond_percent_error_results['Test_Median_Percent_Error'].idxmin()
best_error_dataset = df_cond_percent_error_results.loc[best_error_idx, 'Dataset']
best_error_value = df_cond_percent_error_results.loc[best_error_idx, 'Test_Median_Percent_Error']
print(f"Best Median % Error: {best_error_value:.2f}% from {best_error_dataset}")
# %%