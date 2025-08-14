# Package imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import os
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
from collections import Counter

def create_dataset_tensors_emb_tox(spectra_dataset, embedding_df, device, start_idx=None, stop_idx=None):

    spectra = spectra_dataset.iloc[:,1:-3]
    
    # Force conversion to float64 for all spectra columns
    spectra_numeric = spectra.astype(float)

    # create tensors of spectra, true embeddings, true toxicity values, and chemical name encodings for train and val
    chem_labels = list(spectra_dataset['SMILES_spectra'])
    
    # Ensure log_response is float
    log_tox_tensor = torch.Tensor(spectra_dataset["log_response"].values.astype(float)).unsqueeze(1).to(device)
    
    # More efficient embedding lookup - avoid list comprehension
    # Create a lookup dictionary for faster access
    embedding_dict = {}
    for _, row in embedding_df.iterrows():
        smiles = row['SMILES']
        embedding_dict[smiles] = row.iloc[1:].values.astype(float)
    
    # Build embeddings array efficiently
    embeddings_list = []
    for chem_name in chem_labels:
        if chem_name in embedding_dict:
            embeddings_list.append(embedding_dict[chem_name])
        else:
            # Fallback - use zeros if embedding not found
            embeddings_list.append(np.zeros(512))
    
    embeddings_array = np.array(embeddings_list)
    embeddings_tensor = torch.Tensor(embeddings_array).to(device)
    
    # Use the float-converted spectra
    spectra_tensor = torch.Tensor(spectra_numeric.values.astype(float)).to(device)
    
    # Ensure index is float
    spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy().astype(float)).to(device)

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
batch_size = 128
epochs=350
lr=0.0001
criterion1=nn.MSELoss()
criterion2=nn.MSELoss()
output_size = 513
num_layers = 8

#%%
# Encoder architecture (With Validation Set)
class Cond_Encoder(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super().__init__()
        layers = []
        layer_sizes = np.linspace(input_size, output_size, num_layers + 1, dtype=int)
        for i in range(num_layers):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < num_layers - 1:
                layers.append(nn.LeakyReLU(inplace=True))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

def train_model_condenc(model, train_data, val_data, epochs, learning_rate, criterion1, criterion2, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch, true_embeddings, true_log_tox, _ in train_data:
            batch = batch.to(device)
            true_embeddings = true_embeddings.to(device)
            true_log_tox = true_log_tox.to(device)

            optimizer.zero_grad()
            batch_predicted_combined = model(batch) # Take the first 512 for criterion 1 and the last for criterion 2, look up to make sure i only apply the loss to the subset of the model
            
            # Embedding Loss
            batch_predicted_embeddings = batch_predicted_combined[:, :512] # First 512 columns
            loss1 = criterion1(batch_predicted_embeddings, true_embeddings) # loss1 (embedding loss)
            # Response Loss
            batch_predicted_log_tox = batch_predicted_combined[:, 512:] # Last column
            loss2 = criterion2(batch_predicted_log_tox, true_log_tox) # loss2 (toxicity loss)
            
            total_loss = loss1 + loss2
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        average_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():  # Condense this as we did above for symmetry tho not needed without loss.backward command
            for val_batch, val_true_embeddings, val_true_tox, _ in val_data:
                val_batch = val_batch.to(device)
                val_true_embeddings = val_true_embeddings.to(device)
                val_true_tox = val_true_tox.to(device)

                val_batch_predicted = model(val_batch)
                val_batch_predicted_embeddings = val_batch_predicted[:, :512]

                val_loss = criterion1(val_batch_predicted_embeddings, val_true_embeddings)
                val_loss += loss1.item()

                val_batch = val_batch.to(device)
                val_true_embeddings = val_true_tox.to(device)

                val_batch_predicted_tox = val_batch_predicted[:, 512:]

                val_loss = criterion2(val_batch_predicted_tox, val_true_tox)
                val_loss += loss2.item()
        average_val_loss = val_loss / len(val_loader)

        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'   Training loss: {average_train_loss}')
        print(f'   Validation loss: {average_val_loss}')

    return model

# CONDITIONAL ENCODER TRAINING LOOP - Process all grid search datasets
print("=== CONDITIONAL ENCODER GRID SEARCH TRAINING ===")

# Set up device and load ChemNet reference
device = f.set_up_gpu()
name_smiles_embedding_df = pd.read_csv("/home/dlipsey/MITLincolnLabs/MIT_LL_data/ChemNet_of_df3_QQpos_no_repeats.csv")

# Load the original dataset for response mapping
df3_QQpos = pd.read_csv("/home/dlipsey/MITLincolnLabs/MIT_LL_data/df3_QQpos.csv")  # You need to specify this path

# Define folders
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes"

# Get all dataset files from the grid search folder
dataset_files = [f for f in os.listdir(grid_search_folder) if f.endswith('.pkl') and 'df3_QQpos_spectra' in f]

# Define allowed bin sizes (exclude 0.01)
allowed_bin_prefixes = ['bin0_05_', 'bin0_1_', 'bin0_5_', 'bin1_', 'bin2_', 'bin5_', 'bin10_', 'bin25_', 'bin50_', 'bin100_', 'bin200_', 'bin500_', 'bin1000_']

# Filter dataset files to only include allowed bin sizes
dataset_files = [f for f in dataset_files if any(f.startswith(prefix) for prefix in allowed_bin_prefixes)]

dataset_names = [f.replace('.pkl', '') for f in dataset_files]

print(f"Found {len(dataset_names)} datasets to process (excluding bin size 0.01)")

# Storage for conditional encoder results
cond_encoder_results = []

# Loop through each dataset and evaluate toxicity predictions from element 512 (0-indexed)
for i, dataset_name in enumerate(sorted(dataset_names), 1):
    print(f"\nProcessing {i}/{len(dataset_names)}: {dataset_name}")
    
    try:
        # Load dataset from pickle file
        dataset_path = os.path.join(grid_search_folder, f"{dataset_name}.pkl")
        dataset = pd.read_pickle(dataset_path)

        # Convert to DataFrame if it's not already one
        if not isinstance(dataset, pd.DataFrame):
            print(f"Converting {type(dataset)} to DataFrame")
            dataset = pd.DataFrame(dataset)

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
        
        # Generate predictions from the 513th element (index 512) - toxicity regression output
        cond_encoder_current.eval()
        with torch.no_grad():
            # Train predictions - extract toxicity prediction from element 512
            train_predictions = cond_encoder_current(x_train).cpu().numpy()
            train_tox_predictions = train_predictions[:, 512]  # Element 512 (0-indexed for 513th element)
            
            # Test predictions - extract toxicity prediction from element 512
            test_predictions = cond_encoder_current(x_val).cpu().numpy()
            test_tox_predictions = test_predictions[:, 512]  # Element 512 (0-indexed for 513th element)
        
        # Get true toxicity values
        train_tox_true = y_train_tox.cpu().numpy().flatten()
        test_tox_true = y_val_tox.cpu().numpy().flatten()
        train_tox_pred = train_tox_predictions.flatten()
        test_tox_pred = test_tox_predictions.flatten()
        
        # Calculate percent errors (undo log transform to get back to original response scale)
        train_response_true = np.exp(train_tox_true)
        train_response_pred = np.exp(train_tox_pred)
        test_response_true = np.exp(test_tox_true)
        test_response_pred = np.exp(test_tox_pred)
        
        # Calculate absolute percent errors
        train_median_percent_error = 100 * np.median(np.abs(train_response_pred - train_response_true) / train_response_true)
        test_median_percent_error = 100 * np.median(np.abs(test_response_pred - test_response_true) / test_response_true)
        train_mean_percent_error = 100 * np.mean(np.abs(train_response_pred - train_response_true) / train_response_true)
        test_mean_percent_error = 100 * np.mean(np.abs(test_response_pred - test_response_true) / test_response_true)
        
        # Add conditional encoder predictions to the dataframes
        # Generate 513-dimensional outputs for ALL data (train + test combined)
        full_data_processed = pd.concat([train_data_processed, test_data_processed], ignore_index=True)
        
        # Create tensors for full dataset
        y_full_emb, y_full_tox, x_full, full_indices_tensor = create_dataset_tensors_emb_tox(
            full_data_processed, name_smiles_embedding_df, device, start_idx=1, stop_idx=-3)
        
        # Generate conditional encoder outputs
        cond_encoder_current.eval()
        with torch.no_grad():
            full_cond_outputs = cond_encoder_current(x_full).cpu().numpy()
        
        # Create output DataFrame with 513 dimensions + metadata
        emb_cols = [f'cond_emb_{j}' for j in range(512)]  # ChemNet embedding dimensions
        
        output_df = pd.DataFrame(full_cond_outputs[:, :512], columns=emb_cols)
        output_df['cond_tox_pred'] = full_cond_outputs[:, 512]  # Toxicity prediction
        
        # Add metadata
        output_df['SMILES_spectra'] = full_data_processed['SMILES_spectra'].values
        output_df['Response'] = full_data_processed['Response'].values
        output_df['log_response'] = full_data_processed['log_response'].values
        output_df['index_id'] = full_data_processed['index'].values
        
        # Store results for heatmap analysis (only percent errors)
        cond_encoder_results.append({
            'Dataset': dataset_name,
            'Train_Median_Percent_Error': train_median_percent_error,
            'Test_Median_Percent_Error': test_median_percent_error,
            'Train_Mean_Percent_Error': train_mean_percent_error,
            'Test_Mean_Percent_Error': test_mean_percent_error,
            'Samples': len(filtered_dataset),
            'Train_Samples': len(train_data_processed),
            'Test_Samples': len(test_data_processed)
        })
        
        # Parse bin size and threshold from dataset name for filename
        if 'thresh_zero' in dataset_name:
            bin_part = dataset_name.split('_thresh_zero')[0]  # Keep bin0_1 format
            threshold_part = "thresh_zero"
        else:
            parts = dataset_name.split('_thresh')
            bin_part = parts[0]  # Keep bin0_1 format
            
            thresh_part = parts[1].split('_df3_QQpos_spectra')[0]
            threshold_part = f"thresh{thresh_part}"  # Keep thresh0_001 format
        
        # Save conditional encoder outputs (513 dimensions + 4 metadata = 517 columns total)
        output_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/cond_enc_outputs"
        os.makedirs(output_folder, exist_ok=True)
        
        predictions_filename = f"cond_enc_{bin_part}_{threshold_part}_df3_QQpos_spectra.pkl"
        predictions_path = os.path.join(output_folder, predictions_filename)
        output_df.to_pickle(predictions_path)

        print(f"Toxicity Prediction Performance (from 513th encoder output):")
        print(f"Test Median % Error: {test_median_percent_error:.1f}%")
        print(f"Test Mean % Error: {test_mean_percent_error:.1f}%")
        print(f"Saved prediction dataframe to {predictions_filename}")

        # Clear GPU memory after each dataset
        del x_train, x_val, y_train_emb, y_train_tox, y_val_emb, y_val_tox
        del train_indices_tensor, val_indices_tensor
        del cond_encoder_current, trained_cond_encoder
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        # Clear memory on error too
        torch.cuda.empty_cache()
        continue

print(f"\n=== CONDITIONAL ENCODER EVALUATION COMPLETED ===")
print(f"Successfully processed {len(cond_encoder_results)} datasets")

# Convert results to DataFrame for heatmap analysis
df_cond_percent_error_results = pd.DataFrame(cond_encoder_results)

print("\nConditional Encoder Results Summary:")
print(f"Mean Test Median % Error: {df_cond_percent_error_results['Test_Median_Percent_Error'].mean():.2f}%")
print(f"Mean Test Mean % Error: {df_cond_percent_error_results['Test_Mean_Percent_Error'].mean():.2f}%")

#%%
# CREATE HEATMAPS FOR CONDITIONAL ENCODER RESULTS - PERCENT ERRORS ONLY
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

for dataset_name in df_cond_percent_error_results['Dataset']:
    bin_size, threshold = parse_cond_dataset_name(dataset_name)
    bin_sizes.append(bin_size)
    thresholds.append(threshold)

df_cond_percent_error_results['BinSize'] = bin_sizes
df_cond_percent_error_results['Threshold'] = thresholds

# Create pivot tables for heatmaps
thresholds_subset = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
bins_subset = [0.05, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]

cond_median_error_pivot = df_cond_percent_error_results.pivot(index='BinSize', columns='Threshold', values='Test_Median_Percent_Error')
cond_mean_error_pivot = df_cond_percent_error_results.pivot(index='BinSize', columns='Threshold', values='Test_Mean_Percent_Error')

# Reindex to show all combinations
cond_median_error_pivot = cond_median_error_pivot.reindex(columns=thresholds_subset, index=bins_subset)
cond_mean_error_pivot = cond_mean_error_pivot.reindex(columns=thresholds_subset, index=bins_subset)

# Create separate heatmaps
# Median Percent Error Heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.heatmap(cond_median_error_pivot, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax,
            cbar_kws={'label': 'Median % Error'})
ax.set_title('Conditional Encoder: Median Absolute Percent Error', fontsize=14, fontweight='bold')
ax.set_xlabel('Threshold Value', fontsize=12)
ax.set_ylabel('Bin Size', fontsize=12)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("/home/dlipsey/MITLincolnLabs/Figures/Conditional_encoder_median_percent_error")
plt.show()

# Mean Percent Error Heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.heatmap(cond_mean_error_pivot, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax,
            cbar_kws={'label': 'Mean % Error'})
ax.set_title('Conditional Encoder: Mean Absolute Percent Error', fontsize=14, fontweight='bold')
ax.set_xlabel('Threshold Value', fontsize=12)
ax.set_ylabel('Bin Size', fontsize=12)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("/home/dlipsey/MITLincolnLabs/Figures/Conditional_encoder_mean_percent_error")
plt.show()

print("\n=== CONDITIONAL ENCODER BEST PERFORMANCE ===")
best_median_error_idx = df_cond_percent_error_results['Test_Median_Percent_Error'].idxmin()
best_median_error_dataset = df_cond_percent_error_results.loc[best_median_error_idx, 'Dataset']
best_median_error_value = df_cond_percent_error_results.loc[best_median_error_idx, 'Test_Median_Percent_Error']
print(f"Best Median % Error: {best_median_error_value:.2f}% from {best_median_error_dataset}")

best_mean_error_idx = df_cond_percent_error_results['Test_Mean_Percent_Error'].idxmin()
best_mean_error_dataset = df_cond_percent_error_results.loc[best_mean_error_idx, 'Dataset']
best_mean_error_value = df_cond_percent_error_results.loc[best_mean_error_idx, 'Test_Mean_Percent_Error']
print(f"Best Mean % Error: {best_mean_error_value:.2f}% from {best_mean_error_dataset}")