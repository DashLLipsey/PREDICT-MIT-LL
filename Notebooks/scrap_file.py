# Basic Package Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, r2_score

# imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Non-basic package imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import requests

# Packages I don't understand
from fcd_torch import FCD
import rdkit
from collections import Counter
import gc
import pickle

# Add the Python_files directory to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'Python_files'))

# Now you can import your modules
# import functions_enc as f
import function_depot as fd

def parse_dataset_name(dataset_name):
    """
    Extract bin size and threshold from dataset name.
    
    Args:
        dataset_name (str): Dataset name in format like 'bin0_1_thresh0_1_df_spectra' or 'bin1_thresh_zero_df_spectra'
    
    Returns:
        tuple: (bin_size, threshold) as floats
    
    Examples:
        parse_dataset_name('bin0_1_thresh0_1_df_spectra') -> (0.1, 0.1)
        parse_dataset_name('bin1_thresh_zero_df_spectra') -> (1.0, 0.0)
        parse_dataset_name('bin0_05_thresh100_df_spectra') -> (0.05, 100.0)
    """
    # Handle thresh_zero case (no threshold applied)
    if 'thresh_zero' in dataset_name:
        # Extract bin size
        bin_part = dataset_name.split('_thresh_zero')[0].replace('bin', '')
        bin_size = float(bin_part.replace('_', '.'))
        threshold = 0.0
    else:
        # Extract bin size and threshold for normal cases
        parts = dataset_name.split('_thresh')
        bin_part = parts[0].replace('bin', '')
        bin_size = float(bin_part.replace('_', '.'))
        
        thresh_part = parts[1].split('_df_spectra')[0]
        threshold = float(thresh_part.replace('_', '.'))
    
    return bin_size, threshold

# We are working with the June 25 dataset, with the Morgan Fingerprints and cannonical SMILES included
# df5 = pd.read_csv("/home/dlipsey/MITLincolnLabs/MIT_LL_data/MIT_LL_data5.csv")
# print(df5.shape)
# df5.head()

# First order of business is to standardize our SMILES column. We want to use canonical smiles rather than SMILES_spectra but 
# # we will keep the column name SMILES_spectra for consistency with previous code
# df5 = df5.drop('SMILES_spectra', axis=1) # Drop
# df5 = df5.rename(columns={'canonical_smiles': 'SMILES_spectra'}) # Rename
# cols = df5.columns.tolist()
# cols.remove('SMILES_spectra') 
# df5 = df5[['SMILES_spectra'] + cols] # Move to front

# # Next we want to standardize the Ionization column
# # print(df5["Ionization_Mode"].unique()) # Check unique values
# df5["Ionization_Mode"] = df5["Ionization_Mode"].replace("'Positive'", "'positive'") # Fix capitaliztion
# df5 = df5[df5["Ionization_Mode"] != "'N/A'"] # Remove N/A 
# # print(df5["Ionization_Mode"].unique()) # Check unique values

# # Remove single quotes from all columns
# df5 = df5.applymap(lambda x: x.replace("'", "") if isinstance(x, str) else x)

# # Select specific groups for subset
# selected_groups = ['Q-Orbitrap-positive', 'Q-TOF-positive', 'LTQ-Orbitrap-positive']

# # Create subset with only selected groups
# df5_subset = df5[df5['Group'].isin(selected_groups)]

# print(df5_subset.shape)
# df5_subset.head()

# # SPECTRA DATAFRAME
# # Create dataframe with spectra using spectrum_string_to_dataframe
# df5_spectra = fd.spectrum_string_to_dataframe(df5_subset, spectrum_col='Spectrum', smiles_col='SMILES_spectra')

# # Add Group and Response columns by mapping directly from df5_subset
# # Create dictionaries for faster lookup
# smiles_to_group = df5_subset.set_index('SMILES_spectra')['Group'].to_dict()
# smiles_to_response = df5_subset.set_index('SMILES_spectra')['Response'].to_dict()

# # Map the values directly
# df5_spectra['Group'] = df5_spectra['SMILES_spectra'].map(smiles_to_group)
# df5_spectra['Response'] = df5_spectra['SMILES_spectra'].map(smiles_to_response)

# print("=== SPECTRA DATAFRAME ===")
# print(f"Shape: {df5_spectra.shape}")
# print(f"Unique SMILES: {df5_spectra['SMILES_spectra'].nunique()}")
# print(f"Columns: {list(df5_spectra.columns[:3])} ... {list(df5_spectra.columns[-3:])}")  # Show first and last few columns


# df5_spectra = pd.read_csv('/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_spectra.csv')
# df5_subset = pd.read_csv('/home/dlipsey/MITLincolnLabs/MIT_LL_data/df5_subset.csv')


# Define your parameters
# bin_sizes = [0.05, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000] # 
# thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
# save_directory = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes"

# Create all datasets
# all_datasets = fd.binning_loop(df5_spectra, df5_subset, bin_sizes, thresholds, save_directory, indx_id_indx=-3, startindx=1, stopindx=-3)


### ======================================================= TOXICITY MLP ======================================================= ###

# Load datasets folder path
grid_search_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes"

# Get all .pkl files in the folder
pkl_files = [f for f in os.listdir(grid_search_folder) if f.endswith('.pkl')]
dataset_names = [f.replace('.pkl', '') for f in pkl_files]

print(f"Found {len(dataset_names)} datasets to process for MLP training")

# Set up device for PyTorch
# device = fd.set_up_gpu()
device = torch.device('cpu')

# Initialize storage for MLP results
mlp_results_r2 = []
mlp_results_percent_error = []

# Dictionary to store individual errors for histogram analysis
saved_mlp_errors = {}

# Spectra Toxicity MLP
batch_size = 128
epochs = 500
lr = 0.0001
criterion = nn.MSELoss()
output_size = 1
num_layers = 18


# Process datasets ONE AT A TIME (memory efficient)
for i, dataset_name in enumerate(sorted(dataset_names), 1):
    print(f"Processing MLP {i}/{len(dataset_names)}: {dataset_name}")
    
    try:
        # Load only the current dataset
        file_path = os.path.join(grid_search_folder, f"{dataset_name}.pkl")
        df = pd.read_pickle(file_path)
        
        print(f"Loaded dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns[:3])} ... {list(df.columns[-5:])}")
        
        # Identify feature columns (exclude SMILES, Response, log_response, index_id, Group if present)
        exclude_cols = ['SMILES_spectra', 'Response', 'log_response', 'index_id']
        if 'Group' in df.columns:
            exclude_cols.append('Group')
            
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Prepare features and target
        X = df[feature_cols].astype(np.float32)
        y = df['log_response'].astype(np.float32)
        
        # Remove rows with NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(X_clean) < 10:  # Skip if too few samples
            print(f"  Skipping {dataset_name}: Only {len(X_clean)} valid samples")
            continue
            
        print(f"Clean data shape: X={X_clean.shape}, y={y_clean.shape}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.5, random_state=42
        )
        
        # Create tensors
        X_train_tensor = torch.FloatTensor(X_train.values).to(device)
        X_test_tensor = torch.FloatTensor(X_test.values).to(device)
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
        y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Tensor shapes: X_train={X_train_tensor.shape}, y_train={y_train_tensor.shape}")
        
        input_size = X_train_tensor.shape[1]
        mlp_model = fd.SpecToxMLP_Reg(input_size, output_size, num_layers).to(device)
        print(f"Created MLP with input size: {input_size}")
        
        # Create config for this dataset (if you want to track with wandb)
        bin_size, threshold = parse_dataset_name(dataset_name)

        spectra_config = {
            'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
            'wandb_project': 'MIT-Lincoln-Lab',
            'gpu': False,
            'embedding_type': "Spectra",
            'encoder_type': 'Spectra Toxicity MLP',
            'dataset_name': dataset_name,
            'bin_size': bin_size,
            'threshold': threshold,
            # Model hyperparameters         
            'batch_size': batch_size,
            'output_size': output_size,
            'num_layers': num_layers,
            'learning_rate': lr,
            'epochs': epochs,
        'samples': len(X_clean)
        }       

        # Train the model - use the spectra-specific training function
        trained_mlp, train_losses, val_losses = fd.train_model_MLP_spectra(
            model=mlp_model,
            train_data=train_loader,
            val_data=test_loader,
            epochs=epochs,
            learning_rate=lr,
            criterion=criterion,
            device=device,
            config=spectra_config
        )
        
        # Make predictions
        trained_mlp.eval()
        with torch.no_grad():
            y_train_pred = trained_mlp(X_train_tensor).cpu().numpy().flatten()
            y_test_pred = trained_mlp(X_test_tensor).cpu().numpy().flatten()
        
        # Convert back to numpy for evaluation
        y_train_np = y_train_tensor.cpu().numpy().flatten()
        y_test_np = y_test_tensor.cpu().numpy().flatten()
        
        # Calculate R² metrics
        train_r2 = r2_score(y_train_np, y_train_pred)
        test_r2 = r2_score(y_test_np, y_test_pred)
        
        # Calculate absolute percent error (undo log transform first)
        y_train_true_response = np.exp(y_train_np)
        y_train_pred_response = np.exp(y_train_pred)
        y_test_true_response = np.exp(y_test_np)
        y_test_pred_response = np.exp(y_test_pred)
        
        # Calculate individual errors for test set
        individual_errors = np.abs((y_test_pred_response - y_test_true_response) / y_test_true_response) * 100
        
        # Save individual errors for histogram analysis
        saved_mlp_errors[dataset_name] = individual_errors
        
        # Calculate median and mean absolute percent error
        train_median_percent_error = 100 * (np.median(np.abs(y_train_pred_response - y_train_true_response) / y_train_true_response))
        test_median_percent_error = 100 * (np.median(np.abs(y_test_pred_response - y_test_true_response) / y_test_true_response))
        train_mean_percent_error = 100 * (np.mean(np.abs(y_train_pred_response - y_train_true_response) / y_train_true_response))
        test_mean_percent_error = 100 * (np.mean(np.abs(y_test_pred_response - y_test_true_response) / y_test_true_response))

        # Store results
        mlp_results_r2.append({
            'Dataset': dataset_name,
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Samples': len(X_clean),
            'Features': input_size
        })
        
        mlp_results_percent_error.append({
            'Dataset': dataset_name,
            'Train_Median_Percent_Error': train_median_percent_error,
            'Test_Median_Percent_Error': test_median_percent_error,
            'Train_Mean_Percent_Error': train_mean_percent_error,
            'Test_Mean_Percent_Error': test_mean_percent_error,
            'Samples': len(X_clean),
            'Features': input_size
        })
        
        print(f"Completed: Test R² = {test_r2:.4f}, Test Median % Error = {test_median_percent_error:.1f}%")
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        continue
    
    finally:
        # Always clean up memory after each dataset
        if 'df' in locals():
            del df
        if 'X_clean' in locals():
            del X_clean, y_clean, X_train, X_test, y_train, y_test
        if 'mlp_model' in locals():
            del mlp_model
        if 'X_train_tensor' in locals():
            del X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Periodic deeper cleanup every 20 datasets
        if i % 20 == 0:
            print(f"  Deep cleanup after {i} datasets...")
            gc.collect()

# Convert results to DataFrames
df_mlp_r2_results = pd.DataFrame(mlp_results_r2)
df_mlp_percent_error_results = pd.DataFrame(mlp_results_percent_error)

print(f"\nMLP Training Completed! Processed {len(mlp_results_r2)} datasets successfully.")
print(f"Saved individual errors for {len(saved_mlp_errors)} datasets")
print(f"Results stored in: df_mlp_r2_results, df_mlp_percent_error_results")

### ============================================================= MLP HEATMAP ============================================================= ###

# Function to extract bin size and threshold from dataset name (if not in function_depot)
def parse_dataset_name_local(dataset_name):
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

# Add bin_size and threshold columns to MLP results DataFrames
for df_results in [df_mlp_r2_results, df_mlp_percent_error_results]:
    bin_sizes = []
    thresholds = []
    
    for dataset_name in df_results['Dataset']:
        bin_size, threshold = parse_dataset_name_local(dataset_name)
        bin_sizes.append(bin_size)
        thresholds.append(threshold)
    
    df_results['BinSize'] = bin_sizes
    df_results['Threshold'] = thresholds

# Check for and remove duplicates before creating pivot tables
print("Checking for duplicates in MLP results...")
print(f"Original df_mlp_r2_results shape: {df_mlp_r2_results.shape}")

# Remove duplicates based on BinSize + Threshold combination (keep first occurrence)
df_mlp_r2_results = df_mlp_r2_results.drop_duplicates(subset=['BinSize', 'Threshold'], keep='first')
df_mlp_percent_error_results = df_mlp_percent_error_results.drop_duplicates(subset=['BinSize', 'Threshold'], keep='first')

print(f"After removing duplicates: {df_mlp_r2_results.shape}")

# Now create pivot tables 
mlp_r2_pivot = df_mlp_r2_results.pivot(index='BinSize', columns='Threshold', values='Test_R2') 
mlp_median_percent_error_pivot = df_mlp_percent_error_results.pivot(index='BinSize', columns='Threshold', values='Test_Median_Percent_Error')
mlp_mean_percent_error_pivot = df_mlp_percent_error_results.pivot(index='BinSize', columns='Threshold', values='Test_Mean_Percent_Error')

# List all expected thresholds
thresholds_subset = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
bins_subset = [0.05, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]

# Reindex pivot tables to show all columns, filling missing with NaN
mlp_r2_pivot = mlp_r2_pivot.reindex(columns=thresholds_subset, index=bins_subset)
mlp_median_percent_error_pivot = mlp_median_percent_error_pivot.reindex(columns=thresholds_subset, index=bins_subset)
mlp_mean_percent_error_pivot = mlp_mean_percent_error_pivot.reindex(columns=thresholds_subset, index=bins_subset)

# Create detailed heatmap function for MLP results
def create_detailed_heatmap_mlp(pivot_data, metric_name, cmap, figsize=(12, 8), vmin=None, vmax=None):
    """Create a detailed heatmap for a single MLP metric"""
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(pivot_data, 
                annot=True, 
                fmt='.3f' if 'R²' in metric_name else '.1f', 
                cmap=cmap,
                square=False,
                linewidths=0.5,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': f'Test {metric_name}', 'shrink': 0.8})
    
    plt.title(f'MLP: {metric_name} by Bin Size and Threshold', fontsize=16, fontweight='bold')
    plt.xlabel('Threshold Value', fontsize=14)
    plt.ylabel('Bin Size', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Improve readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Add text annotation for best performance
    if 'R²' in metric_name:
        best_val = pivot_data.max().max()
        plt.text(0.02, 0.98, f'Best R²: {best_val:.4f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
    else:
        best_val = pivot_data.min().min()
        plt.text(0.02, 0.98, f'Best {metric_name}: {best_val:.1f}%', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f"/home/dlipsey/MITLincolnLabs/Figures/MLP_{metric_name}_by_Bin_Size_and_Threshold.png", dpi=300, bbox_inches='tight')
    plt.show()

# Create detailed individual heatmaps
print("Creating detailed MLP heatmaps...")

create_detailed_heatmap_mlp(mlp_r2_pivot, 'R²', 'RdYlBu')     
create_detailed_heatmap_mlp(mlp_median_percent_error_pivot, 'Median_Absolute_Percent_Error', 'RdYlBu_r', vmin=1.0, vmax=100.0) 
create_detailed_heatmap_mlp(mlp_mean_percent_error_pivot, 'Mean_Absolute_Percent_Error', 'RdYlBu_r', vmin=1.0, vmax=100.0)

print("MLP analysis and heatmap generation complete!")