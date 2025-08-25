### ==== IMPORTS ====
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

#import wandb
import itertools
import GPUtil
from collections import Counter, OrderedDict
import dask.dataframe as dd
import os
from fcd_torch import FCD

import poetry

### ==== ENCODERS ====
#%%
# Encoder architecture (With Validation Set)
# batch_size = 64
# epochs=500
# lr=0.0001
# criterion=nn.MSELoss()
# output_size = 512
# num_layers = 8

class Encoder(nn.Module):
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

def train_model_encoder(model, train_data, val_data, epochs, learning_rate, criterion, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch, true_embeddings, _ in train_data:  
            batch = batch.to(device)
            true_embeddings = true_embeddings.to(device)

            optimizer.zero_grad()
            batch_predicted_embeddings = model(batch)
            loss = criterion(batch_predicted_embeddings, true_embeddings)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_train_loss = running_loss / len(train_data) 

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch, val_true_embeddings, _ in val_data:  
                val_batch = val_batch.to(device)
                val_true_embeddings = val_true_embeddings.to(device)

                val_batch_predicted_embeddings = model(val_batch)

                val_batch_loss = criterion(val_batch_predicted_embeddings, val_true_embeddings)  
                val_loss += val_batch_loss.item() 
        average_val_loss = val_loss / len(val_data) 

        if epoch % 50 == 0 or epoch == epochs - 1:  
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'   Training loss: {average_train_loss:.6f}')
            print(f'   Validation loss: {average_val_loss:.6f}')

    return model

#%%
# Spectra Toxicity MLP
# batch_size = 128
# epochs=1000
# lr=0.0001
# criterion=nn.MSELoss()
# output_size = 1
# num_layers = 10

# Everything below this line SHOULD be able to run without modification
class SpecToxMLP_Reg(nn.Module):
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

def train_model_MLP_spectra(model, train_data, val_data, epochs, learning_rate, criterion, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch, true_log_tox, _ in train_data:
            batch = batch.to(device)
            true_log_tox = true_log_tox.to(device)

            optimizer.zero_grad()
            batch_predicted_log_tox = model(batch)
            loss = criterion(batch_predicted_log_tox, true_log_tox)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch, val_true_tox, _ in val_data:
                val_batch = val_batch.to(device)
                val_true_tox = val_true_tox.to(device)

                val_batch_predicted_tox = model(val_batch)

                val_loss = criterion(val_batch_predicted_tox, val_true_tox)
                val_loss += loss.item()
        average_val_loss = val_loss / len(val_loader)

        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'   Training loss: {average_train_loss}')
        print(f'   Validation loss: {average_val_loss}')

    return model

#%%
# ChemNet MLP
# batch_size = 128
# epochs=100
# lr=0.0001
# criterion=nn.MSELoss()
# output_size = 1
# num_layers = 5

# Everything below this line SHOULD be able to run without modification
class ToxMLP(nn.Module):
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

def train_model_MLP(model, train_data, val_data, epochs, learning_rate, criterion, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch, true_log_tox, _ in train_data:
            batch = batch.to(device)
            true_log_tox = true_log_tox.to(device)

            optimizer.zero_grad()
            batch_predicted_log_tox = model(batch)
            loss = criterion(batch_predicted_log_tox, true_log_tox)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch, val_true_tox, _ in val_data:
                val_batch = val_batch.to(device)
                val_true_tox = val_true_tox.to(device)

                val_batch_predicted_tox = model(val_batch)

                val_loss = criterion(val_batch_predicted_tox, val_true_tox)
                val_loss += loss.item()
        average_val_loss = val_loss / len(val_loader)

        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'   Training loss: {average_train_loss}')
        print(f'   Validation loss: {average_val_loss}')

    return model

#%%
# Conditional encoder 
# batch_size = 64
# epochs=500
# lr=0.0001
# criterion1=nn.MSELoss()
# criterion2=nn.MSELoss()
# output_size = 513
# num_layers = 8

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
            
            print(loss1, loss2) # So we see what the losses are to pin on what lambda should be
            total_loss = loss1 + ((2) * loss2 ) # lambda = 2, bigger lambda will make the prediction accuracy much more important
            


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

#%%

### ==== FUNCTIONS ====

def set_up_gpu():
    if torch.cuda.is_available():
        # Get the list of GPUs
        gpus = GPUtil.getGPUs()

        # Find the GPU with the most free memory
        best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)

        # Print details about the selected GPU
        print(f"Selected GPU ID: {best_gpu.id}")
        print(f"  Name: {best_gpu.name}")
        print(f"  Memory Free: {best_gpu.memoryFree} MB")
        print(f"  Memory Used: {best_gpu.memoryUsed} MB")
        print(f"  GPU Load: {best_gpu.load * 100:.2f}%")

        # Set the device for later use
        device = torch.device(f'cuda:{best_gpu.id}')
        print('Current device ID: ', device)

        # Set the current device in PyTorch
        torch.cuda.set_device(best_gpu.id)
    else:
        device = torch.device('cpu')
        print('Using CPU')
        

    # Confirm the currently selected device in PyTorch
    print("PyTorch current device ID:", torch.cuda.current_device())
    print("PyTorch current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    return device

def spectrum_string_to_dataframe(df, spectrum_col, smiles_col):
    """
    Converts a DataFrame with a spectrum column (string of 'x:y' pairs) into a matrix
    where columns are unique x values, rows are spectra (even for duplicate SMILES), and values are y (intensity).
    Creates and preserves an index_id column for tracking. All spectral columns will be float type with float values.
    Spectral columns are sorted by their float values in ascending order.
    """
    # Create a copy of the input DataFrame and add index_id
    df_copy = df.copy()
    df_copy['index_id'] = range(len(df_copy))
    
    # Collect all unique x values (m/z) and convert to float
    x_values_set = set()
    data_rows = []
    
    for idx, row in df_copy.iterrows():
        spectrum = row[spectrum_col]
        pairs = spectrum.split()
        xy_dict = {}
        
        for pair in pairs:
            try:
                x, y = pair.split(":") # Split into x and y
                x_float = float(x)
                y_float = float(y)
                xy_dict[x_float] = y_float
                x_values_set.add(x_float)
            except Exception:
                continue
        
        # Store row data including index_id
        data_rows.append({
            'original_index': idx,
            smiles_col: row[smiles_col],
            'index_id': row['index_id'],
            'xy_dict': xy_dict
        })
    
    # Sort x values by their float values in ascending order
    x_values = sorted(x_values_set)
    
    # Build the result DataFrame with columns in sorted order
    result_data = {}
    
    # Add SMILES column first
    result_data[smiles_col] = [row[smiles_col] for row in data_rows]
    
    # Add spectral columns in sorted order
    for x_val in x_values:
        result_data[x_val] = [float(row['xy_dict'].get(x_val, 0.0)) for row in data_rows]
    
    # Add index_id column last
    result_data['index_id'] = [row['index_id'] for row in data_rows]
    
    # Create DataFrame - columns will be in the order we added them
    df_matrix = pd.DataFrame(result_data)
    
    # Set the index to match original DataFrame
    original_indices = [row['original_index'] for row in data_rows]
    df_matrix.index = original_indices
    
    return df_matrix

# Cate's smiles to ChemNet embedding code
def get_chemnet_emb_from_smiles(smiles_list):
    """
    Get ChemNet embeddings from a list of SMILES strings.

    Parameters:
    smiles_list (list): List of SMILES strings.

    Returns:
    dict: A dictionary mapping each SMILES string to its corresponding ChemNet embedding.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fcd = FCD(device, n_jobs=1)
    
    smiles_emb_dict = {}

    for smiles in smiles_list:
        try:
            emb = fcd.get_predictions([smiles])[0]
            smiles_emb_dict[smiles] = list(emb)
        except KeyError as e:
            if e == 'PropertyTable':
                smiles_emb_dict[smiles] = 'unknown'

    return smiles_emb_dict

# Redefine the fuction so it makes a list rather than a dictionary, Done to get dataset
def get_chemnet_emb_from_smiles_list(smiles_list):
    """
    Get ChemNet embeddings for a list of SMILES strings, preserving order and duplicates.
    Returns a list of embeddings (or 'unknown') in the same order as input.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fcd = FCD(device, n_jobs=1)
    embeddings = []
    for smiles in smiles_list:
        try:
            emb = fcd.get_predictions([smiles])[0]
            embeddings.append(list(emb))
        except KeyError as e:
            if e == 'PropertyTable':
                embeddings.append('unknown')
    return embeddings

# Add the 'Response' and 'log_response' columns 
# This is currently specifically for df3 column names
def add_response_and_log_response(spectra_df, original_df, smiles_col='SMILES_spectra'):
    """
    Adds 'Response' and 'log_response' columns to spectra_df by mapping from original_df using the SMILES column.
    """
    smiles_to_response = original_df.drop_duplicates(subset=smiles_col).set_index(smiles_col)['Response']
    spectra_df['Response'] = spectra_df[smiles_col].map(smiles_to_response)
    spectra_df['log_response'] = np.log(spectra_df['Response'])
    return spectra_df

# Define a function to assign EPA levels
def assign_epa_level(response):
    if response <= 50:
        return "EPA_level_1"
    elif response <= 500:
        return "EPA_level_2"
    elif response <= 5000:
        return "EPA_level_3"
    else:
        return "EPA_level_4"

# Add EPA levels (one-hot encoded)
def add_epa_levels(df, response_col='Response', assign_func=assign_epa_level):
    """
    Adds EPA level columns (one-hot) to the DataFrame based on the response column.
    Removes the original response column.
    """
    df = df.copy()
    df["EPA_level"] = df[response_col].apply(assign_func)
    df = pd.get_dummies(df, columns=["EPA_level"], prefix='', prefix_sep='')
    epa_cols = [col for col in df.columns if str(col).startswith("EPA_level_")]
    df[epa_cols] = df[epa_cols].astype(int)
    #df.drop(columns=[response_col], inplace=True)
    return df

# Threshold filter function
def apply_threshold_filter(df, threshold):
    """
    Applies a threshold filter to spectral data, setting values below threshold to zero.
    
    Parameters:
    df: DataFrame with first column as SMILES, last column as index_id, rest as spectral intensity columns
    threshold: Float, minimum value to keep (values below this become 0)
    
    Returns:
    DataFrame with filtered spectral data (values below threshold set to 0)
    """
    
    # Create a copy to avoid modifying the original
    filtered_df = df.copy()
    
    # Get spectral columns (all except first and last column)
    spectral_cols = filtered_df.columns[1:-1]
    
    # Ensure spectral data is numeric
    # filtered_df[spectral_cols] = filtered_df[spectral_cols].apply(pd.to_numeric, errors='coerce')
    
    # Apply threshold using numpy where - more explicit control
    spectral_data = filtered_df[spectral_cols].values
    spectral_data = np.where(spectral_data > threshold, spectral_data, 0)
    filtered_df[spectral_cols] = spectral_data
    
    # index_id column is preserved unchanged
    return filtered_df

# Uniform binning function
def bin_spectra_by_mz_range(df, bin_size):
    """
    Bins spectra data by grouping m/z columns into ranges of specified size.
    
    Parameters:
    df: DataFrame with first column as SMILES, last column as index_id, rest as m/z columns (float names)
    bin_size: Float, the size of each bin (e.g., 10 means bins of 0-10, 10-20, etc.)
    
    Returns:
    DataFrame with SMILES column, binned m/z columns named by bin midpoints, and index_id column
    """
    smiles_col = df.columns[0]
    index_col = df.columns[-1]  # Preserve the last column (index_id)
    mz_cols = df.columns[1:-1]  # Exclude first and last columns
    
    # Create bins and assign each m/z to a bin
    bin_assignments = {}
    for mz in mz_cols:
        bin_start = (mz // bin_size) * bin_size
        bin_end = bin_start + bin_size
        bin_midpoint = bin_start + (bin_size / 2)
        
        # Round to avoid floating point precision issues
        bin_midpoint = round(bin_midpoint, 3)  
        
        if bin_midpoint not in bin_assignments:
            bin_assignments[bin_midpoint] = []
        bin_assignments[bin_midpoint].append(mz)
    
    # Create new DataFrame with binned data
    result_df = pd.DataFrame()
    result_df[smiles_col] = df[smiles_col]
    
    # Sum intensities for each bin
    for bin_midpoint in sorted(bin_assignments.keys()):
        cols_in_bin = bin_assignments[bin_midpoint]
        result_df[bin_midpoint] = df[cols_in_bin].sum(axis=1)
    
    # Preserve index_id column
    result_df[index_col] = df[index_col]
    
    return result_df

# Bin filling function
def fill_missing_bins(df, bin_size):
    """
    Fills in missing bin columns in a binned DataFrame.
    
    Parameters:
    df: DataFrame with first column as SMILES, last column as index_id, rest as binned m/z columns (float names)
    bin_size: Float, the original bin size used for binning
    
    Returns:
    DataFrame with all missing bin midpoints filled in with zeros
    """
    smiles_col = df.columns[0]
    index_col = df.columns[-1]  # Preserve the last column (index_id)
    existing_bins = sorted([col for col in df.columns[1:-1] if isinstance(col, (int, float))])
    
    if not existing_bins:
        return df
    
    # Calculate the step size 
    step_size = bin_size
    
    # Find the range of bins to fill
    min_bin = existing_bins[0]
    max_bin = existing_bins[-1]
    
    # Generate all possible bin midpoints from first non-zero step to max_bin
    all_bins = []
    current_bin = step_size / 2  # Start from first non-zero bin (don't include 0)
    while current_bin <= max_bin:
        all_bins.append(current_bin)
        current_bin += step_size
    
    # Find missing bins
    missing_bins = set(all_bins) - set(existing_bins)
    
    # Add missing bins with zeros
    result_df = df.copy()
    for bin_midpoint in missing_bins:
        result_df[bin_midpoint] = 0.0
    
    # Reorder columns: SMILES column first, then sorted bin columns, then index_id column
    bin_cols = sorted([col for col in result_df.columns[1:-1] if isinstance(col, (int, float))])
    ordered_cols = [smiles_col] + bin_cols + [index_col]
    result_df = result_df[ordered_cols]
    
    return result_df

def binning_loop(df_spectra, df_original, bin_sizes, thresholds, save_directory):
    """
    Creates all binned and thresholded datasets for a complete grid search.
    
    Parameters:
    - df_spectra: DataFrame with spectral data (output from spectrum_string_to_dataframe)
    - df_original: Original DataFrame with response data (e.g., df4_QQpos)
    - bin_sizes: List of bin sizes to use
    - thresholds: List of threshold values to use
    - save_directory: Directory path to save the pickle files
    
    Returns:
    - Dictionary with all created datasets keyed by variable names
    """
    import warnings
    
    created_datasets = {}
    
    # Create ALL binned and thresholded datasets (complete grid search)
    print("Creating all binned and thresholded datasets...")
    df_spectra_original = df_spectra.copy()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
            
        for bin_size in bin_sizes:
            for threshold in thresholds:
                
                # Create variable name
                bin_str = str(bin_size).replace('.', '_')
                thresh_str = str(threshold).replace('.', '_')
                var_name = f"bin{bin_str}_thresh{thresh_str}_df_spectra"
                    
                # Start with original data
                current_data = df_spectra_original.copy()
            
                # Apply threshold filtering first
                threshold_filtered_data = apply_threshold_filter(current_data, threshold)
                
                # Then apply binning
                binned_data = bin_spectra_by_mz_range(threshold_filtered_data, bin_size)
            
                # Fill missing bins
                filled_data = fill_missing_bins(binned_data, bin_size)
            
                # Add response and log response values
                final_data = add_response_and_log_response(filled_data, df_original)
                
                # Ensure index_id is preserved from original data
                if 'index_id' in df_spectra.columns:
                    final_data['index_id'] = df_spectra['index_id'].iloc[:len(final_data)].values
                
                # Store in created_datasets dictionary
                created_datasets[var_name] = final_data
                
                # Save to file
                save_path = f"{save_directory}/{var_name}.pkl"
                final_data.to_pickle(save_path)
                print(f"Saved {var_name} to {save_path} - Shape: {final_data.shape}")

    print(f"  - {len(bin_sizes)} bin sizes: {bin_sizes}")
    print(f"  - {len(thresholds)} threshold values: {thresholds}")
    print(f"  - Plus the existing {len(bin_sizes)} thresh0 datasets")

    # Create the missing threshold 0 datasets
    print("Creating binned-only datasets (thresh0)...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for bin_size in bin_sizes:
            # Create variable name for thresh0 (no threshold)
            bin_str = str(bin_size).replace('.', '_')
            var_name = f"bin{bin_str}_thresh_zero_df_spectra"
        
            print(f"Creating {var_name}...")
        
            # Start with original data (no threshold filtering)
            current_data = df_spectra_original.copy()
        
            # Binning only
            binned_data = bin_spectra_by_mz_range(current_data, bin_size)
        
            # Fill missing bins
            filled_data = fill_missing_bins(binned_data, bin_size)
        
            # Add response and log response values
            final_data = add_response_and_log_response(filled_data, df_original)
            
            # Ensure index_id is preserved from original data
            if 'index_id' in df_spectra.columns:
                final_data['index_id'] = df_spectra['index_id'].iloc[:len(final_data)].values
            
            # Store in created_datasets dictionary
            created_datasets[var_name] = final_data
            
            # Save to file
            save_path = f"{save_directory}/{var_name}.pkl"
            final_data.to_pickle(save_path)
            print(f"Saved {var_name} to {save_path}")

    print(f"Created {len(bin_sizes)} thresh0 datasets!")
    print(f"Total datasets created: {len(created_datasets)}")
    
    return created_datasets



### ==== TENSORS CREATION FUNCTIONS ====

# This is our default function, the one we use to prep the data for the encoder that takes us from spectra to ChemNet encodings 
def create_dataset_tensors(spectra_dataset, embedding_df, device, start_idx=None, stop_idx=None):
    """
    Create tensors from the provided spectra dataset and embedding DataFrame.

    Parameters:
    ----------
    spectra_dataset : pd.DataFrame
        DataFrame containing spectral data and chemical labels. Assumes specific 
        columns for processing based on the `carl` flag.

    embedding_df : pd.DataFrame
        DataFrame containing embeddings for chemicals, with 'Embedding Floats' 
        column corresponding to ChemNet embeddings.

    device : torch.device
        The device (CPU or GPU) on which to store the tensors.

    carl : bool, optional
        If True, processes the dataset assuming it has a different structure 
        (specifically without an 'Unnamed: 0' column). Default is False.

    Returns:
    -------
    tuple
        A tuple containing:
        - embeddings_tensor (torch.Tensor): Tensor of true embeddings for the chemicals.
        - spectra_tensor (torch.Tensor): Tensor of spectral data.
        - chem_encodings_tensor (torch.Tensor): Tensor of chemical name encodings.
        - spectra_indices_tensor (torch.Tensor): Tensor of indices corresponding to the spectra.
    """
    spectra = spectra_dataset.iloc[:,start_idx:stop_idx]

    # create tensors of spectra, true embeddings, and chemical name encodings for train and val
    chem_labels = list(spectra_dataset['SMILES_spectra'])
    embeddings_tensor = torch.Tensor([embedding_df.loc[embedding_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
    spectra_tensor = torch.Tensor(spectra.values).to(device)
    spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

    return embeddings_tensor, spectra_tensor, spectra_indices_tensor

def create_dataset_tensors_tox(spectra_dataset,device, start_idx=None, stop_idx=None):

    spectra = spectra_dataset.iloc[:,start_idx:stop_idx] # Prev was [1, -4]

    # create tensors of spectra, true toxicity values, and chemical name encodings for train and val
    #chem_labels = list(spectra_dataset['SMILES_spectra'])
    log_tox_tensor = torch.Tensor(spectra_dataset["log_response"].values).unsqueeze(1).to(device)
    spectra_tensor = torch.Tensor(spectra.values).to(device)
    spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

    return log_tox_tensor, spectra_tensor, spectra_indices_tensor

def create_dataset_tensors_tox_spec(spectra_dataset,device, start_idx=None, stop_idx=None):

    embedding_cols = [col for col in spectra_dataset.columns if col.startswith('Embedding Float')]
    spectra = spectra_dataset[embedding_cols]

    # create tensors of spectra, true toxicity values, and chemical name encodings for train and val
    #chem_labels = list(spectra_dataset['SMILES_spectra'])
    log_tox_tensor = torch.Tensor(spectra_dataset["log_response"].values).unsqueeze(1).to(device)
    spectra_tensor = torch.Tensor(spectra.values).to(device)
    spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

    return log_tox_tensor, spectra_tensor, spectra_indices_tensor

def create_dataset_tensors_emb_tox(spectra_dataset, embedding_df, device, start_idx=None, stop_idx=None):

    spectra = spectra_dataset.iloc[:,start_idx:stop_idx] # prev was [1,-3]

    # create tensors of spectra, true embeddings, true toxicity values, and chemical name encodings for train and val
    chem_labels = list(spectra_dataset['SMILES_spectra'])
    log_tox_tensor = torch.Tensor(spectra_dataset["log_response"].values).unsqueeze(1).to(device)
    embeddings_tensor = torch.Tensor([embedding_df.loc[embedding_df['SMILES'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
    spectra_tensor = torch.Tensor(spectra.values).to(device)
    spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

    return embeddings_tensor, log_tox_tensor, spectra_tensor, spectra_indices_tensor 
