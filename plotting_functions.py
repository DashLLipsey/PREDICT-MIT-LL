import seaborn as sns
from scipy.spatial import distance
import matplotlib.pyplot as plt
# from scipy.spatial import ConvexHull
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import wandb
import torch
from torch.utils.data import DataLoader
import functions_enc as f
from scipy.stats import zscore
import os
import random
from sklearn.preprocessing import StandardScaler
# import time

# # Here are the functions currently defined in this file. There is definitely overlap between some of these functions, 
# # and some could be combined into a single function with optional arguments to handle different use cases.

# def plot_ims_spectra_pca(data, sample_size=1000):
# def plot_similarity_comparison(spectra, chem, synthetic_spectra_df, gen_type, first_cut_col, last_cut_col, num_samples=100, similarity_type='per_pair'):
# def preds_to_emb_pca_plot(predicted_embeddings, output_name_encodings, sorted_chem_names, emb_df, mass_spec_encoder_embeddings=False, mass_spec_chems=False):
# def add_hulls(ax, pca, chem_data, threshold=3):
# def plot_emb_pca(all_embeddings, ims_embeddings, results_type, input_type, embedding_type='ChemNet', mass_spec_embeddings=None, log_wandb=True, chemnet_embeddings_to_plot=None, mse_insert=None, insert_position=[0.05, 0.05], show_wandb_run_name=True, plot_hulls=False, hull_data=None):
# def plot_generation_results_pca(true_spectra, synthetic_spectra, chem_labels, results_type, sample_size=None, chem_of_interest=None, log_wandb=False, mse_insert=None, insert_position=[0.05, 0.05], show_wandb_run_name=False):
# def plot_generation_results_pca_single_chem_side_by_side(true_spectra, synthetic_spectra, chem_labels, results_type, sample_size=None, chem_of_interest=None, log_wandb=False, mse_insert=None, insert_position=[0.05, 0.05], show_wandb_run_name=False, x_lims=None, y_lims=None, save_plot_path=None):
# def plot_pca(data, batch_size, model, device, encoder_criterion, sorted_chem_names, all_embeddings_df, ims_embeddings_df, results_type, input_type, embedding_type='ChemNet', show_wandb_run_name=True, log_wandb=True):
# def plot_carl_real_synthetic_comparison(true_carl, synthetic_carl, results_type, chem_label, log_wandb=False, show_wandb_run_name=False, criterion=None, run_name=None, save_plot_path=None):
# def format_data_for_plotting(data):
# def plot_spectra_real_synthetic_comparison(true_spec, synthetic_spec, results_type, chem_label, log_wandb=False, show_wandb_run_name=True, criterion=None, run_name=None):
# def plot_and_save_generator_results(data, batch_size, sorted_chem_names, model, device, criterion, num_plots, plot_overlap_pca=False, save_plots_to_wandb=True, show_wandb_run_name=True, test_or_train='Train'):


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def plot_comparison_multiple_spectra_per_plot(
        dataset1, dataset2, dataset1_type, dataset2_type, 
        chem_label, save_plot_path=None, num_spectra=10,
        custom_top_row_cutoff=None, custom_bottom_row_cutoff=None,
        ):
    """
    Plot comparison of two datasets with multiple spectra per plot.
    This function generates a 2x2 grid of subplots comparing two datasets.
    Parameters:
    ----------
    dataset1 : pd.DataFrame
        DataFrame containing the first dataset to plot.
    dataset2 : pd.DataFrame
        DataFrame containing the second dataset to plot.
    dataset1_type : str
        Type of the first dataset (e.g., 'Train Spectra', 'Test CARLs', etc.).
    dataset2_type : str
        Type of the second dataset.
    chem_label : str
        Name of the chemical to plot.
    save_plot_path : str, optional
        Path to save the plot. If None, the plot will not be saved.
    """
    _, axes = plt.subplots(2, 2, figsize=(20, 14))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # x axis should run from lowest drift time (184) to highest drift time (184 + len(true_carl)//2)
    numbers = range(184, (dataset1.shape[1]//2)+184)

    axes[0].set_title(f'{chem_label} Positive {dataset1_type}', fontsize=24)
    axes[1].set_title(f'{chem_label} Positive {dataset2_type}', fontsize=24)
    axes[2].set_title(f'{chem_label} Negative {dataset1_type}', fontsize=24)
    axes[3].set_title(f'{chem_label} Negative {dataset2_type}', fontsize=24)

    for i, (row1, row2) in enumerate(zip(dataset1.iterrows(), dataset2.iterrows())):
        if i >= num_spectra:
            break
        spec1 = row1[1].values
        spec2 = row2[1].values
        if custom_top_row_cutoff is None:
            axes[0].plot(numbers, spec1[:len(numbers)])
            axes[1].plot(numbers, spec2[:len(numbers)])
        else:
            numbers = range(custom_top_row_cutoff[0], custom_top_row_cutoff[1])
            axes[0].plot(numbers, spec1[custom_top_row_cutoff[0]-184:custom_top_row_cutoff[1]-184])
            axes[1].plot(numbers, spec2[custom_top_row_cutoff[0]-184:custom_top_row_cutoff[1]-184])

        if custom_bottom_row_cutoff is None:
            axes[2].plot(numbers, spec1[len(numbers):])
            axes[3].plot(numbers, spec2[len(numbers):])
        else:
            numbers = range(custom_bottom_row_cutoff[0], custom_bottom_row_cutoff[1])
            # add 654 ((len(spectrum)/2) + 184) to account for the offset and negative spectra
            axes[2].plot(numbers, spec1[custom_bottom_row_cutoff[0]+654:custom_bottom_row_cutoff[1]+654])
            axes[3].plot(numbers, spec2[custom_bottom_row_cutoff[0]+654:custom_bottom_row_cutoff[1]+654])
        # axes[0].plot(numbers, spec1[:len(numbers)])
        # axes[1].plot(numbers, spec2[:len(numbers)])
        # axes[2].plot(numbers, spec1[len(numbers):])
        # axes[3].plot(numbers, spec2[len(numbers):])

    for ax in axes:
            ax.set_xlabel('Drift Time', fontsize=16)
            ax.set_ylabel('Ion Intensity', fontsize=16)
            # ax.legend(fontsize=14)
    
    # Adjust subplot layout to prevent overlap between titles and x-labels
    plt.tight_layout()
    if save_plot_path is not None:
        plt.savefig(save_plot_path, format='png', dpi=300)

    plt.show()
    

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def calculate_average_spectrum_and_percentiles(data):
    average_spectrum = np.mean(data, axis=0)
    lower_bound = np.percentile(data, 25, axis=0)
    upper_bound = np.percentile(data, 75, axis=0)
    return average_spectrum, lower_bound, upper_bound
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def plot_average_spectrum(
        left_plot_data, middle_plot_data, chem_names, 
        condition_name ='', condition_1_value='', condition_2_value='',
        save_file_path_pt1=None, save_file_path_pt2=None,
        left_plot_type='', middle_plot_type='',
        right_plot_data=None, right_plot_type='', condition_3_value='',
        left_plot_start_idx=2, left_plot_stop_idx=-9,
        middle_plot_stop_idx=-2, right_plot_stop_idx=-2,
        ):
    """
    Plot the average spectrum for a given dataset.

    This function calculates the average spectrum for a given dataset and plots it.

    Parameters:
    ----------
    data : pd.DataFrame
        DataFrame containing the dataset to plot.

    Returns:
    -------
    None
        Displays the average spectrum plot.
    """
    for chem_label in chem_names:
        print(f'Plotting {chem_label}...')
        left_plot_chem_data = left_plot_data[left_plot_data['Label'] == chem_label].iloc[:, left_plot_start_idx:left_plot_stop_idx]
        middle_plot_chem_data = middle_plot_data[middle_plot_data['Label'] == chem_label].iloc[:, :middle_plot_stop_idx]

        avg_spectrum_left, lower_bound_left, upper_bound_left = calculate_average_spectrum_and_percentiles(left_plot_chem_data)
        avg_spectrum_middle, lower_bound_middle, upper_bound_middle = calculate_average_spectrum_and_percentiles(middle_plot_chem_data)

        # x axis should run from lowest drift time (184) to highest drift time (184 + len(true_carl)//2)
        numbers = range(184, (len(avg_spectrum_left)//2)+184)

        if right_plot_data is not None:
            fig, axes = plt.subplots(1, 3, figsize=(30, 12), layout="constrained")
            label_fontsize = 34
            title_fontsize = 38
            legend_fontsize = 24

            fig.set_constrained_layout_pads(w_pad=40./72., h_pad=40./72.,)

            left_plot_title = f'Avg {left_plot_type}{chem_label}{condition_1_value}{condition_name} Spectra'
            middle_plot_title = f'Avg {middle_plot_type}{chem_label}{condition_2_value}{condition_name} Spectra'
            right_plot_title = f'Avg {right_plot_type}{chem_label}{condition_3_value}{condition_name} Spectra'

            right_plot_chem_data = right_plot_data[right_plot_data['Label'] == chem_label].iloc[:, :right_plot_stop_idx]
            avg_spectrum_right, lower_bound_right, upper_bound_right = calculate_average_spectrum_and_percentiles(right_plot_chem_data)
        else:
            _, axes = plt.subplots(1, 2, figsize=(14, 8))
            label_fontsize = 16
            title_fontsize = 20
            legend_fontsize = 14

            left_plot_title = f'Average of {left_plot_type}{chem_label}{condition_1_value}{condition_name} Spectra'
            # plot_type argument for synthetic data
            middle_plot_title = f'Average of {middle_plot_type}{chem_label}{condition_2_value}{condition_name} Spectra'

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        axes[0].plot(numbers, avg_spectrum_left[:len(numbers)], label='Positive', color='orange')
        axes[0].plot(numbers, avg_spectrum_left[len(numbers):], label='Negative', color='blue')
        axes[0].fill_between(numbers, lower_bound_left[:len(numbers)], upper_bound_left[:len(numbers)], color='orange', alpha=0.5, label='Positive IQR (25%-75%)')
        axes[0].fill_between(numbers, lower_bound_left[len(numbers):], upper_bound_left[len(numbers):], color='lightblue', alpha=0.5, label='Negative IQR (25%-75%)')
        axes[0].set_title(left_plot_title, fontsize=title_fontsize)
        axes[0].set_xlabel('Drift Time', fontsize=label_fontsize)
        axes[0].set_ylabel('Ion Intensity', fontsize=label_fontsize)
        axes[0].legend(fontsize=legend_fontsize)

        axes[1].plot(numbers, avg_spectrum_middle[:len(numbers)], label='Positive', color='orange')
        axes[1].plot(numbers, avg_spectrum_middle[len(numbers):], label='Negative', color='blue')
        axes[1].fill_between(numbers, lower_bound_middle[:len(numbers)], upper_bound_middle[:len(numbers)], color='orange', alpha=0.5, label='Positive IQR (25%-75%)')
        axes[1].fill_between(numbers, lower_bound_middle[len(numbers):], upper_bound_middle[len(numbers):], color='lightblue', alpha=0.5, label='Negative IQR (25%-75%)')
        axes[1].set_title(middle_plot_title, fontsize=title_fontsize)
        axes[1].set_xlabel('Drift Time', fontsize=label_fontsize)
        axes[1].set_ylabel('Ion Intensity', fontsize=label_fontsize)
        axes[1].legend(fontsize=legend_fontsize)

        if right_plot_data is not None:
            # avg_spectrum_right, lower_bound_right, upper_bound_right = calculate_average_spectrum_and_percentiles(right_plot_data)
            axes[2].plot(numbers, avg_spectrum_right[:len(numbers)], label='Positive', color='orange')
            axes[2].plot(numbers, avg_spectrum_right[len(numbers):], label='Negative', color='blue')
            axes[2].fill_between(numbers, lower_bound_right[:len(numbers)], upper_bound_right[:len(numbers)], color='orange', alpha=0.5, label='Positive IQR (25%-75%)')
            axes[2].fill_between(numbers, lower_bound_right[len(numbers):], upper_bound_right[len(numbers):], color='lightblue', alpha=0.5, label='Negative IQR (25%-75%)')
            axes[2].set_title(right_plot_title, fontsize=title_fontsize)
            axes[2].set_xlabel('Drift Time', fontsize=label_fontsize)
            axes[2].set_ylabel('Ion Intensity', fontsize=label_fontsize)
            axes[2].legend(fontsize=legend_fontsize)

        if save_file_path_pt1 is not None:
            save_file_path = save_file_path_pt1 + chem_label + save_file_path_pt2
            plt.savefig(save_file_path, bbox_inches='tight', format='png', dpi=300)

        plt.show()
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def plot_ims_spectrum(
        spectrum, chem_label, real_or_synthetic, 
        preprocessing_type='Spectrum', save_plot_path=None,
        rip_start_col=None, rip_stop_col=None
        ):
    # x axis should run from lowest drift time (184) to highest drift time (184 + len(spectrum)//2)
    numbers = range(184, (len(spectrum)//2)+184)

    plt.plot(numbers, spectrum[:len(numbers)], label='Positive')
    if rip_start_col is not None:
        plt.axvline(x=rip_start_col, color='red', linestyle='--')
        plt.axvline(x=rip_stop_col, color='red', linestyle='--')

    plt.plot(numbers, spectrum[len(numbers):], label='Negative')
    plt.title(f'{real_or_synthetic} {chem_label} {preprocessing_type}', fontsize=20)
    plt.xlabel('Drift Time', fontsize=16)
    plt.ylabel('Ion Intensity', fontsize=16)
    plt.legend(fontsize=14)

    if save_plot_path is not None:
        plt.savefig(save_plot_path, format='png', dpi=300)
    plt.show()

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def plot_ims_spectra_pca(data, sample_size=1000):
    """
    Perform PCA on IMS spectra and plot the transformed data.

    This function generates a PCA scatter plot for IMS spectra.

    Parameters:
    ----------
    data : pd.DataFrame
        DataFrame containing IMS spectra data.

    Returns:
    -------
    None
        Displays the PCA scatter plot with IMS spectra.
    """

    sample = data.sample(n=sample_size, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sample.iloc[:, 2:-9])
    sample.iloc[:, 2:-9] = scaled_data

    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(sample.iloc[:, 2:-9])

    _, ax = plt.subplots(figsize=(8,6))

    # Create a color cycle for distinct colors
    color_cycle = plt.gca()._get_lines.prop_cycler

    all_chemical_names = list(sample.columns[-8:])

    for chem in all_chemical_names:
        color = next(color_cycle)['color']
        transformed_data = pca.transform(sample[sample['Label'] == chem].iloc[:, 2:-9])
        ax.scatter(transformed_data[:, 0], transformed_data[:, 1], color = color, label=chem)#, s=200)
        
    # Add legend
    legend1 = ax.legend(loc='upper right', title='Label')
    ax.add_artist(legend1)

    # Remove spines to reduce white space
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xticks([])
    plt.yticks([])
    plt.title(f'IMS Train Spectra PCA', fontsize=18)
    plt.show()

def plot_conditions_pca(
        condition_one, condition_two, save_file_path_pt1, 
        save_file_path_pt2, data_one_name, data_two_name, 
        sample_size=1000, z_score_threshold=2, 
        data_split = 'Condition', fit_to_all=True, 
        condition_two_start_idx=2, condition_two_stop_idx=-9,
        ):
    scaler = StandardScaler()
    if fit_to_all:
        # Fit scaler to all data
        full_data = pd.concat([condition_one, condition_two], ignore_index=True)
    else:
        # Fit scaler to condition one data only
        full_data = condition_one.copy()

    scaler.fit(full_data.iloc[:, 2:-9])
    
    # scaled_full_data = scaler.transform(full_data.iloc[:, 2:-9])
    full_data.iloc[:, 2:-9] = scaler.transform(full_data.iloc[:, 2:-9])#scaled_full_data
    # del scaled_full_data

    # scaled_data = scaler.transform(condition_one.iloc[:, 2:-9])
    condition_one.iloc[:, 2:-9] = scaler.transform(condition_one.iloc[:, 2:-9])#scaled_data
    # scaled_data = scaler.transform(condition_two.iloc[:, 2:-9])
    condition_two.iloc[:, condition_two_start_idx:condition_two_stop_idx] = scaler.transform(condition_two.iloc[:, condition_two_start_idx:condition_two_stop_idx])#scaled_data
    # del scaled_data
    print('here')
    all_chemical_names = list(condition_one.columns[-8:])

    for chem in all_chemical_names:
        print(chem)
        # Fit PCA on all spectra for a given chemical
        pca = PCA(n_components=2)
        full_data_chem = full_data[full_data['Label'] == chem]
        pca.fit(full_data_chem.iloc[:, 2:-9])
        del full_data_chem

        print(f'Plotting {chem}...')
        _, ax = plt.subplots(figsize=(12,8))
        condition_one_chem = condition_one[condition_one['Label'] == chem]
        if condition_one_chem.shape[0] > sample_size:
            condition_one_sample = condition_one_chem.sample(n=sample_size, random_state=42)
        else:
            condition_one_sample = condition_one_chem

        transformed_data = pca.transform(condition_one_sample.iloc[:, 2:-9])
        # Exclude outliers
        z_scores = np.abs(zscore(condition_one_sample.iloc[:, 2:-9]))
        filtered_data = condition_one_sample[(z_scores < z_score_threshold).all(axis=1)]
        
        transformed_data = pca.transform(filtered_data.iloc[:, 2:-9])
        ax.scatter(transformed_data[:, 0], transformed_data[:, 1], color='purple', label=f'{chem} {data_one_name}')

        if chem in list(condition_two['Label']):
            condition_two_chem = condition_two[condition_two['Label'] == chem]
            if condition_two_chem.shape[0] > sample_size:
                condition_two_sample = condition_two_chem.sample(n=sample_size, random_state=42)
            else:
                condition_two_sample = condition_two_chem
            # print(condition_two_sample.shape)
            z_scores = np.abs(zscore(condition_two_sample.iloc[:, condition_two_start_idx:condition_two_stop_idx]))
            filtered_data = condition_two_sample[(z_scores < z_score_threshold+1).all(axis=1)]
            # print(filtered_data.shape)
            
            if filtered_data.shape[0] > 1:
                transformed_data = pca.transform(filtered_data.iloc[:, condition_two_start_idx:condition_two_stop_idx])
                # transformed_data = pca.transform(condition_two_sample[condition_two_sample['Label'] == chem].iloc[:, 2:-9])
                ax.scatter(transformed_data[:, 0], transformed_data[:, 1], color ='blue', label=f'{chem} {data_two_name}', marker='x')
            else:
                print(f'Chem {chem} not in condition two data')
        else:
            print(f'Chem {chem} not in condition two data')
        
        # Add legend
        legend1 = ax.legend(loc='upper right', title='Label')
        ax.add_artist(legend1)

        plt.xticks([])
        plt.yticks([])
        plt.title(f'{chem} IMS Spectra by {data_split} PCA', fontsize=18)
        save_file_path = save_file_path_pt1 + chem + save_file_path_pt2
        plt.savefig(save_file_path, format='png', dpi=300)
        plt.show()

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------


def plot_similarity_comparison(
    spectra, chem, synthetic_spectra_df, 
    gen_type, first_cut_col, last_cut_col, 
    num_samples=100, similarity_type='per_pair',
    ):
    """
    Plots a comparison of the similarity between real and synthetic spectra for a given chemical.

    Parameters:
    spectra (pd.DataFrame): DataFrame containing the real spectra data.
    chem (str): The label of the chemical to compare.
    synthetic_spectra_df (pd.DataFrame): DataFrame containing the synthetic spectra data.
    gen_type (str): The type of generation method used for synthetic spectra.
    first_cut_col (int): The starting column index for the synthetic spectra data.
    last_cut_col (int): The ending column index for the synthetic spectra data.
    num_samples (int, optional): The number of samples to use for comparison. Default is 100.

    Returns:
    None: This function does not return any value. It displays a histogram plot comparing the similarity distributions.
    """

    chem_spectra = spectra[spectra['Label'] == chem].iloc[:, 2:-9]
    chem_subset = chem_spectra.sample(n=num_samples, random_state=42)
    chem_subset_list = np.array(chem_subset.values.tolist())
    mse_matrix = distance.cdist(chem_subset_list, chem_subset_list, 'euclidean')
    average_difference = np.mean(mse_matrix[np.triu_indices(mse_matrix.shape[0], k=1)])

    chem_subset_synthetic = synthetic_spectra_df.sample(n=num_samples, random_state=42).iloc[:, first_cut_col:last_cut_col]
    chem_subset_synthetic_list = np.array(chem_subset_synthetic.values.tolist())

    if similarity_type == 'per_pair':
        normalized_mse_matrix_real = mse_matrix.flatten() / average_difference

        mse_matrix_real_synthetic = distance.cdist(chem_subset_list, chem_subset_synthetic_list, 'euclidean').flatten()
        normalized_mse_matrix_real_synthetic = mse_matrix_real_synthetic / average_difference
    elif similarity_type == 'spect_avg':
        per_spectrum_average_similarity = np.mean(mse_matrix, axis=0)
        normalized_mse_matrix_real = per_spectrum_average_similarity / average_difference

        mse_matrix_real_synthetic = distance.cdist(chem_subset_list, chem_subset_synthetic_list, 'euclidean')
        per_spectrum_average_similarity = np.mean(mse_matrix_real_synthetic, axis=0)
        normalized_mse_matrix_real_synthetic = per_spectrum_average_similarity / average_difference
        
    sns.histplot(normalized_mse_matrix_real_synthetic, bins=8, kde=False, color='darkgreen', alpha=0.5, label='Synthetic')
    sns.histplot(normalized_mse_matrix_real, bins=8, kde=False, color='blue', alpha=0.5, label='Real')
    plt.xlabel('Similarity', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend()
    plt.title(f'{chem} Normalized Similarity {gen_type} Gen.', fontsize=16)
    plt.show()
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def preds_to_emb_pca_plot(
        predicted_embeddings, output_name_encodings, 
        sorted_chem_names, emb_df, 
        mass_spec_encoder_embeddings=False, mass_spec_chems=False
        ):
    """
    Generate and return data for PCA visualization of predicted embeddings alongside ChemNet embeddings.

    This function flattens the predicted embeddings and their corresponding chemical names, 
    optionally includes mass spectrometry embeddings, and prepares data for PCA plotting.

    Parameters:
    ----------
    predicted_embeddings : list of list of torch.Tensor
        A nested list of predicted embeddings, where each inner list contains tensors for a batch.

    output_name_encodings : list of list of torch.Tensor
        A nested list of one-hot encoded tensors representing the chemical names for the predicted embeddings.

    sorted_chem_names : list of str
        A list of chemical names corresponding to the indices of the one-hot encodings.

    emb_df : pandas.DataFrame
        A DataFrame containing true embeddings, with 'Embedding Floats' as one of its columns.

    mass_spec_encoder_embeddings : bool, optional
        If True, includes mass spectrometry encoder embeddings in the output.

    mass_spec_chems : list of str, optional
        A list of chemical names corresponding to mass spectrometry embeddings.

    Returns:
    -------
    tuple
        A tuple containing:
        - true_embeddings (pd.DataFrame): DataFrame of true embeddings used for comparison.
        - predicted_embeddings_flattened (list): Flattened list of predicted embeddings.
        - chem_names (list): List of chemical names corresponding to the predicted embeddings.
    """
    try:
        # Currently, preds and name encodings are lists of [n_batches, batch_size], flattening to lists of [n_samples]
        predicted_embeddings_flattened = [emb.cpu().detach().numpy() for emb_list in predicted_embeddings for emb in emb_list]
        chem_name_encodings_flattened = [enc.cpu() for enc_list in output_name_encodings for enc in enc_list]
    except AttributeError as e:
        predicted_embeddings_flattened = [emb for emb_list in predicted_embeddings for emb in emb_list]
        chem_name_encodings_flattened = [enc for enc_list in output_name_encodings for enc in enc_list]

    # Get chemical names from encodings
    chem_names = [sorted_chem_names[list(encoding).index(1)] for encoding in chem_name_encodings_flattened]

    if mass_spec_encoder_embeddings:
        for emb in mass_spec_encoder_embeddings:
            predicted_embeddings_flattened.append(torch.Tensor(emb))
        chem_names += mass_spec_chems

    try:
        # making list of all embeddings and chem names except for BKG
        embeddings = [emb for emb in emb_df['Embedding Floats']][1:]
        cols = emb_df.index[1:]
        true_embeddings = pd.DataFrame(embeddings).T
        true_embeddings.columns = cols
        
    except KeyError as e:
        if str(e) == "'Embedding Floats'":
            true_embeddings = emb_df
    
    return (true_embeddings, predicted_embeddings_flattened, chem_names)

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 
def add_hulls(ax, pca, chem_data, threshold=3):
    # embeddings_by_bkg_list = [chem_data.iloc[::2], chem_data.iloc[1::2]]
    # for df in embeddings_by_bkg_list:
    z_scores = np.abs(zscore(chem_data))
    threshold = threshold  # Adjust threshold as needed

    # Exclude outliers
    filtered_data = chem_data[(z_scores < threshold).all(axis=1)]
    # print(f'Excluded {round((len(chem_data) - len(filtered_data))/len(chem_data)*100)}% of data points as outliers.')

    transformed_data = pca.transform(filtered_data)
    hull = ConvexHull(transformed_data)
    for simplex in hull.simplices:
        ax.plot(transformed_data[simplex, 0], transformed_data[simplex, 1], 'r-') 
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 
def plot_emb_pca(
        all_embeddings, ims_embeddings, results_type, input_type, embedding_type='ChemNet', mass_spec_embeddings = None, log_wandb=False, 
        chemnet_embeddings_to_plot=None, mse_insert=None, insert_position=[0.05, 0.05], show_wandb_run_name=False, plot_hulls=False, hull_data=None):
    """
    This function performs Principal Component Analysis (PCA) on chemical embeddings and visualizes the results
    in a 2D scatter plot. It overlays additional data from ion mobility spectrometry (IMS) and mass spectrometry 
    if provided. The plot includes legends for different markers and can display the mean squared error (MSE) 
    and Weights & Biases (WandB) run name.

    Parameters:
    ----------
    all_embeddings : pd.DataFrame
        DataFrame containing all chemical embeddings to be plotted.

    ims_embeddings : pd.DataFrame
        DataFrame containing IMS embeddings, including a 'Label' column for chemical names.

    results_type : str
        A string indicating the type of results being plotted (train, val or test), used for title annotation.

    input_type : str
        A string indicating the type of input data (IMS, Carl or MNIST), used for legend annotation.

    embedding_type : str, optional
        A string specifying the type of embedding being plotted (ChemNet or OneHot). Default is 'ChemNet'.

    mass_spec_embeddings : pd.DataFrame, optional
        DataFrame containing mass spectrometry embeddings, including a 'Label' column. Default is None.

    log_wandb : bool, optional
        If True, logs the plot to Weights & Biases (WandB). Default is False.

    chemnet_embeddings_to_plot : pd.DataFrame, optional
        DataFrame containing specific ChemNet embeddings to plot. Default is None, which means all embeddings will be used.

    mse_insert : float, optional
        The mean squared error value to display in the plot. Default is None.

    insert_position : list of float, optional
        A list specifying the position to insert the MSE text in the plot, given as [x, y] in axis coordinates. Default is [0.05, 0.05].

    show_wandb_run_name : bool, optional
        If True, includes the current WandB run name in the plot. Default is True.

    Returns:
    -------
    None
        The function displays a PCA plot and optionally logs it to WandB.
    """
    pca = PCA(n_components=2)
    pca.fit(all_embeddings.T)

    if chemnet_embeddings_to_plot is not None:
        transformed_embeddings = pca.transform(chemnet_embeddings_to_plot.T)
        all_chemical_names = list(chemnet_embeddings_to_plot.columns)
    else:
        transformed_embeddings = pca.transform(all_embeddings.T) 
        all_chemical_names = list(all_embeddings.columns)

    _, ax = plt.subplots(figsize=(8,6))

    # Create a color cycle for distinct colors
    color_cycle = plt.gca()._get_lines.prop_cycler

    ims_labels = list(ims_embeddings['Label'])
    if mass_spec_embeddings is not None:
        mass_spec_labels=list(mass_spec_embeddings['Label'])
    else:
        mass_spec_labels = False
    
    # Scatter plot
    for chem in all_chemical_names:
        idx = all_chemical_names.index(chem)
        color = next(color_cycle)['color']
        # Plot ChemNet embeddings
        if idx < 8: # only label 1st 8 chemicals to avoid giant legend
            ax.scatter(transformed_embeddings[idx, 0], transformed_embeddings[idx, 1], color = color, label=chem, s=150, alpha=0.5)
        else:
            ax.scatter(transformed_embeddings[idx, 0], transformed_embeddings[idx, 1], color = color, s=150, alpha=0.5)

        # Transform encoder-generated ims_embeddings for the current chemical, if we have ims data for chem
        if chem in ims_labels:
            # transform all data for the given chemical. Exclude last col (label)
            ims_transformed = pca.transform(ims_embeddings[ims_embeddings['Label'] == chem].iloc[:, :-1])
            
            # Scatter plot for ims_embeddings with a different marker
            ax.scatter(ims_transformed[:, 0], ims_transformed[:, 1], marker='o', facecolors='none', edgecolors=color)#, s=200)#marker='x', color=color)#, s=75)
            if plot_hulls:
                if hull_data is None:
                    hull_data = ims_embeddings
                add_hulls(ax, pca, hull_data[hull_data['Label'] == chem].iloc[:, :-1])


        # repeat for mass spec
        if mass_spec_labels:
            if chem in mass_spec_labels:
                # transform all data for the given chemical. Exclude last col (label)
                mass_spec_transformed = pca.transform(mass_spec_embeddings[mass_spec_embeddings['Label'] == chem].iloc[:, :-1].values)
                
                # Scatter plot for mass_spec_embeddings with a different marker
                ax.scatter(mass_spec_transformed[:, 0], mass_spec_transformed[:, 1], marker='*', color=color)#, s=75)
    # Add legend
    legend1 = ax.legend(loc='upper right', title='Label')
    ax.add_artist(legend1)

    marker_legends = [
    plt.Line2D([0], [0], marker='o', color='w', label=embedding_type, markerfacecolor='black', markersize=6),
    plt.Line2D([0], [0], marker='o', color='w', label=input_type, markerfacecolor='none', markeredgecolor='black', markersize=6),
    ]
    
    if mass_spec_embeddings is not None:
        marker_legends.append(plt.Line2D([0], [0], marker='*', color='w', label='Mass Spec', markerfacecolor='black', markersize=10))

    # Add the second legend
    legend2 = ax.legend(handles=marker_legends, title='Marker Types', loc='upper left')
    ax.add_artist(legend2)

    if mse_insert is not None:
        # Add mse text in the corner with a box
        plt.text(insert_position[0], insert_position[1], f'MSE: {format(mse_insert, ".2e")}', 
            transform=plt.gca().transAxes,  # Use axis coordinates
            fontsize=14,
            verticalalignment='bottom',  # Align text to the top
            horizontalalignment='left',  # Align text to the right
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))  # Box properties
    
    if show_wandb_run_name == True:
        run_name = wandb.run.name
        # Add wandb run text in the corner
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.text(xlim[1] - 0.01 * (xlim[1] - xlim[0]),  # x position with an offset
                ylim[0] + 0.01 * (ylim[1] - ylim[0]),  # y position with an offset
                f'WandB run: {run_name}', 
                fontsize=8,
                verticalalignment='bottom',  # Align text to the top
                horizontalalignment='right',  # Align text to the right
                bbox=dict(facecolor='white', alpha=0.001, edgecolor='white'))

    plt.xticks([])
    plt.yticks([])
    if embedding_type != 'ChemNet':
        plt.title(f'{embedding_type} vs. Encoder {results_type} Output PCA', fontsize=18)
    else:
        plt.title(f'ChemNet vs. Encoder {results_type} Output PCA', fontsize=18)

    if log_wandb:
        plt.savefig('tmp_plot.png', format='png', dpi=300)
        wandb.log({'PCA of Predicted Chemical Embeddings': wandb.Image('tmp_plot.png')})

    plt.show()

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def plot_generation_results_pca(
        true_spectra, synthetic_spectra, chem_labels, results_type, sample_size=None, 
        chem_of_interest=None, log_wandb=False, mse_insert=None, 
        insert_position=[0.05, 0.05], show_wandb_run_name=False):
    
    pca = PCA(n_components=2)
    # assert true_spectra.shape[1] == 1677, "True spectra df does not have 1677 columns"
    pca.fit(true_spectra.iloc[:,:-1])

    _, ax = plt.subplots(figsize=(8,6))

    # Create a color cycle for distinct colors
    color_cycle = plt.gca()._get_lines.prop_cycler

    if sample_size is not None:
        true_spectra = true_spectra.sample(n = sample_size, random_state=42)

    # Scatter plot
    for chem, color in zip(chem_labels, color_cycle):
        # color = next(color_cycle)['color']
        color = color['color']
        # Transform data for the current chemical, exclude last col (label)
        transformed_true_spectra = pca.transform(true_spectra[true_spectra['Label'] == chem].iloc[:, :-1])
        # if specified, make makers larger for the chemical of interest
        if chem_of_interest is not None:
            if chem != chem_of_interest:
                ax.scatter(transformed_true_spectra[:, 0], transformed_true_spectra[:, 1], marker='o', label=chem, color=color, s=10)
            else:
                ax.scatter(transformed_true_spectra[:, 0], transformed_true_spectra[:, 1], marker='o', label=chem, color=color, s=100)
        else:
            ax.scatter(transformed_true_spectra[:, 0], transformed_true_spectra[:, 1], marker='o', label=chem, color=color)

        synthetic_chem = synthetic_spectra[synthetic_spectra['Label'] == chem].iloc[:, :-1]
        # only plot synthetic spectra if there are any for given chemical
        if synthetic_chem.shape[0] > 0:
            transformed_synthetic_spectra = pca.transform(synthetic_chem)
            # Scatter plot for synthetic spectra with a different marker
            ax.scatter(transformed_synthetic_spectra[:, 0], transformed_synthetic_spectra[:, 1], marker='*', color=color)

    # Add legend
    legend1 = ax.legend(loc='upper right', title='Label')
    ax.add_artist(legend1)

    marker_legends = [
    plt.Line2D([0], [0], marker='o', color='w', label='Experimental Spectra', markerfacecolor='black', markersize=6),
    plt.Line2D([0], [0], marker='*', color='w', label='Synthetic Spectra', markerfacecolor='black', markersize=10),
    ]
    

    # Add the second legend
    legend2 = ax.legend(handles=marker_legends, title='Marker Types', loc='upper left')
    ax.add_artist(legend2)

    if mse_insert is not None:
        # Add mse text in the corner with a box
        plt.text(insert_position[0], insert_position[1], f'MSE: {format(mse_insert, ".2e")}', 
            transform=plt.gca().transAxes,  # Use axis coordinates
            fontsize=14,
            verticalalignment='bottom',  # Align text to the top
            horizontalalignment='left',  # Align text to the right
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))  # Box properties
    
    if show_wandb_run_name == True:
        run_name = wandb.run.name
        # Add wandb run text in the corner
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.text(xlim[1] - 0.01 * (xlim[1] - xlim[0]),  # x position with an offset
                ylim[0] + 0.01 * (ylim[1] - ylim[0]),  # y position with an offset
                f'WandB run: {run_name}', 
                fontsize=8,
                verticalalignment='bottom',  # Align text to the top
                horizontalalignment='right',  # Align text to the right
                bbox=dict(facecolor='white', alpha=0.001, edgecolor='white'))

    plt.xticks([])
    plt.yticks([])
    plt.title(f'Experimental vs. Synthetic {results_type} Spectra PCA', fontsize=18)

    if log_wandb:
        plt.savefig('tmp_plot.png', format='png', dpi=300)
        wandb.log({'PCA of Experimental vs. Synthetic Spectra': wandb.Image('tmp_plot.png')})

    plt.show()


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def plot_generation_results_pca_single_chem_side_by_side(
        true_spectra, synthetic_spectra, chem_labels, results_type, sample_size=None, 
        chem_of_interest=None, log_wandb=False, mse_insert=None, 
        insert_position=[0.05, 0.05], show_wandb_run_name=False, 
        x_lims=None, y_lims=None, save_plot_path=None, 
        true_spectra_start_idx=0, true_spectra_stop_idx=-1, 
        synthetic_spectra_start_idx=0, synthetic_spectra_stop_idx=-1,
        ):
    
    # if pca is None:
    pca = PCA(n_components=2)
    pca.fit(true_spectra.iloc[:,true_spectra_start_idx:true_spectra_stop_idx])

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Setting x and y limits so that plot scales are the same
    if x_lims is not None:
        ax1.set_xlim(x_lims[0], x_lims[1])
        ax2.set_xlim(x_lims[0], x_lims[1])
    if y_lims is not None:
        ax1.set_ylim(y_lims[0], y_lims[1])
        ax2.set_ylim(y_lims[0], y_lims[1])
    if x_lims is None: # if plot boundaries are not specified, set them so that scales are consistent between plots
        true_transformed = pca.transform(true_spectra.iloc[:, true_spectra_start_idx:true_spectra_stop_idx])
        synthetic_transformed = pca.transform(synthetic_spectra.iloc[:, synthetic_spectra_start_idx:synthetic_spectra_stop_idx])
        x_lims = [
            min(true_transformed[:, 0].min(), synthetic_transformed[:, 0].min()) * 1.2, 
            max(true_transformed[:, 0].max(), synthetic_transformed[:, 0].max()) * 1.2
        ]
        y_lims = [
            min(true_transformed[:, 1].min(), synthetic_transformed[:, 1].min()) * 1.2,
            max(true_transformed[:, 1].max(), synthetic_transformed[:, 1].max()) * 1.2
        ]
    ax1.set_xlim(x_lims[0], x_lims[1])
    ax2.set_xlim(x_lims[0], x_lims[1])
    ax1.set_ylim(y_lims[0], y_lims[1])
    ax2.set_ylim(y_lims[0], y_lims[1])

    # Create a color cycle for distinct colors
    color_cycle = plt.gca()._get_lines.prop_cycler

    # Plot for true spectra
    for chem, color in zip(chem_labels, color_cycle):
        true_spectra_chem = true_spectra[true_spectra['Label'] == chem]
        if sample_size is not None and true_spectra_chem.shape[0] > sample_size:
            true_spectra_chem = true_spectra_chem.sample(n=sample_size, random_state=42)
        transformed_true_spectra = pca.transform(true_spectra_chem.iloc[:, true_spectra_start_idx:true_spectra_stop_idx])
        
        color = color['color']
        marker_size = 10

        # if chem_of_interest is not None:
        marker_size = 50 if chem == chem_of_interest else 10
        if chem == chem_of_interest:
            ax1.scatter(transformed_true_spectra[:, 0], transformed_true_spectra[:, 1], marker='o', label=chem, color=color, s=marker_size)
            synthetic_chem = synthetic_spectra[synthetic_spectra['Label'] == chem].iloc[:, synthetic_spectra_start_idx:synthetic_spectra_stop_idx]
    
            if synthetic_chem.shape[0] > 0:
                if sample_size is not None and synthetic_chem.shape[0] > sample_size:
                    synthetic_chem = synthetic_chem.sample(n=sample_size, random_state=42)
                    # print(synthetic_chem.shape)
                transformed_synthetic_spectra = pca.transform(synthetic_chem)
                ax2.scatter(transformed_synthetic_spectra[:, 0], transformed_synthetic_spectra[:, 1], marker='o', label=chem, color=color, s=marker_size)
        else:
            true_sample = true_spectra[true_spectra['Label'] == chem].iloc[:, true_spectra_start_idx:true_spectra_stop_idx].sample(n=10, random_state=42)
            transformed_sample = pca.transform(true_sample)
            ax1.scatter(transformed_sample[:, 0], transformed_sample[:, 1], marker='o', label=chem, color=color, s=marker_size)
            ax2.scatter(transformed_sample[:, 0], transformed_sample[:, 1], marker='o', label=chem, color=color, s=marker_size)       

    
    # if chem_of_interest is not None:
    ax1.set_title(f'Experimental {results_type} Spectra PCA {chem_of_interest}', fontsize=18)
    ax2.set_title(f'Synthetic {results_type} Spectra PCA {chem_of_interest}', fontsize=18)
    # else:
    #     ax1.set_title(f'Experimental {results_type} Spectra PCA', fontsize=18)
    #     ax2.set_title(f'Synthetic {results_type} Spectra PCA', fontsize=18)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.set_xticks([])
    ax2.set_yticks([])

    # Add legends
    ax1.legend(loc='upper right', title='Label')
    ax2.legend(loc='upper right', title='Label')

    if mse_insert is not None:
        plt.text(insert_position[0], insert_position[1], f'MSE: {format(mse_insert, ".2e")}', 
                 transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom',
                 horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))  
    
    if show_wandb_run_name:
        run_name = wandb.run.name
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.text(xlim[1] - 0.01 * (xlim[1] - xlim[0]), ylim[0] + 0.01 * (ylim[1] - ylim[0]),
                 f'WandB run: {run_name}', fontsize=8, verticalalignment='bottom',
                 horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.001, edgecolor='white'))

    plt.tight_layout()

    if log_wandb:
        plt.savefig('tmp_plot.png', format='png', dpi=300)
        wandb.log({'PCA of Experimental vs. Synthetic Spectra': wandb.Image('tmp_plot.png')})
    if save_plot_path is not None:
        # plot_path = os.path.join(save_plot_path, f'{chem_of_interest}_real_synthetic_pca_comparison.png')
        plt.savefig(save_plot_path, format='png', dpi=300)

    plt.show()

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 
def plot_pca(
    data, batch_size, model, device, encoder_criterion, sorted_chem_names, 
    all_embeddings_df, ims_embeddings_df, results_type, 
    input_type, embedding_type='ChemNet',
    show_wandb_run_name=True, log_wandb=True, 
    ):
    """
    Perform PCA on chemical embeddings and plot the transformed data.

    This function generates a PCA scatter plot for ChemNet embeddings, 
    including IMS and mass spectrometry embeddings if provided.

    Parameters:
    ----------
    all_embeddings : pd.DataFrame
        DataFrame containing ChemNet embeddings for all chemicals, with each column 
        representing one chemical's embedding.

    ims_embeddings : pd.DataFrame
        DataFrame containing IMS (ion mobility spectrometry) embeddings, must include 
        a 'Label' column with chemical names.

    mass_spec_embeddings : pd.DataFrame, optional
        DataFrame containing mass spectrometry embeddings, similar structure to `ims_embeddings`.
        Default is None.

    log_wandb : bool, optional
        If True, logs the generated plot to Weights & Biases (wandb). Default is False.

    chemnet_embeddings_to_plot : pd.DataFrame, optional
        DataFrame containing ChemNet embeddings specifically to be plotted.
    
    results_type: str
        Type of results - train, val, or test

    input_type : str
        The type of input data - IMS, Carl, MNIST, etc

    embedding_type : str, optional
        The type of embedding being visualized - ChemNet, OneHot, etc. Default is ChemNet.

    mse_insert : float, optional
        Mean Squared Error value to display on the plot.

    insert_position : list of float, optional
        Location in axis coordinates for MSE text insertion. Default is [0.05, 0.05].

    show_wandb_run_name : bool, optional
        If True, displays the current WandB run name on the plot. Default is True.

    Returns:
    -------
    None
        Displays the PCA scatter plot with ChemNet, IMS, and mass spec embeddings.

    Notes:
    -----
    - PCA is performed on the transpose of `all_embeddings` to align with IMS and mass spec data.
    """
    dataset = DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=False
    )

    preds, name_encodings, avg_loss, _ = f.predict_embeddings(dataset, model, device, encoder_criterion)
    true_embeddings, predicted_embeddings_flattened, chem_names = preds_to_emb_pca_plot(
        preds, name_encodings, sorted_chem_names, ims_embeddings_df,  
        )
    preds_df = pd.DataFrame(predicted_embeddings_flattened)
    preds_df['Label'] = chem_names
    
    plot_emb_pca(
        all_embeddings_df, preds_df, results_type=results_type, input_type=input_type,
        embedding_type=embedding_type, log_wandb=log_wandb, 
        chemnet_embeddings_to_plot=true_embeddings, mse_insert=avg_loss,
        show_wandb_run_name=show_wandb_run_name
        )

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def plot_carl_real_synthetic_comparison(
        true_carl, synthetic_carl, results_type, chem_label, 
        log_wandb=False, show_wandb_run_name=False, criterion=None, 
        run_name=None, save_plot_path=None, carl_or_spec='CARL',
        comparison_names=['Real', 'Synthetic']):
    
    _, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # x axis should run from lowest drift time (184) to highest drift time (184 + len(true_carl)//2)
    numbers = range(184, (len(true_carl)//2)+184)
    # y axis should run from min of both carls to max of both carls
    min_y = min(min(list(true_carl)), min(list(synthetic_carl))) - 200
    max_y = max(max(list(true_carl)), max(list(synthetic_carl))) + 200

    axes[0].plot(numbers, true_carl[:len(numbers)], label='Positive')
    axes[0].plot(numbers, true_carl[len(numbers):], label='Negative')
    axes[0].set_title(f'{comparison_names[0]} {results_type} {chem_label} {carl_or_spec}', fontsize=20)
    axes[0].set_xlabel('Drift Time', fontsize=16)
    axes[0].set_ylabel('Ion Intensity', fontsize=16)
    axes[0].set_ylim(min_y, max_y)
    axes[0].legend(fontsize=14)

    axes[1].plot(numbers, synthetic_carl[:len(numbers)], label='Positive')
    axes[1].plot(numbers, synthetic_carl[len(numbers):], label='Negative')
    axes[1].set_title(f'{comparison_names[1]} {results_type} {chem_label} {carl_or_spec}', fontsize=20)
    axes[1].set_xlabel('Drift Time', fontsize=16)
    axes[1].set_ylabel('Ion Intensity', fontsize=16)
    axes[1].set_ylim(min_y, max_y)
    axes[1].legend(fontsize=14)

    if criterion is not None:
        xlim = axes[0].get_xlim()
        ylim = axes[0].get_ylim()
        axes[0].text(xlim[1] - 0.02 * (xlim[1] - xlim[0]),  # x position with an offset
                    ylim[0] + 0.05 * (ylim[1] - ylim[0]),  # y position with an offset 
                    'Real vs. Synthetic MSE: {:.2e}'.format(criterion(torch.Tensor(true_carl), torch.Tensor(synthetic_carl)).item()),
                    fontsize=14,
                    verticalalignment='bottom',  # Align text to the bottom
                    horizontalalignment='right',  # Align text to the left
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    if show_wandb_run_name == True:
        if run_name is None:
            run_name = wandb.run.name
        # Add wandb run text in the corner
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.text(xlim[1] - 0.01 * (xlim[1] - xlim[0]),  # x position with an offset
                ylim[0] + 0.01 * (ylim[1] - ylim[0]),  # y position with an offset
                f'WandB run: {run_name}', 
                fontsize=8,
                verticalalignment='bottom',  # Align text to the bottom
                horizontalalignment='right',  # Align text to the right
                bbox=dict(facecolor='white', alpha=0.001, edgecolor='white'))

    if log_wandb:
        plt.savefig('tmp_plot.png', format='png', dpi=300)
        wandb.log({'Comparison of Real and Synthetic Data': wandb.Image('tmp_plot.png')})

    plt.tight_layout()

    if save_plot_path is not None:
        plt.savefig(save_plot_path, format='png', dpi=300)

    plt.show()
    plt.close()

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def format_data_for_plotting(data):
    assert data.columns[0] == 'Unnamed: 0', 'First column should be "Unnamed: 0"'
    if data.shape[1] == 1679:
        assert data.columns[-1] == 'Label', 'Last column should be "Label"'
        labels = list(data['Label'])
        return data.iloc[:,2:], labels
    if data.shape[1] == 1687:
        assert data.columns[-1] == 'TEPO', 'Last column should be "TEPO"'
        sorted_chem_names = data.columns[-8:]
        labels = list(data['Label'])
        return data.iloc[:,2:-8], sorted_chem_names, labels

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def plot_spectra_real_synthetic_comparison(true_spec, synthetic_spec, results_type, chem_label, left_plot_type='Experimental', right_plot_type='Synthetic', log_wandb=False, show_wandb_run_name=False, criterion=None, run_name=None):
    _, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # x axis should run from lowest drift time (184) to highest drift time (184 + len(true_carl)//2)
    numbers = range(184, (len(true_spec)//2)+184)
    # y axis should run from min of both carls to max of both carls
    min_y = min(list(true_spec)+list(synthetic_spec)) - 200
    max_y = max(list(true_spec)+list(synthetic_spec)) + 200

    axes[0].plot(numbers, true_spec[:len(numbers)], label='Positive')
    axes[0].plot(numbers, true_spec[len(numbers):], label='Negative')
    axes[0].set_title(f'{left_plot_type} {chem_label} {results_type} Spectrum', fontsize=20)
    axes[0].set_xlabel('Drift Time', fontsize=16)
    axes[0].set_ylabel('Ion Intensity', fontsize=16)
    axes[0].set_ylim(min_y, max_y)
    axes[0].legend(fontsize=14)

    axes[1].plot(numbers, synthetic_spec[:len(numbers)], label='Positive')
    axes[1].plot(numbers, synthetic_spec[len(numbers):], label='Negative')
    axes[1].set_title(f'{right_plot_type} {chem_label} {results_type} Spectrum', fontsize=20)
    axes[1].set_xlabel('Drift Time', fontsize=16)
    axes[1].set_ylabel('Ion Intensity', fontsize=16)
    axes[1].set_ylim(min_y, max_y)
    axes[1].legend(fontsize=14)

    if criterion is not None:
        xlim = axes[0].get_xlim()
        ylim = axes[0].get_ylim()
        axes[0].text(xlim[1] - 0.02 * (xlim[1] - xlim[0]),  # x position with an offset
                    ylim[0] + 0.05 * (ylim[1] - ylim[0]),  # y position with an offset 
                    'Real vs. Synthetic MSE: {:.2e}'.format(criterion(torch.Tensor(true_spec), torch.Tensor(synthetic_spec)).item()),
                    fontsize=14,
                    verticalalignment='bottom',  # Align text to the bottom
                    horizontalalignment='right',  # Align text to the left
                    bbox=dict(facecolor='white', alpha=1, edgecolor='black'))

    if show_wandb_run_name == True:
        if run_name is None:
            run_name = wandb.run.name
        # Add wandb run text in the corner
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.text(xlim[1] - 0.01 * (xlim[1] - xlim[0]),  # x position with an offset
                ylim[0] + 0.01 * (ylim[1] - ylim[0]),  # y position with an offset
                f'WandB run: {run_name}', 
                fontsize=8,
                verticalalignment='bottom',  # Align text to the bottom
                horizontalalignment='right',  # Align text to the right
                bbox=dict(facecolor='white', alpha=0.001, edgecolor='white'))

    if log_wandb:
        plt.savefig('tmp_plot.png', format='png', dpi=300)
        wandb.log({'Comparison of Experimental and Synthetic CARLs': wandb.Image('tmp_plot.png')})

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def plot_and_save_generator_results(
    data, batch_size, sorted_chem_names, model, device, 
    criterion, num_plots, plot_overlap_pca=False, 
    save_plots_to_wandb=True, show_wandb_run_name=True, 
    test_or_train='Train', carl_or_spec='CARL'
    ):
    # get predictions from trained model and plot them
    dataset = DataLoader(data, batch_size=batch_size)
    predicted_carls, output_name_encodings, _, _ = f.predict_embeddings(dataset, model, device, criterion)
    
    for _ in range(num_plots):
        random_carl = random.randint(0, len(data))
        encodings_list = [enc for enc_list in output_name_encodings for enc in enc_list]
        predicted_carls_list = [pred for pred_list in predicted_carls for pred in pred_list]       
        chem = sorted_chem_names[list(encodings_list[random_carl]).index(1)]
        plot_carl_real_synthetic_comparison(
            data[random_carl][2].cpu(), predicted_carls_list[random_carl], test_or_train, 
            chem, save_plots_to_wandb, show_wandb_run_name, carl_or_spec=carl_or_spec)

    if plot_overlap_pca:
        true_spectra = [spec[2].cpu() for spec in data]
        true_spectra_df = pd.DataFrame(true_spectra)
        spectra_labels = [sorted_chem_names[list(enc).index(1)] for enc in encodings_list]
        true_spectra_df['Labels'] = spectra_labels

        # synthetic_spectra_df = pd.DataFrame(predicted_carls_list)
        # print(synthetic_spectra_df.shape)
        # synthetic_spectra_df['Labels'] = spectra_labels
        # print('got here')
        # plot_generation_results_pca(
        #     true_spectra_df, synthetic_spectra_df.sample(n=1000, random_state=42), 
        #     sorted_chem_names, test_or_train, sample_size=1000, log_wandb=True,
        #     mse_insert=None, insert_position=[0.05, 0.05], show_wandb_run_name=True)
    

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
