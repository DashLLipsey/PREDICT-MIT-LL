# EI Only Dataset Processing
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
import function_depot as fd

# Load data ONCE
print("Loading EI only data...")
EI_only_spectra = pd.read_parquet('/home/dlipsey/MITLincolnLabs/MIT_LL_data/EI_only_spectra.parquet')
print(f"EI_only_spectra shape: {EI_only_spectra.shape}")

EI_only_subset = pd.read_parquet('/home/dlipsey/MITLincolnLabs/MIT_LL_data/EI_only_subset.parquet')
print(f"EI_only_subset shape: {EI_only_subset.shape}")

# Check for duplicate columns before processing
print(f"Checking for duplicate columns in EI_only_spectra...")
duplicate_cols = EI_only_spectra.columns[EI_only_spectra.columns.duplicated()].tolist()
if duplicate_cols:
    print(f"WARNING: Found {len(duplicate_cols)} duplicate columns: {duplicate_cols[:10]}")
    # Remove duplicate columns
    EI_only_spectra = EI_only_spectra.loc[:, ~EI_only_spectra.columns.duplicated()]
    print(f"After removing duplicates: {EI_only_spectra.shape}")

# CONVERT DATA TYPES FIRST
print("Converting spectral columns to float data types...")
spectral_cols = EI_only_spectra.columns[1:-5]
print(f"Converting {len(spectral_cols)} spectral columns to float")

# Convert column names to float
new_columns = []
for col in EI_only_spectra.columns:
    if col in spectral_cols:
        try:
            new_columns.append(float(col))
        except ValueError:
            print(f"Warning: Could not convert column '{col}' to float, keeping as is")
            new_columns.append(col)
    else:
        new_columns.append(col)

EI_only_spectra.columns = new_columns

# Convert spectral column values to float64
spectral_cols_float = [col for col in EI_only_spectra.columns if isinstance(col, float)]
print(f"Converting values in {len(spectral_cols_float)} columns to float64...")

for col in spectral_cols_float:
    EI_only_spectra[col] = pd.to_numeric(EI_only_spectra[col], errors='coerce').astype('float64')

print("Data type conversion complete.")

# Verify no duplicates after conversion
print(f"Final check - duplicate columns: {EI_only_spectra.columns.duplicated().sum()}")

# Define parameters
bin_sizes = [0.1, 0.5, 1, 2]
thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
save_directory = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/EI_only_dataframes"
print("Starting binning process...")
# Create all datasets with corrected indices
all_datasets = fd.binning_loop(EI_only_spectra, EI_only_subset, bin_sizes, thresholds, save_directory, 
                              indx_id_indx=-5,  # Points to 'index_id'
                              startindx=1,      # First spectral column  
                              stopindx=-5)      # Stop before metadata columns
print("Binning complete!")

























# # Basic Package Imports
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # sklearn
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.metrics import f1_score
# from sklearn.metrics import mean_squared_error, r2_score

# # imblearn
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler

# # Non-basic package imports
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# import requests

# # Packages I don't understand
# from fcd_torch import FCD
# import rdkit
# from collections import Counter
# import gc
# import pickle

# # Add the Python_files directory to the Python path
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'Python_files'))

# # Now you can import your modules
# import function_depot as fd

# # Load data ONCE
# print("Loading data...")
# merged_EI_MSMS_spectra = pd.read_parquet('/home/dlipsey/MITLincolnLabs/MIT_LL_data/merged_EI_MSMS_spectra.parquet')
# print(f"merged_EI_MSMS_spectra shape: {merged_EI_MSMS_spectra.shape}")

# merged_EI_MSMS_subset = pd.read_parquet('/home/dlipsey/MITLincolnLabs/MIT_LL_data/merged_EI_MSMS_subset.parquet')
# print(f"merged_EI_MSMS_subset shape: {merged_EI_MSMS_subset.shape}")

# # Check for duplicate columns before processing
# print(f"Checking for duplicate columns in merged_EI_MSMS_spectra...")
# duplicate_cols = merged_EI_MSMS_spectra.columns[merged_EI_MSMS_spectra.columns.duplicated()].tolist()
# if duplicate_cols:
#     print(f"WARNING: Found {len(duplicate_cols)} duplicate columns: {duplicate_cols[:10]}")
#     # Remove duplicate columns
#     merged_EI_MSMS_spectra = merged_EI_MSMS_spectra.loc[:, ~merged_EI_MSMS_spectra.columns.duplicated()]
#     print(f"After removing duplicates: {merged_EI_MSMS_spectra.shape}")

# # CONVERT DATA TYPES FIRST
# print("Converting spectral columns to float data types...")
# spectral_cols = merged_EI_MSMS_spectra.columns[1:-5]
# print(f"Converting {len(spectral_cols)} spectral columns to float")

# # Convert column names to float
# new_columns = []
# for col in merged_EI_MSMS_spectra.columns:
#     if col in spectral_cols:
#         try:
#             new_columns.append(float(col))
#         except ValueError:
#             print(f"Warning: Could not convert column '{col}' to float, keeping as is")
#             new_columns.append(col)
#     else:
#         new_columns.append(col)

# merged_EI_MSMS_spectra.columns = new_columns

# # Convert spectral column values to float64
# spectral_cols_float = [col for col in merged_EI_MSMS_spectra.columns if isinstance(col, float)]
# print(f"Converting values in {len(spectral_cols_float)} columns to float64...")

# for col in spectral_cols_float:
#     merged_EI_MSMS_spectra[col] = pd.to_numeric(merged_EI_MSMS_spectra[col], errors='coerce').astype('float64')

# print("Data type conversion complete.")

# # Verify no duplicates after conversion
# print(f"Final check - duplicate columns: {merged_EI_MSMS_spectra.columns.duplicated().sum()}")

# # Define parameters
# bin_sizes = [0.1, 0.5, 1, 2]
# thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
# save_directory = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/merged_EI_msms_dataframes"
# print("Starting binning process...")
# # Create all datasets with corrected indices
# all_datasets = fd.binning_loop(merged_EI_MSMS_spectra, merged_EI_MSMS_subset, bin_sizes, thresholds, save_directory, 
#                               indx_id_indx=-5,  # Points to 'index_id'
#                               startindx=1,      # First spectral column  
#                               stopindx=-5)      # Stop before metadata columns
# print("Binning complete!")






















# # Basic Package Imports
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # sklearn
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.metrics import f1_score
# from sklearn.metrics import mean_squared_error, r2_score

# # imblearn
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler

# # Non-basic package imports
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# import requests

# # Packages I don't understand
# from fcd_torch import FCD
# import rdkit
# from collections import Counter
# import gc
# import pickle

# # Add the Python_files directory to the Python path
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'Python_files'))

# # Now you can import your modules
# import function_depot as fd

# # Load data ONCE
# print("Loading data...")
# df6 = pd.read_csv("/home/dlipsey/MITLincolnLabs/MIT_LL_data/dataset_sep23.csv")
# print(f"df6 shape: {df6.shape}")

# df6_spectra = pd.read_parquet('/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet')
# print(f"df6_spectra shape: {df6_spectra.shape}")

# df6_subset = pd.read_parquet('/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet')
# print(f"df6_subset shape: {df6_subset.shape}")

# # Check for duplicate columns before processing
# print(f"Checking for duplicate columns in df6_spectra...")
# duplicate_cols = df6_spectra.columns[df6_spectra.columns.duplicated()].tolist()
# if duplicate_cols:
#     print(f"WARNING: Found {len(duplicate_cols)} duplicate columns: {duplicate_cols[:10]}")
#     # Remove duplicate columns
#     df6_spectra = df6_spectra.loc[:, ~df6_spectra.columns.duplicated()]
#     print(f"After removing duplicates: {df6_spectra.shape}")

# # CONVERT DATA TYPES FIRST
# print("Converting spectral columns to float data types...")
# spectral_cols = df6_spectra.columns[1:-5]
# print(f"Converting {len(spectral_cols)} spectral columns to float")

# # Convert column names to float
# new_columns = []
# for col in df6_spectra.columns:
#     if col in spectral_cols:
#         try:
#             new_columns.append(float(col))
#         except ValueError:
#             print(f"Warning: Could not convert column '{col}' to float, keeping as is")
#             new_columns.append(col)
#     else:
#         new_columns.append(col)

# df6_spectra.columns = new_columns

# # Convert spectral column values to float64
# spectral_cols_float = [col for col in df6_spectra.columns if isinstance(col, float)]
# print(f"Converting values in {len(spectral_cols_float)} columns to float64...")

# for col in spectral_cols_float:
#     df6_spectra[col] = pd.to_numeric(df6_spectra[col], errors='coerce').astype('float64')

# print("Data type conversion complete.")

# # Verify no duplicates after conversion
# print(f"Final check - duplicate columns: {df6_spectra.columns.duplicated().sum()}")
# # 0.5, 1, 2, 0.05, 0.1, 5, 10, 25, 50, 100, 200, 500, 1000
# # 0.01, 0.05, 0.1, 0.001, 0.005, 0.5, 1, 2, 5, 10, 50, 100
# # Define parameters
# bin_sizes = [0.1, 0.5, 1, 2] # 0.1, [5, 10, 25, 50] # [100, 200, 500]
# thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
# save_directory = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes_df6"
# print("Starting binning process...")
# # Create all datasets with corrected indices
# all_datasets = fd.binning_loop(df6_spectra, df6_subset, bin_sizes, thresholds, save_directory, 
#                               indx_id_indx=-5,  # Points to 'index_id'
#                               startindx=1,      # First spectral column  
#                               stopindx=-5)      # Stop before metadata columns
# print("Binning complete!")

















# # # Basic Package Imports
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # sklearn
# # from sklearn.preprocessing import OneHotEncoder
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.decomposition import PCA
# # from sklearn.model_selection import GridSearchCV
# # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# # from sklearn.metrics import f1_score
# # from sklearn.metrics import mean_squared_error, r2_score

# # # imblearn
# # from imblearn.over_sampling import RandomOverSampler
# # from imblearn.under_sampling import RandomUnderSampler

# # # Non-basic package imports
# # import torch
# # import torch.nn as nn
# # from torch.utils.data import TensorDataset, DataLoader
# # import requests

# # # Packages I don't understand
# # from fcd_torch import FCD
# # import rdkit
# # from collections import Counter
# # import gc
# # import pickle

# # # Add the Python_files directory to the Python path
# # import sys
# # import os
# # sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'Python_files'))

# # # Now you can import your modules
# # # import functions_enc as f
# # import function_depot as fd

# # df6 = pd.read_csv("/home/dlipsey/MITLincolnLabs/MIT_LL_data/dataset_sep23.csv")
# # print(df6.shape)
# # df6_spectra = pd.read_parquet('/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet')
# # print(df6_spectra.shape)
# # df6_subset = pd.read_parquet('/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet')
# # print(df6_subset.shape)

# # # Basic Package Imports
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # sklearn
# # from sklearn.preprocessing import OneHotEncoder
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.decomposition import PCA
# # from sklearn.model_selection import GridSearchCV
# # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# # from sklearn.metrics import f1_score
# # from sklearn.metrics import mean_squared_error, r2_score

# # # imblearn
# # from imblearn.over_sampling import RandomOverSampler
# # from imblearn.under_sampling import RandomUnderSampler

# # # Non-basic package imports
# # import torch
# # import torch.nn as nn
# # from torch.utils.data import TensorDataset, DataLoader
# # import requests

# # # Packages I don't understand
# # from fcd_torch import FCD
# # import rdkit
# # from collections import Counter
# # import gc
# # import pickle

# # # Add the Python_files directory to the Python path
# # import sys
# # import os
# # sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'Python_files'))

# # # Now you can import your modules
# # # import functions_enc as f
# # import function_depot as fd

# # df6 = pd.read_csv("/home/dlipsey/MITLincolnLabs/MIT_LL_data/dataset_sep23.csv")
# # print(df6.shape)
# # df6_spectra = pd.read_parquet('/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet')
# # print(df6_spectra.shape)
# # df6_subset = pd.read_parquet('/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_subset.parquet')
# # print(df6_subset.shape)

# # # CONVERT DATA TYPES FIRST - This is essential!
# # print("Converting spectral columns to float data types...")

# # # Get spectral column indices (exclude first column and last 4 columns)
# # spectral_cols = df6_spectra.columns[1:-4]
# # print(f"Converting {len(spectral_cols)} spectral columns to float")

# # # Step 1: Convert column names to float (they should already be floats, but ensuring)
# # new_columns = []
# # for col in df6_spectra.columns:
# #     if col in spectral_cols:
# #         # Convert column name to float
# #         new_columns.append(float(col))
# #     else:
# #         # Keep non-spectral columns as they are
# #         new_columns.append(col)

# # # Update column names
# # df6_spectra.columns = new_columns

# # # Step 2: Convert spectral column values to float64
# # spectral_cols_float = [col for col in df6_spectra.columns if isinstance(col, float)]
# # print(f"Converting values in {len(spectral_cols_float)} columns to float64...")

# # for col in spectral_cols_float:
# #     df6_spectra[col] = pd.to_numeric(df6_spectra[col], errors='coerce').astype('float64')

# # print("Conversion complete.")

# # # Verify the changes
# # print("\nVerifying data types:")
# # spectral_cols_final = [col for col in df6_spectra.columns if isinstance(col, float)]
# # print(f"Spectral columns data types:")
# # print(df6_spectra[spectral_cols_final].dtypes.value_counts())

# # # Check if all spectral columns are now float64
# # all_float = (df6_spectra[spectral_cols_final].dtypes == 'float64').all()
# # print(f"All spectral columns are float64: {all_float}")

# # # Save back to parquet with corrected data types
# # print("\nSaving corrected DataFrame to parquet...")
# # df6_spectra.to_parquet('/home/dlipsey/MITLincolnLabs/MIT_LL_data/df6_spectra.parquet', index=False)

# # print("Data type conversion complete. Now proceeding with binning...")

# # # Define your parameters
# # bin_sizes = [0.5, 1, 2] #0.05, 0.1, 5, 10, 25, 50, 100, 200, 500, 1000
# # thresholds = [0.01, 0.05, 0.1, ] #0.001, 0.005, 0.5, 1, 2, 5, 10, 50, 100
# # save_directory = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/grid_search_dataframes_df6"

# # # Create all datasets - NOW this should work
# # all_datasets = fd.binning_loop(df6_spectra, df6_subset, bin_sizes, thresholds, save_directory, indx_id_indx=-4, startindx=1, stopindx=-4)

# # # spectral_cols = df6_spectra.columns[1:-4]  # 2nd to 5th from last
# # # print(df6_spectra[spectral_cols].dtypes.value_counts())
















