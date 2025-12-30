### ==== IMPORTS ====
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import wandb
import itertools
import GPUtil
from collections import Counter, OrderedDict
import dask.dataframe as dd
import os
from fcd_torch import FCD
import seaborn as sns
import poetry
### ======================================================= WANDB CONFIGS ====================================================== ###
# These are default and basic, when running larger runs I will need to add more parameters in the pyhon files themselves
chemnet_config = {
        'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
        'wandb_project': 'MIT-Lincoln-Lab',
        'gpu':True,
        'embedding_type' : "ChemNet",
        'encoder_type' : "ChemNet Encoder"
    }
morganfp_config = {
        'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
        'wandb_project': 'MIT-Lincoln-Lab',
        'gpu':True,
        'embedding_type' : "Morgan Fingerprints",
        'encoder_type' : "Encoder"
    }
spectra_config = {
        'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
        'wandb_project': 'MIT-Lincoln-Lab',
        'gpu':True,
        'embedding_type' : "Spectra",
        'encoder_type' : "MLP"
    }

chemnet_mlp_config = {
        'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
        'wandb_project': 'MIT-Lincoln-Lab',
        'gpu':True,
        'embedding_type' : "ChemNet",
        'encoder_type' : "MLP"
    }

chemnet_tox_config = {
        'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
        'wandb_project': 'MIT-Lincoln-Lab',
        'gpu':True,
        'embedding_type' : "ChemNet + Toxicity",
        'encoder_type' : "Conditional Encoder"
    }
chemnet_tox_morgan_config = {
        'wandb_entity': 'dashlipsey-worcester-polytechnic-institute',
        'wandb_project': 'MIT-Lincoln-Lab',
        'gpu':True,
        'embedding_type' : "ChemNet + Toxicity + Morgan Fingerprints",
        'encoder_type' : "Conditional Encoder"
    }

### ======================================= Weighted Loss ======================================= ###

# Here is the code I got from copilot defining the weighted loss function, as a 2 two step function:
# In this T is the benchmark value, and alpha is the weight applied to values below T.
def weighted_loss(y_pred, y_true, alpha):
    T = torch.log(torch.tensor(100.0))  # ln(100)
    # base loss (MSE here, but could be MAE or Huber)
    base_loss = (y_pred - y_true) ** 2

    # weight function
    weights = torch.where(
        y_true <= T,
        torch.full_like(y_true, alpha),
        torch.ones_like(y_true)
    )

    # apply weights
    return (weights * base_loss).mean()

# Here we have the same process, but instead of 2 steps we have 4 to provide a more nuanced weighting scheme.
# In this case, T1, T2, and T3 are the benchmark values, and alpha1, alpha2, alpha3, and alpha4 are the weights for each range. The T
# values are embedded into the function as they are less likely to change than the alpha values.
def weighted_loss(y_pred, y_true, alpha1=4, alpha2=3, alpha3=2, alpha4=1):
    # Calculate log benchmarks
    T1 = torch.log(torch.tensor(5.0))    # ln(5)
    T2 = torch.log(torch.tensor(50.0))   # ln(50)
    T3 = torch.log(torch.tensor(100.0))  # ln(100)

    # Base loss (MSE here, but could be MAE or Huber)
    base_loss = (y_pred - y_true) ** 2

    # Weight function based on y_true ranges
    weights = torch.where(
        y_true <= T1,
        torch.full_like(y_true, alpha1),  # First range (y_true <= ln(5))
        torch.where(
            y_true <= T2,
            torch.full_like(y_true, alpha2),  # Second range (ln(5) < y_true <= ln(50))
            torch.where(
                y_true <= T3,
                torch.full_like(y_true, alpha3),  # Third range (ln(50) < y_true <= ln(100))
                torch.full_like(y_true, alpha4)   # Fourth range (y_true > ln(100))
            )
        )
    )

    # Apply weights to the base loss
    return (weights * base_loss).mean()


### ======================================================= ENCODERS ======================================================= ###
#%%

# This encoder sturucture is used for almost all of my encoders below until we get to the conditional ones
class base_Encoder(nn.Module):
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



# ChemNet Encoder
# batch_size = __
# epochs= __
# lr= 0.0001
# criterion = nn.MSELoss()
# output_size = 512
# num_layers = __

def train_model_chemnet_encoder(model, train_data, val_data, epochs, learning_rate, criterion, device, config = chemnet_config):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    wandb.init(entity=config['wandb_entity'],
               project=config['wandb_project'],
               config=config)    
    # Initialize lists to store losses
    train_losses = []
    val_losses = []

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
        wandb.log({"average_train_loss": average_train_loss})

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
        wandb.log({"average_val_loss": average_val_loss})

        # Store losses for this epoch
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)

        if epoch % 50 == 0 or epoch == epochs - 1:  
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'   Training loss: {average_train_loss:.6f}')
            print(f'   Validation loss: {average_val_loss:.6f}')
    wandb.finish()
    return model, train_losses, val_losses

#%%
# Morgan Fingerprint Encoder
# batch_size = __
# epochs = __
# lr = 0.0001
criterion = nn.MSELoss()
# output_size = 2048
# num_layers = __

# IMPORTANT NOTE: Morgan fingerprints are typically binary vectors (0s and 1s). So the normal 
# method of using MSELoss may not be the best choice. Consider using BCEWithLogitsLoss or BCELoss

def train_model_morgan_fp_encoder(model, train_data, val_data, epochs, learning_rate, criterion, device, config = morganfp_config):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    wandb.init(entity=config['wandb_entity'],
               project=config['wandb_project'],
               config=config) 
    # Initialize lists to store losses
    train_losses = []
    val_losses = []

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
        wandb.log({"average_train_loss": average_train_loss})

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
        wandb.log({"average_val_loss": average_val_loss})

        # Store losses for this epoch
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)

        if epoch % 50 == 0 or epoch == epochs - 1:  
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'   Training loss: {average_train_loss:.6f}')
            print(f'   Validation loss: {average_val_loss:.6f}')
    wandb.finish()
    return model, train_losses, val_losses

#%%
# Spectra Toxicity MLP
# batch_size = __
# epochs = __
# lr = 0.0001
# criterion = nn.MSELoss()
# output_size = 1
# num_layers = __

def train_model_MLP_spectra(model, train_data, val_data, epochs, learning_rate, criterion, device, config = spectra_config):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    wandb.init(entity=config['wandb_entity'],
               project=config['wandb_project'],
               config=config) 
    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch, true_log_tox in train_data:
            batch = batch.to(device)
            true_log_tox = true_log_tox.to(device)

            optimizer.zero_grad()
            batch_predicted_log_tox = model(batch)
            loss = criterion(batch_predicted_log_tox, true_log_tox)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_train_loss = running_loss / len(train_data)
        wandb.log({"average_train_loss": average_train_loss})

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch, val_true_tox in val_data:
                val_batch = val_batch.to(device)
                val_true_tox = val_true_tox.to(device)

                val_batch_predicted_tox = model(val_batch)

                val_batch_loss = criterion(val_batch_predicted_tox, val_true_tox)
                val_loss += val_batch_loss.item()
        average_val_loss = val_loss / len(val_data)
        wandb.log({"average_val_loss": average_val_loss})

        # Store losses for this epoch
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)

        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'   Training loss: {average_train_loss:.6f}')
        print(f'   Validation loss: {average_val_loss:.6f}')
    wandb.finish()
    return model, train_losses, val_losses


# GENERAL TOXICITY MLP
def train_model_MLP(model, train_data, val_data, epochs, learning_rate, criterion, device, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    wandb.init(entity=config['wandb_entity'],
               project=config['wandb_project'],
               config=config) 
    # Initialize lists to store losses
    train_losses = []
    val_losses = []

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
        average_train_loss = running_loss / len(train_data)
        wandb.log({"average_train_loss": average_train_loss})

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch, val_true_tox, _ in val_data:
                val_batch = val_batch.to(device)
                val_true_tox = val_true_tox.to(device)

                val_batch_predicted_tox = model(val_batch)

                val_batch_loss = criterion(val_batch_predicted_tox, val_true_tox)
                val_loss += val_batch_loss.item()
        average_val_loss = val_loss / len(val_data)
        wandb.log({"average_val_loss": average_val_loss})

        # Store losses for this epoch
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)

        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'   Training loss: {average_train_loss:.6f}')
        print(f'   Validation loss: {average_val_loss:.6f}')
    wandb.finish()
    return model, train_losses, val_losses

# Conditional encoder (ChemNet + Toxicity) 
# batch_size = __
# epochs = __
# lr = 0.0001
# criterion1 = nn.MSELoss()
# criterion2 = nn.MSELoss()
# output_size = 513
# num_layers = __
lambda1 = 1
lambda2 = 5

# Smaller conditional encoder architecture
class Cond_Encoder_12(nn.Module): # From Cond_Encoder_chemnet_tox
    def __init__(self, input_size, output_size, num_layers):
        super().__init__()
        self.max_tox_value = np.log(10000)  # Maximum value for sigmoid scaling
        
        layers = []
        layer_sizes = np.linspace(input_size, output_size, num_layers + 1, dtype=int)
        for i in range(num_layers):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < num_layers - 1:
                layers.append(nn.LeakyReLU(inplace=True))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        output = self.encoder(x)
        
        # Split the output
        embedding_output = output[:, :512]  # ChemNet embeddings (no activation)
        toxicity_raw = output[:, 512:]      # Raw toxicity output
        
        # Apply scaled sigmoid only to toxicity part
        toxicity_output = torch.sigmoid(toxicity_raw) * self.max_tox_value
        
        # Concatenate back together
        final_output = torch.cat([embedding_output, toxicity_output], dim=1)
        
        return final_output

# From train_model_condenc_chemnet_tox
def train_model_condenc_12(model, train_data, val_data, epochs, learning_rate, criterion1, criterion2, 
                                    lambda1, lambda2, device, config = chemnet_tox_config):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    wandb.init(entity=config['wandb_entity'],
               project=config['wandb_project'],
               config=config) 
    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    # New: Store individual loss components (after lambda weighting)
    train_embedding_losses = []
    train_toxicity_losses = []
    val_embedding_losses = []
    val_toxicity_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_embedding_loss = 0.0
        running_toxicity_loss = 0.0
        
        for batch, true_embeddings, true_log_tox, _ in train_data:
            batch = batch.to(device)
            true_embeddings = true_embeddings.to(device)
            true_log_tox = true_log_tox.to(device)

            optimizer.zero_grad()
            batch_predicted_combined = model(batch)
            
            # Embedding Loss
            batch_predicted_embeddings = batch_predicted_combined[:, :512]
            loss1 = criterion1(batch_predicted_embeddings, true_embeddings)
            # Response Loss
            batch_predicted_log_tox = batch_predicted_combined[:, 512:]
            loss2 = criterion2(batch_predicted_log_tox, true_log_tox)
            
            # Apply lambda weighting
            weighted_loss1 = lambda1 * loss1
            weighted_loss2 = lambda2 * loss2
            
            # Loss function with modular weights (lambda1 and lambda2)
            total_loss = weighted_loss1 + weighted_loss2

            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            running_loss += total_loss.item()
            running_embedding_loss += weighted_loss1.item()
            running_toxicity_loss += weighted_loss2.item()
            
        average_train_loss = running_loss / len(train_data)
        average_train_embedding_loss = running_embedding_loss / len(train_data)
        average_train_toxicity_loss = running_toxicity_loss / len(train_data)
        wandb.log({"average_train_loss": average_train_loss})
        wandb.log({"average_train_embedding_loss": average_train_embedding_loss})
        wandb.log({"average_train_toxicity_loss": average_train_toxicity_loss})
        
        model.eval()
        val_loss = 0.0
        val_embedding_loss = 0.0
        val_toxicity_loss = 0.0
        
        with torch.no_grad():
            for val_batch, val_true_embeddings, val_true_tox, _ in val_data:
                val_batch = val_batch.to(device)
                val_true_embeddings = val_true_embeddings.to(device)
                val_true_tox = val_true_tox.to(device)

                val_batch_predicted = model(val_batch)
                val_batch_predicted_embeddings = val_batch_predicted[:, :512]
                val_batch_predicted_tox = val_batch_predicted[:, 512:]

                # Calculate individual losses
                val_loss1 = criterion1(val_batch_predicted_embeddings, val_true_embeddings)
                val_loss2 = criterion2(val_batch_predicted_tox, val_true_tox)
                
                # Apply lambda weighting
                val_weighted_loss1 = lambda1 * val_loss1
                val_weighted_loss2 = lambda2 * val_loss2
                
                # Accumulate losses
                val_loss += (val_weighted_loss1 + val_weighted_loss2).item()
                val_embedding_loss += val_weighted_loss1.item()
                val_toxicity_loss += val_weighted_loss2.item()
                
        average_val_loss = val_loss / len(val_data)
        average_val_embedding_loss = val_embedding_loss / len(val_data)
        average_val_toxicity_loss = val_toxicity_loss / len(val_data)
        wandb.log({"average_val_loss": average_val_loss})
        wandb.log({"average_val_embedding_loss": average_val_embedding_loss})
        wandb.log({"average_val_toxicity_loss": average_val_toxicity_loss})

        # Store losses for this epoch
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)
        train_embedding_losses.append(average_train_embedding_loss)
        train_toxicity_losses.append(average_train_toxicity_loss)
        val_embedding_losses.append(average_val_embedding_loss)
        val_toxicity_losses.append(average_val_toxicity_loss)

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'   Training loss: {average_train_loss:.6f}')
            print(f'   Training embedding loss: {average_train_embedding_loss:.6f}')
            print(f'   Training toxicity loss: {average_train_toxicity_loss:.6f}')
            print(f'   Validation loss: {average_val_loss:.6f}')
            print(f'   Validation embedding loss: {average_val_embedding_loss:.6f}')
            print(f'   Validation toxicity loss: {average_val_toxicity_loss:.6f}')
    wandb.finish()
    return model, train_losses, val_losses, train_embedding_losses, train_toxicity_losses, val_embedding_losses, val_toxicity_losses


#%%
# Conditional encoder (ChemNet + Toxicity + Morgan Fingerprints) 
# From train_model_condenc_chemnet_tox_morgan
def train_model_condenc_123(model, train_data, val_data, epochs, learning_rate, criterion1, criterion2, criterion3, 
                                           lambda1, lambda2, lambda3, device, config = chemnet_tox_morgan_config):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    wandb.init(entity=config['wandb_entity'],
               project=config['wandb_project'],
               config=config) 
               
    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    
    # Store individual loss components (after lambda weighting)
    train_embedding_losses = []
    train_toxicity_losses = []
    train_morgan_losses = []
    val_embedding_losses = []
    val_toxicity_losses = []
    val_morgan_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_embedding_loss = 0.0
        running_toxicity_loss = 0.0
        running_morgan_loss = 0.0
        
        for batch, true_embeddings, true_log_tox, true_morgan, _ in train_data:
            batch = batch.to(device)
            true_embeddings = true_embeddings.to(device)
            true_log_tox = true_log_tox.to(device)
            true_morgan = true_morgan.to(device)

            optimizer.zero_grad()
            batch_predicted_combined = model(batch)
            
            # Embedding Loss
            batch_predicted_embeddings = batch_predicted_combined[:, :512] # First 512 columns
            loss1 = criterion1(batch_predicted_embeddings, true_embeddings) # loss1 (embedding loss)
            # Response Loss
            batch_predicted_log_tox = batch_predicted_combined[:, 512:513] # 512th column
            loss2 = criterion2(batch_predicted_log_tox, true_log_tox) # loss2 (toxicity loss)
            # Morgan Loss
            batch_predicted_morgan = batch_predicted_combined[:, 513:] # Last 2048 columns
            loss3 = criterion3(batch_predicted_morgan, true_morgan) # loss3 (morgan loss)

            # Apply lambda weighting
            weighted_loss1 = lambda1 * loss1
            weighted_loss2 = lambda2 * loss2
            weighted_loss3 = lambda3 * loss3
            
            # Total loss with modular weights
            total_loss = weighted_loss1 + weighted_loss2 + weighted_loss3

            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            running_loss += total_loss.item()
            running_embedding_loss += weighted_loss1.item()
            running_toxicity_loss += weighted_loss2.item()
            running_morgan_loss += weighted_loss3.item()
            
        average_train_loss = running_loss / len(train_data)
        average_train_embedding_loss = running_embedding_loss / len(train_data)
        average_train_toxicity_loss = running_toxicity_loss / len(train_data)
        average_train_morgan_loss = running_morgan_loss / len(train_data)
        wandb.log({"average_train_loss": average_train_loss})
        wandb.log({"average_train_embedding_loss": average_train_embedding_loss})
        wandb.log({"average_train_toxicity_loss": average_train_toxicity_loss})
        wandb.log({"average_train_morgan_loss": average_train_morgan_loss})

        model.eval()
        val_loss = 0.0
        val_embedding_loss = 0.0
        val_toxicity_loss = 0.0
        val_morgan_loss = 0.0
        
        with torch.no_grad():
            for val_batch, val_true_embeddings, val_true_tox, val_true_morgan, _ in val_data:
                val_batch = val_batch.to(device)
                val_true_embeddings = val_true_embeddings.to(device)
                val_true_tox = val_true_tox.to(device)
                val_true_morgan = val_true_morgan.to(device)

                val_batch_predicted = model(val_batch)
                val_batch_predicted_embeddings = val_batch_predicted[:, :512]
                val_batch_predicted_tox = val_batch_predicted[:, 512:513]
                val_batch_predicted_morgan = val_batch_predicted[:, 513:]

                # Calculate individual losses
                val_loss1 = criterion1(val_batch_predicted_embeddings, val_true_embeddings)
                val_loss2 = criterion2(val_batch_predicted_tox, val_true_tox)
                val_loss3 = criterion3(val_batch_predicted_morgan, val_true_morgan)
                
                # Apply lambda weighting
                val_weighted_loss1 = lambda1 * val_loss1
                val_weighted_loss2 = lambda2 * val_loss2
                val_weighted_loss3 = lambda3 * val_loss3
                
                # Accumulate losses
                val_loss += (val_weighted_loss1 + val_weighted_loss2 + val_weighted_loss3).item()
                val_embedding_loss += val_weighted_loss1.item()
                val_toxicity_loss += val_weighted_loss2.item()
                val_morgan_loss += val_weighted_loss3.item()
                
        average_val_loss = val_loss / len(val_data)
        average_val_embedding_loss = val_embedding_loss / len(val_data)
        average_val_toxicity_loss = val_toxicity_loss / len(val_data)
        average_val_morgan_loss = val_morgan_loss / len(val_data)
        wandb.log({"average_val_loss": average_val_loss})
        wandb.log({"average_val_embedding_loss": average_val_embedding_loss})
        wandb.log({"average_val_toxicity_loss": average_val_toxicity_loss})
        wandb.log({"average_val_morgan_loss": average_val_morgan_loss})

        # Store losses for this epoch
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)
        train_embedding_losses.append(average_train_embedding_loss)
        train_toxicity_losses.append(average_train_toxicity_loss)
        train_morgan_losses.append(average_train_morgan_loss)
        val_embedding_losses.append(average_val_embedding_loss)
        val_toxicity_losses.append(average_val_toxicity_loss)
        val_morgan_losses.append(average_val_morgan_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch [{epoch+1}/{epochs}]')        
            print(f'   Training loss: {average_train_loss:.6f}')
            print(f'   Training embedding loss: {average_train_embedding_loss:.6f}')
            print(f'   Training toxicity loss: {average_train_toxicity_loss:.6f}')
            print(f'   Training morgan loss: {average_train_morgan_loss:.6f}')
            print(f'   Validation loss: {average_val_loss:.6f}')
            print(f'   Validation embedding loss: {average_val_embedding_loss:.6f}')
            print(f'   Validation toxicity loss: {average_val_toxicity_loss:.6f}')
            print(f'   Validation morgan loss: {average_val_morgan_loss:.6f}')
    wandb.finish()
    return model, train_losses, val_losses, train_embedding_losses, train_toxicity_losses, train_morgan_losses, val_embedding_losses, val_toxicity_losses, val_morgan_losses

# Conditional encoder (ChemNet + Toxicity + Morgan Fingerprints + group(external condition)) 
# MUST BE ADJUSTED TO INCLUDE GROUP CONDITION
# batch_size = __
# epochs = __
# lr = 0.0001
# criterion1 = nn.MSELoss()
# criterion2 = nn.MSELoss()
# criterion3 = nn.MSELoss()
# output_size = 2561
# num_layers = __
# lambda1 = 1
# lambda2 = 5
# lambda3 = 1

### ================================================= AFFINE TRANSFORMATION ================================================= ###
def affine_trans_sig(z, z_min, z_max, IS):
    """
    PyTorch-compatible affine transformation for use in neural networks.
    Maps values from range [z_min, z_max] to the Identity Segment range.
    
    Parameters:
    - z: torch.Tensor of predicted values from network
    - z_min: float, minimum value of the input range (network output range)
    - z_max: float, maximum value of the input range (network output range)
    - IS: tuple/list with two floats, (a, b), the Identity Segment range
    
    Returns:
    - Fz: transformed z values mapped to IS range (same shape as z)
    """
    a, b = IS
    
    # Avoid division by zero
    if z_max == z_min:
        return torch.full_like(z, (a + b) / 2)
    
    # Calculate alpha and beta for affine transformation
    # Maps [z_min, z_max] -> [a, b]
    alpha = (b - a) / (z_max - z_min)
    beta = a - alpha * z_min
    
    Fz = alpha * z + beta
    return Fz

def inv_affine_trans_sig(Fz, target_min, target_max, IS):
    """
    PyTorch-compatible inverse affine transformation.
    Maps values from Identity Segment range to target range [target_min, target_max].
    
    Parameters:
    - Fz: torch.Tensor of values in IS range (after sigmoid)
    - target_min: float, minimum value of the target output range
    - target_max: float, maximum value of the target output range
    - IS: tuple/list with two floats, (a, b), the Identity Segment range
    
    Returns:
    - z: values mapped to target range (same shape as Fz)
    """
    a, b = IS
    
    # Avoid division by zero
    if target_max == target_min:
        return torch.full_like(Fz, (target_min + target_max) / 2)
    
    # Calculate alpha and beta for inverse transformation
    # Maps [a, b] -> [target_min, target_max]
    alpha = (target_max - target_min) / (b - a)
    beta = target_min - alpha * a
    
    z = alpha * (Fz) + beta
    return z

# Conditional Encoder architecture
class Cond_Encoder_123_affine(nn.Module): # From Cond_encoder_full
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

        output = self.encoder(x)
        
        # Split the output into three parts
        embedding_output = output[:, :512]    # ChemNet embeddings
        toxicity_raw = output[:, 512:513]     # Raw toxicity output (1 column)
        morgan_output = output[:, 513:]       # Morgan fingerprints
        
        # Apply sigmoid activation to each part with their appropriate ranges
        
        # Embedding processing (range: -1 to 1)
        embedding_transformed = affine_trans_sig(embedding_output, -2, 2, (-0.5, 0.5))
        embedding_sigmoid = torch.sigmoid(4 * (embedding_transformed)) - 0.5
        embedding_final = inv_affine_trans_sig(embedding_sigmoid, -2, 2, (-0.5, 0.5))
        
        # Toxicity processing (range: 0 to log(max_tox))
        toxicity_transformed = affine_trans_sig(toxicity_raw, -10.0, np.log(100000), (-0.5, 0.5)) # np.log(100000), np.log(46965.46394)
        toxicity_sigmoid = torch.sigmoid(4 * (toxicity_transformed))  - 0.5
        toxicity_final = inv_affine_trans_sig(toxicity_sigmoid, -10.0, np.log(100000), (-0.5, 0.5))

        # Morgan processing (range: 0 to 1)
        morgan_transformed = affine_trans_sig(morgan_output, 0, 1.0, (-0.2, 0.2))
        morgan_sigmoid = torch.sigmoid(4 * (morgan_transformed)) - 0.5
        morgan_final = inv_affine_trans_sig(morgan_sigmoid, 0, 1.0, (-0.2, 0.2))
        
        # Concatenate back together
        final_output = torch.cat([embedding_final, toxicity_final, morgan_final], dim=1)
        # final_output = torch.cat([embedding_output, toxicity_final, morgan_output], dim=1)

        return final_output


 # Here are the older versions without the affine transforms but one with the tox bound    
class Cond_Encoder_123(nn.Module): # From Cond_Encoder_full
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
        output = self.encoder(x)
        
        # Split the output into three parts
        embedding_output = output[:, :512]    # ChemNet embeddings (no activation)
        toxicity_raw = output[:, 512:513]     # Raw toxicity output (1 column)
        morgan_output = output[:, 513:]       # Morgan fingerprints (no activation)
        
        # Apply scaled sigmoid only to toxicity part
        toxicity_output = torch.sigmoid(toxicity_raw) * np.log(100000) 
        # embedding_output = torch.sigmoid(embedding_output) * 3 
        # morgan_output = torch.sigmoid(morgan_output) * 2

        # Concatenate back together
        final_output = torch.cat([embedding_output, toxicity_output, morgan_output], dim=1)
        
        return final_output

def train_model_condenc_123e1(model, train_data, val_data, epochs, learning_rate, criterion1, criterion2, criterion3, 
                                           lambda1, lambda2, lambda3, device, config = chemnet_tox_morgan_config):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    wandb.init(entity=config['wandb_entity'],
               project=config['wandb_project'],
               config=config) 
               
    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    
    # Store individual loss components (after lambda weighting)
    train_embedding_losses = []
    train_toxicity_losses = []
    train_morgan_losses = []
    val_embedding_losses = []
    val_toxicity_losses = []
    val_morgan_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_embedding_loss = 0.0
        running_toxicity_loss = 0.0
        running_morgan_loss = 0.0
        
        # Modified: batch now includes group information as part of input
        for batch_with_group, true_embeddings, true_log_tox, true_morgan, _ in train_data:
            batch_with_group = batch_with_group.to(device)  # Input includes spectra + group encoding
            true_embeddings = true_embeddings.to(device)
            true_log_tox = true_log_tox.to(device)
            true_morgan = true_morgan.to(device)

            optimizer.zero_grad()
            batch_predicted_combined = model(batch_with_group)  # Forward pass with group info
            
            # Embedding Loss
            batch_predicted_embeddings = batch_predicted_combined[:, :512] # First 512 columns
            loss1 = criterion1(batch_predicted_embeddings, true_embeddings) # loss1 (embedding loss)
            # Response Loss
            batch_predicted_log_tox = batch_predicted_combined[:, 512:513] # 512th column
            loss2 = criterion2(batch_predicted_log_tox, true_log_tox) # loss2 (toxicity loss)
            # Morgan Loss
            batch_predicted_morgan = batch_predicted_combined[:, 513:] # Last 2048 columns
            loss3 = criterion3(batch_predicted_morgan, true_morgan) # loss3 (morgan loss)

            # Apply lambda weighting
            weighted_loss1 = lambda1 * loss1
            weighted_loss2 = lambda2 * loss2
            weighted_loss3 = lambda3 * loss3
            
            # Total loss with modular weights (group is NOT included in loss)
            total_loss = weighted_loss1 + weighted_loss2 + weighted_loss3

            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            running_loss += total_loss.item()
            running_embedding_loss += weighted_loss1.item()
            running_toxicity_loss += weighted_loss2.item()
            running_morgan_loss += weighted_loss3.item()
            
        average_train_loss = running_loss / len(train_data)
        average_train_embedding_loss = running_embedding_loss / len(train_data)
        average_train_toxicity_loss = running_toxicity_loss / len(train_data)
        average_train_morgan_loss = running_morgan_loss / len(train_data)
        wandb.log({"average_train_loss": average_train_loss})
        wandb.log({"average_train_embedding_loss": average_train_embedding_loss})
        wandb.log({"average_train_toxicity_loss": average_train_toxicity_loss})
        wandb.log({"average_train_morgan_loss": average_train_morgan_loss})

        model.eval()
        val_loss = 0.0
        val_embedding_loss = 0.0
        val_toxicity_loss = 0.0
        val_morgan_loss = 0.0
        
        with torch.no_grad():
            # Modified: validation batch also includes group information
            for val_batch_with_group, val_true_embeddings, val_true_tox, val_true_morgan, _ in val_data:
                val_batch_with_group = val_batch_with_group.to(device)  # Input includes spectra + group encoding
                val_true_embeddings = val_true_embeddings.to(device)
                val_true_tox = val_true_tox.to(device)
                val_true_morgan = val_true_morgan.to(device)

                val_batch_predicted = model(val_batch_with_group)  # Forward pass with group info
                val_batch_predicted_embeddings = val_batch_predicted[:, :512]
                val_batch_predicted_tox = val_batch_predicted[:, 512:513]
                val_batch_predicted_morgan = val_batch_predicted[:, 513:]

                # Calculate individual losses (group is NOT included in loss calculation)
                val_loss1 = criterion1(val_batch_predicted_embeddings, val_true_embeddings)
                val_loss2 = criterion2(val_batch_predicted_tox, val_true_tox)
                val_loss3 = criterion3(val_batch_predicted_morgan, val_true_morgan)
                
                # Apply lambda weighting
                val_weighted_loss1 = lambda1 * val_loss1
                val_weighted_loss2 = lambda2 * val_loss2
                val_weighted_loss3 = lambda3 * val_loss3
                
                # Accumulate losses
                val_loss += (val_weighted_loss1 + val_weighted_loss2 + val_weighted_loss3).item()
                val_embedding_loss += val_weighted_loss1.item()
                val_toxicity_loss += val_weighted_loss2.item()
                val_morgan_loss += val_weighted_loss3.item()
                
        average_val_loss = val_loss / len(val_data)
        average_val_embedding_loss = val_embedding_loss / len(val_data)
        average_val_toxicity_loss = val_toxicity_loss / len(val_data)
        average_val_morgan_loss = val_morgan_loss / len(val_data)
        wandb.log({"average_val_loss": average_val_loss})
        wandb.log({"average_val_embedding_loss": average_val_embedding_loss})
        wandb.log({"average_val_toxicity_loss": average_val_toxicity_loss})
        wandb.log({"average_val_morgan_loss": average_val_morgan_loss})

        # Store losses for this epoch
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)
        train_embedding_losses.append(average_train_embedding_loss)
        train_toxicity_losses.append(average_train_toxicity_loss)
        train_morgan_losses.append(average_train_morgan_loss)
        val_embedding_losses.append(average_val_embedding_loss)
        val_toxicity_losses.append(average_val_toxicity_loss)
        val_morgan_losses.append(average_val_morgan_loss)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'   Training loss: {average_train_loss:.6f}')
            print(f'   Training embedding loss: {average_train_embedding_loss:.6f}')
            print(f'   Training toxicity loss: {average_train_toxicity_loss:.6f}')
            print(f'   Training morgan loss: {average_train_morgan_loss:.6f}')
            print(f'   Validation loss: {average_val_loss:.6f}')
            print(f'   Validation embedding loss: {average_val_embedding_loss:.6f}')
            print(f'   Validation toxicity loss: {average_val_toxicity_loss:.6f}')
            print(f'   Validation morgan loss: {average_val_morgan_loss:.6f}')
    wandb.finish()
    return model, train_losses, val_losses, train_embedding_losses, train_toxicity_losses, train_morgan_losses, val_embedding_losses, val_toxicity_losses, val_morgan_losses
#%%

### Additional condition: collision energy
def train_model_condenc_123e1e2(model, train_data, val_data, epochs, learning_rate, criterion1, criterion2, criterion3, 
                                           lambda1, lambda2, lambda3, device, config = chemnet_tox_morgan_config):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    wandb.init(entity=config['wandb_entity'],
               project=config['wandb_project'],
               config=config) 
               
    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    
    # Store individual loss components (after lambda weighting)
    train_embedding_losses = []
    train_toxicity_losses = []
    train_morgan_losses = []
    val_embedding_losses = []
    val_toxicity_losses = []
    val_morgan_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_embedding_loss = 0.0
        running_toxicity_loss = 0.0
        running_morgan_loss = 0.0
        
        # Modified: batch now includes group and collision energy information as part of input
        for batch_with_ext, true_embeddings, true_log_tox, true_morgan, _ in train_data:
            batch_with_ext = batch_with_ext.to(device)  # Input includes spectra + group encoding + collision energy encoding
            true_embeddings = true_embeddings.to(device)
            true_log_tox = true_log_tox.to(device)
            true_morgan = true_morgan.to(device)

            optimizer.zero_grad()
            batch_predicted_combined = model(batch_with_ext)  # Forward pass with external conditions info
            
            # Embedding Loss
            batch_predicted_embeddings = batch_predicted_combined[:, :512] # First 512 columns
            loss1 = criterion1(batch_predicted_embeddings, true_embeddings) # loss1 (embedding loss)
            # Response Loss
            batch_predicted_log_tox = batch_predicted_combined[:, 512:513] # 512th column
            loss2 = criterion2(batch_predicted_log_tox, true_log_tox) # loss2 (toxicity loss)
            # Morgan Loss
            batch_predicted_morgan = batch_predicted_combined[:, 513:] # Last 2048 columns
            loss3 = criterion3(batch_predicted_morgan, true_morgan) # loss3 (morgan loss)

            # Apply lambda weighting
            weighted_loss1 = lambda1 * loss1
            weighted_loss2 = lambda2 * loss2
            weighted_loss3 = lambda3 * loss3
            
            # Total loss with modular weights (external conditions are NOT included in loss)
            total_loss = weighted_loss1 + weighted_loss2 + weighted_loss3

            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            running_loss += total_loss.item()
            running_embedding_loss += weighted_loss1.item()
            running_toxicity_loss += weighted_loss2.item()
            running_morgan_loss += weighted_loss3.item()
            
        average_train_loss = running_loss / len(train_data)
        average_train_embedding_loss = running_embedding_loss / len(train_data)
        average_train_toxicity_loss = running_toxicity_loss / len(train_data)
        average_train_morgan_loss = running_morgan_loss / len(train_data)
        wandb.log({"average_train_loss": average_train_loss})
        wandb.log({"average_train_embedding_loss": average_train_embedding_loss})
        wandb.log({"average_train_toxicity_loss": average_train_toxicity_loss})
        wandb.log({"average_train_morgan_loss": average_train_morgan_loss})

        model.eval()
        val_loss = 0.0
        val_embedding_loss = 0.0
        val_toxicity_loss = 0.0
        val_morgan_loss = 0.0
        
        with torch.no_grad():
            # Modified: validation batch also includes group and collision energy information
            for val_batch_with_ext, val_true_embeddings, val_true_tox, val_true_morgan, _ in val_data:
                val_batch_with_ext = val_batch_with_ext.to(device)  # Input includes spectra + group encoding + collision energy encoding
                val_true_embeddings = val_true_embeddings.to(device)
                val_true_tox = val_true_tox.to(device)
                val_true_morgan = val_true_morgan.to(device)

                val_batch_predicted = model(val_batch_with_ext)  # Forward pass with external conditions info
                val_batch_predicted_embeddings = val_batch_predicted[:, :512]
                val_batch_predicted_tox = val_batch_predicted[:, 512:513]
                val_batch_predicted_morgan = val_batch_predicted[:, 513:]

                # Calculate individual losses (external conditions are NOT included in loss calculation)
                val_loss1 = criterion1(val_batch_predicted_embeddings, val_true_embeddings)
                val_loss2 = criterion2(val_batch_predicted_tox, val_true_tox)
                val_loss3 = criterion3(val_batch_predicted_morgan, val_true_morgan)
                
                # Apply lambda weighting
                val_weighted_loss1 = lambda1 * val_loss1
                val_weighted_loss2 = lambda2 * val_loss2
                val_weighted_loss3 = lambda3 * val_loss3
                
                # Accumulate losses
                val_loss += (val_weighted_loss1 + val_weighted_loss2 + val_weighted_loss3).item()
                val_embedding_loss += val_weighted_loss1.item()
                val_toxicity_loss += val_weighted_loss2.item()
                val_morgan_loss += val_weighted_loss3.item()
                
        average_val_loss = val_loss / len(val_data)
        average_val_embedding_loss = val_embedding_loss / len(val_data)
        average_val_toxicity_loss = val_toxicity_loss / len(val_data)
        average_val_morgan_loss = val_morgan_loss / len(val_data)
        wandb.log({"average_val_loss": average_val_loss})
        wandb.log({"average_val_embedding_loss": average_val_embedding_loss})
        wandb.log({"average_val_toxicity_loss": average_val_toxicity_loss})
        wandb.log({"average_val_morgan_loss": average_val_morgan_loss})

        # Store losses for this epoch
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)
        train_embedding_losses.append(average_train_embedding_loss)
        train_toxicity_losses.append(average_train_toxicity_loss)
        train_morgan_losses.append(average_train_morgan_loss)
        val_embedding_losses.append(average_val_embedding_loss)
        val_toxicity_losses.append(average_val_toxicity_loss)
        val_morgan_losses.append(average_val_morgan_loss)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'   Training loss: {average_train_loss:.6f}')
            print(f'   Training embedding loss: {average_train_embedding_loss:.6f}')
            print(f'   Training toxicity loss: {average_train_toxicity_loss:.6f}')
            print(f'   Training morgan loss: {average_train_morgan_loss:.6f}')
            print(f'   Validation loss: {average_val_loss:.6f}')
            print(f'   Validation embedding loss: {average_val_embedding_loss:.6f}')
            print(f'   Validation toxicity loss: {average_val_toxicity_loss:.6f}')
            print(f'   Validation morgan loss: {average_val_morgan_loss:.6f}')
    wandb.finish()
    return model, train_losses, val_losses, train_embedding_losses, train_toxicity_losses, train_morgan_losses, val_embedding_losses, val_toxicity_losses, val_morgan_losses



# Conditional encoder with filtered Morgan fingerprints
class Cond_Encoder_1234(nn.Module):
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
        output = self.encoder(x)
        
        # Split the output into four parts
        embedding_output = output[:, :512]    # ChemNet embeddings (no activation)
        toxicity_raw = output[:, 512:513]     # Raw toxicity output (1 column)
        morgan_output = output[:, 513:513+2048]  # Morgan fingerprints (2048 columns)
        filtered_morgan_output = output[:, 513+2048:]  # Filtered Morgan fingerprints (remaining columns)
        
        # Apply scaled sigmoid only to toxicity part
        toxicity_output = torch.sigmoid(toxicity_raw) * np.log(100000) 

        # Concatenate back together
        final_output = torch.cat([embedding_output, toxicity_output, morgan_output, filtered_morgan_output], dim=1)
        
        return final_output


### Training function with filtered Morgan fingerprints
def train_model_condenc_1234e1e2(model, train_data, val_data, epochs, learning_rate, criterion1, criterion2, criterion3, criterion4,
                                       lambda1, lambda2, lambda3, lambda4, device, config = chemnet_tox_morgan_config):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    wandb.init(entity=config['wandb_entity'],
               project=config['wandb_project'],
               config=config) 
               
    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    
    # Store individual loss components (after lambda weighting)
    train_embedding_losses = []
    train_toxicity_losses = []
    train_morgan_losses = []
    train_filtered_morgan_losses = []
    val_embedding_losses = []
    val_toxicity_losses = []
    val_morgan_losses = []
    val_filtered_morgan_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_embedding_loss = 0.0
        running_toxicity_loss = 0.0
        running_morgan_loss = 0.0
        running_filtered_morgan_loss = 0.0
        
        # Modified: batch now includes group and collision energy information as part of input
        for batch_with_ext, true_embeddings, true_log_tox, true_morgan, true_filtered_morgan, _ in train_data:
            batch_with_ext = batch_with_ext.to(device)  # Input includes spectra + group encoding + collision energy encoding
            true_embeddings = true_embeddings.to(device)
            true_log_tox = true_log_tox.to(device)
            true_morgan = true_morgan.to(device)
            true_filtered_morgan = true_filtered_morgan.to(device)

            optimizer.zero_grad()
            batch_predicted_combined = model(batch_with_ext)  # Forward pass with external conditions info
            
            # Embedding Loss
            batch_predicted_embeddings = batch_predicted_combined[:, :512] # First 512 columns
            loss1 = criterion1(batch_predicted_embeddings, true_embeddings) # loss1 (embedding loss)
            # Response Loss
            batch_predicted_log_tox = batch_predicted_combined[:, 512:513] # 512th column
            loss2 = criterion2(batch_predicted_log_tox, true_log_tox) # loss2 (toxicity loss)
            # Morgan Loss
            batch_predicted_morgan = batch_predicted_combined[:, 513:513+2048] # Next 2048 columns
            loss3 = criterion3(batch_predicted_morgan, true_morgan) # loss3 (morgan loss)
            # Filtered Morgan Loss
            batch_predicted_filtered_morgan = batch_predicted_combined[:, 513+2048:] # Remaining columns
            loss4 = criterion4(batch_predicted_filtered_morgan, true_filtered_morgan) # loss4 (filtered morgan loss)

            # Apply lambda weighting
            weighted_loss1 = lambda1 * loss1
            weighted_loss2 = lambda2 * loss2
            weighted_loss3 = lambda3 * loss3
            weighted_loss4 = lambda4 * loss4
            
            # Total loss with modular weights (external conditions are NOT included in loss)
            total_loss = weighted_loss1 + weighted_loss2 + weighted_loss3 + weighted_loss4

            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            running_loss += total_loss.item()
            running_embedding_loss += weighted_loss1.item()
            running_toxicity_loss += weighted_loss2.item()
            running_morgan_loss += weighted_loss3.item()
            running_filtered_morgan_loss += weighted_loss4.item()
            
        average_train_loss = running_loss / len(train_data)
        average_train_embedding_loss = running_embedding_loss / len(train_data)
        average_train_toxicity_loss = running_toxicity_loss / len(train_data)
        average_train_morgan_loss = running_morgan_loss / len(train_data)
        average_train_filtered_morgan_loss = running_filtered_morgan_loss / len(train_data)
        wandb.log({"average_train_loss": average_train_loss})
        wandb.log({"average_train_embedding_loss": average_train_embedding_loss})
        wandb.log({"average_train_toxicity_loss": average_train_toxicity_loss})
        wandb.log({"average_train_morgan_loss": average_train_morgan_loss})
        wandb.log({"average_train_filtered_morgan_loss": average_train_filtered_morgan_loss})

        model.eval()
        val_loss = 0.0
        val_embedding_loss = 0.0
        val_toxicity_loss = 0.0
        val_morgan_loss = 0.0
        val_filtered_morgan_loss = 0.0
        
        with torch.no_grad():
            # Modified: validation batch also includes group and collision energy information
            for val_batch_with_ext, val_true_embeddings, val_true_tox, val_true_morgan, val_true_filtered_morgan, _ in val_data:
                val_batch_with_ext = val_batch_with_ext.to(device)  # Input includes spectra + group encoding + collision energy encoding
                val_true_embeddings = val_true_embeddings.to(device)
                val_true_tox = val_true_tox.to(device)
                val_true_morgan = val_true_morgan.to(device)
                val_true_filtered_morgan = val_true_filtered_morgan.to(device)

                val_batch_predicted = model(val_batch_with_ext)  # Forward pass with external conditions info
                val_batch_predicted_embeddings = val_batch_predicted[:, :512]
                val_batch_predicted_tox = val_batch_predicted[:, 512:513]
                val_batch_predicted_morgan = val_batch_predicted[:, 513:513+2048]
                val_batch_predicted_filtered_morgan = val_batch_predicted[:, 513+2048:]

                # Calculate individual losses (external conditions are NOT included in loss calculation)
                val_loss1 = criterion1(val_batch_predicted_embeddings, val_true_embeddings)
                val_loss2 = criterion2(val_batch_predicted_tox, val_true_tox)
                val_loss3 = criterion3(val_batch_predicted_morgan, val_true_morgan)
                val_loss4 = criterion4(val_batch_predicted_filtered_morgan, val_true_filtered_morgan)
                
                # Apply lambda weighting
                val_weighted_loss1 = lambda1 * val_loss1
                val_weighted_loss2 = lambda2 * val_loss2
                val_weighted_loss3 = lambda3 * val_loss3
                val_weighted_loss4 = lambda4 * val_loss4
                
                # Accumulate losses
                val_loss += (val_weighted_loss1 + val_weighted_loss2 + val_weighted_loss3 + val_weighted_loss4).item()
                val_embedding_loss += val_weighted_loss1.item()
                val_toxicity_loss += val_weighted_loss2.item()
                val_morgan_loss += val_weighted_loss3.item()
                val_filtered_morgan_loss += val_weighted_loss4.item()
                
        average_val_loss = val_loss / len(val_data)
        average_val_embedding_loss = val_embedding_loss / len(val_data)
        average_val_toxicity_loss = val_toxicity_loss / len(val_data)
        average_val_morgan_loss = val_morgan_loss / len(val_data)
        average_val_filtered_morgan_loss = val_filtered_morgan_loss / len(val_data)
        wandb.log({"average_val_loss": average_val_loss})
        wandb.log({"average_val_embedding_loss": average_val_embedding_loss})
        wandb.log({"average_val_toxicity_loss": average_val_toxicity_loss})
        wandb.log({"average_val_morgan_loss": average_val_morgan_loss})
        wandb.log({"average_val_filtered_morgan_loss": average_val_filtered_morgan_loss})

        # Store losses for this epoch
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)
        train_embedding_losses.append(average_train_embedding_loss)
        train_toxicity_losses.append(average_train_toxicity_loss)
        train_morgan_losses.append(average_train_morgan_loss)
        train_filtered_morgan_losses.append(average_train_filtered_morgan_loss)
        val_embedding_losses.append(average_val_embedding_loss)
        val_toxicity_losses.append(average_val_toxicity_loss)
        val_morgan_losses.append(average_val_morgan_loss)
        val_filtered_morgan_losses.append(average_val_filtered_morgan_loss)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'   Training loss: {average_train_loss:.6f}')
            print(f'   Training embedding loss: {average_train_embedding_loss:.6f}')
            print(f'   Training toxicity loss: {average_train_toxicity_loss:.6f}')
            print(f'   Training morgan loss: {average_train_morgan_loss:.6f}')
            print(f'   Training filtered morgan loss: {average_train_filtered_morgan_loss:.6f}')
            print(f'   Validation loss: {average_val_loss:.6f}')
            print(f'   Validation embedding loss: {average_val_embedding_loss:.6f}')
            print(f'   Validation toxicity loss: {average_val_toxicity_loss:.6f}')
            print(f'   Validation morgan loss: {average_val_morgan_loss:.6f}')
            print(f'   Validation filtered morgan loss: {average_val_filtered_morgan_loss:.6f}')
    wandb.finish()
    return model, train_losses, val_losses, train_embedding_losses, train_toxicity_losses, train_morgan_losses, train_filtered_morgan_losses, val_embedding_losses, val_toxicity_losses, val_morgan_losses, val_filtered_morgan_losses


def train_model_condenc_1234e1e2_weightloss(model, train_data, val_data, epochs, learning_rate, criterion1, criterion3, criterion4,
                                 lambda1, lambda2, lambda3, lambda4, device, config=None,
                                 alpha1=2, alpha2=1.5, alpha3=1.0, alpha4=0.5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    wandb.init(entity=config['wandb_entity'],
               project=config['wandb_project'],
               config=config)

    # Initialize lists for storing losses
    train_losses = []
    val_losses = []
    train_embedding_losses, train_toxicity_losses, train_morgan_losses, train_filtered_morgan_losses = ([] for _ in range(4))
    val_embedding_losses, val_toxicity_losses, val_morgan_losses, val_filtered_morgan_losses = ([] for _ in range(4))

    # Log weighted loss parameters
    wandb.config.update({
        "alpha1": alpha1, "alpha2": alpha2, "alpha3": alpha3, "alpha4": alpha4
    })

    for epoch in range(epochs):
        # Training Mode
        model.train()
        running_loss = 0.0
        running_embedding_loss, running_toxicity_loss = 0.0, 0.0
        running_morgan_loss, running_filtered_morgan_loss = 0.0, 0.0

        for batch_with_ext, true_embeddings, true_log_tox, true_morgan, true_filtered_morgan, _ in train_data:
            batch_with_ext = batch_with_ext.to(device)  # Input includes spectra + group encoding + collision energy encoding
            true_embeddings = true_embeddings.to(device)
            true_log_tox = true_log_tox.to(device)
            true_morgan = true_morgan.to(device)
            true_filtered_morgan = true_filtered_morgan.to(device)

            optimizer.zero_grad()
            batch_predicted_combined = model(batch_with_ext)  # Forward pass with external conditions info

            # Embedding Loss
            batch_predicted_embeddings = batch_predicted_combined[:, :512]  # First 512 columns
            loss1 = criterion1(batch_predicted_embeddings, true_embeddings)  # loss1 (embedding loss)

            # Toxicity Loss (4-step weighted_loss)
            batch_predicted_log_tox = batch_predicted_combined[:, 512:513]  # 512th column
            loss2 = weighted_loss(batch_predicted_log_tox, true_log_tox, alpha1, alpha2, alpha3, alpha4)  # Use 4-step weighted loss

            # Morgan Loss
            batch_predicted_morgan = batch_predicted_combined[:, 513:513 + 2048]  # Next 2048 columns
            loss3 = criterion3(batch_predicted_morgan, true_morgan)  # loss3 (morgan loss)

            # Filtered Morgan Loss
            batch_predicted_filtered_morgan = batch_predicted_combined[:, 513 + 2048:]  # Remaining columns
            loss4 = criterion4(batch_predicted_filtered_morgan, true_filtered_morgan)  # loss4 (filtered morgan loss)

            # Apply lambda weighting
            weighted_loss1 = lambda1 * loss1
            weighted_loss2 = lambda2 * loss2
            weighted_loss3 = lambda3 * loss3
            weighted_loss4 = lambda4 * loss4

            # Total loss with modular weights
            total_loss = weighted_loss1 + weighted_loss2 + weighted_loss3 + weighted_loss4
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            running_loss += total_loss.item()
            running_embedding_loss += weighted_loss1.item()
            running_toxicity_loss += weighted_loss2.item()
            running_morgan_loss += weighted_loss3.item()
            running_filtered_morgan_loss += weighted_loss4.item()

        # Calculate average train losses
        average_train_loss = running_loss / len(train_data)
        average_train_embedding_loss = running_embedding_loss / len(train_data)
        average_train_toxicity_loss = running_toxicity_loss / len(train_data)
        average_train_morgan_loss = running_morgan_loss / len(train_data)
        average_train_filtered_morgan_loss = running_filtered_morgan_loss / len(train_data)

        wandb.log({
            "average_train_loss": average_train_loss,
            "average_train_embedding_loss": average_train_embedding_loss,
            "average_train_toxicity_loss": average_train_toxicity_loss,
            "average_train_morgan_loss": average_train_morgan_loss,
            "average_train_filtered_morgan_loss": average_train_filtered_morgan_loss
        })

        # Validation Mode
        model.eval()
        val_loss = 0.0
        val_embedding_loss, val_toxicity_loss = 0.0, 0.0
        val_morgan_loss, val_filtered_morgan_loss = 0.0, 0.0

        with torch.no_grad():
            for val_batch_with_ext, val_true_embeddings, val_true_tox, val_true_morgan, val_true_filtered_morgan, _ in val_data:
                val_batch_with_ext = val_batch_with_ext.to(device)
                val_true_embeddings = val_true_embeddings.to(device)
                val_true_tox = val_true_tox.to(device)
                val_true_morgan = val_true_morgan.to(device)
                val_true_filtered_morgan = val_true_filtered_morgan.to(device)

                val_batch_predicted = model(val_batch_with_ext)
                val_batch_predicted_embeddings = val_batch_predicted[:, :512]
                val_batch_predicted_tox = val_batch_predicted[:, 512:513]
                val_batch_predicted_morgan = val_batch_predicted[:, 513:513 + 2048]
                val_batch_predicted_filtered_morgan = val_batch_predicted[:, 513 + 2048:]

                # Calculate individual validation losses
                val_loss1 = criterion1(val_batch_predicted_embeddings, val_true_embeddings)
                val_loss2 = weighted_loss(val_batch_predicted_tox, val_true_tox, alpha1, alpha2, alpha3, alpha4)  # Weighted loss
                val_loss3 = criterion3(val_batch_predicted_morgan, val_true_morgan)
                val_loss4 = criterion4(val_batch_predicted_filtered_morgan, val_true_filtered_morgan)

                # Apply lambda weighting
                val_weighted_loss1 = lambda1 * val_loss1
                val_weighted_loss2 = lambda2 * val_loss2
                val_weighted_loss3 = lambda3 * val_loss3
                val_weighted_loss4 = lambda4 * val_loss4

                # Total weighted val loss
                val_loss += (val_weighted_loss1 + val_weighted_loss2 + val_weighted_loss3 + val_weighted_loss4).item()
                val_embedding_loss += val_weighted_loss1.item()
                val_toxicity_loss += val_weighted_loss2.item()
                val_morgan_loss += val_weighted_loss3.item()
                val_filtered_morgan_loss += val_weighted_loss4.item()

        # Calculate average validation losses
        average_val_loss = val_loss / len(val_data)
        wandb.log({
            "average_val_loss": average_val_loss
        })

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"Training loss: {average_train_loss:.6f}")
            print(f"Validation loss: {average_val_loss:.6f}")

    wandb.finish()
    return model, train_losses, val_losses, train_embedding_losses, train_toxicity_losses, train_morgan_losses, train_filtered_morgan_losses, val_embedding_losses, val_toxicity_losses, val_morgan_losses, val_filtered_morgan_losses

# The full model traiing function with the 2 step weighted loss function incorportated

def train_model_condenc_1234e1e2_weightloss2(model, train_data, val_data, epochs, learning_rate, criterion1, criterion3, criterion4,
                                 lambda1, lambda2, lambda3, lambda4, device, config=None, 
                                 weighted_loss_params=None):
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize wandb and log parameters
    wandb.init(entity=config['wandb_entity'],
               project=config['wandb_project'],
               config=config)
    wandb.config.update(weighted_loss_params)  # Log the weighted loss parameter "alpha"

    # Extract `alpha` for weighted loss
    alpha = weighted_loss_params['alpha']

    # Lists for storing training and validation losses for monitoring
    train_losses, val_losses = [], []
    train_embedding_losses, train_toxicity_losses, train_morgan_losses, train_filtered_morgan_losses = [], [], [], []
    val_embedding_losses, val_toxicity_losses, val_morgan_losses, val_filtered_morgan_losses = [], [], [], []

    for epoch in range(epochs):
        model.train()  # Set the model to training mode

        # Accumulate training losses
        running_loss = 0.0
        running_embedding_loss = 0.0
        running_toxicity_loss = 0.0
        running_morgan_loss = 0.0
        running_filtered_morgan_loss = 0.0

        for batch_with_ext, true_embeddings, true_log_tox, true_morgan, true_filtered_morgan, _ in train_data:
            # Move data to the specified device
            batch_with_ext = batch_with_ext.to(device)
            true_embeddings = true_embeddings.to(device)
            true_log_tox = true_log_tox.to(device)
            true_morgan = true_morgan.to(device)
            true_filtered_morgan = true_filtered_morgan.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            batch_predicted_combined = model(batch_with_ext)

            # Embedding Loss
            batch_predicted_embeddings = batch_predicted_combined[:, :512]
            loss1 = criterion1(batch_predicted_embeddings, true_embeddings)

            # Toxicity Loss (using the updated weighted_loss function)
            batch_predicted_log_tox = batch_predicted_combined[:, 512:513]
            loss2 = weighted_loss(batch_predicted_log_tox, true_log_tox, alpha)

            # Morgan Loss
            batch_predicted_morgan = batch_predicted_combined[:, 513:513 + 2048]
            loss3 = criterion3(batch_predicted_morgan, true_morgan)

            # Filtered Morgan Loss
            batch_predicted_filtered_morgan = batch_predicted_combined[:, 513 + 2048:]
            loss4 = criterion4(batch_predicted_filtered_morgan, true_filtered_morgan)

            # Apply Lambda Scaling
            weighted_loss1 = lambda1 * loss1
            weighted_loss2 = lambda2 * loss2
            weighted_loss3 = lambda3 * loss3
            weighted_loss4 = lambda4 * loss4

            # Total Loss
            total_loss = weighted_loss1 + weighted_loss2 + weighted_loss3 + weighted_loss4

            # Backpropagation and optimizer step
            total_loss.backward()
            optimizer.step()

            # Accumulate losses for this batch
            running_loss += total_loss.item()
            running_embedding_loss += weighted_loss1.item()
            running_toxicity_loss += weighted_loss2.item()
            running_morgan_loss += weighted_loss3.item()
            running_filtered_morgan_loss += weighted_loss4.item()

        # Store and log epoch-wise averaged training losses
        train_losses.append(running_loss / len(train_data))
        train_embedding_losses.append(running_embedding_loss / len(train_data))
        train_toxicity_losses.append(running_toxicity_loss / len(train_data))
        train_morgan_losses.append(running_morgan_loss / len(train_data))
        train_filtered_morgan_losses.append(running_filtered_morgan_loss / len(train_data))

        wandb.log({
            "epoch": epoch,
            "train_loss": train_losses[-1],
            "train_embedding_loss": train_embedding_losses[-1],
            "train_toxicity_loss": train_toxicity_losses[-1],
            "train_morgan_loss": train_morgan_losses[-1],
            "train_filtered_morgan_loss": train_filtered_morgan_losses[-1]
        })

        # Validation Phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_running_embedding_loss = 0.0
        val_running_toxicity_loss = 0.0
        val_running_morgan_loss = 0.0
        val_running_filtered_morgan_loss = 0.0

        with torch.no_grad():
            for val_batch_with_ext, val_true_embeddings, val_true_tox, val_true_morgan, val_true_filtered_morgan, _ in val_data:
                # Move validation data to the specified device
                val_batch_with_ext = val_batch_with_ext.to(device)
                val_true_embeddings = val_true_embeddings.to(device)
                val_true_tox = val_true_tox.to(device)
                val_true_morgan = val_true_morgan.to(device)
                val_true_filtered_morgan = val_true_filtered_morgan.to(device)

                # Forward pass
                val_batch_predicted = model(val_batch_with_ext)

                # Embedding Loss
                val_predicted_embeddings = val_batch_predicted[:, :512]
                val_loss1 = criterion1(val_predicted_embeddings, val_true_embeddings)

                # Toxicity Loss (using the updated weighted_loss function)
                val_predicted_tox = val_batch_predicted[:, 512:513]
                val_loss2 = weighted_loss(val_predicted_tox, val_true_tox, alpha)

                # Morgan Loss
                val_predicted_morgan = val_batch_predicted[:, 513:513 + 2048]
                val_loss3 = criterion3(val_predicted_morgan, val_true_morgan)

                # Filtered Morgan Loss
                val_predicted_filtered_morgan = val_batch_predicted[:, 513 + 2048:]
                val_loss4 = criterion4(val_predicted_filtered_morgan, val_true_filtered_morgan)

                # Apply Lambda Scaling
                val_weighted_loss1 = lambda1 * val_loss1
                val_weighted_loss2 = lambda2 * val_loss2
                val_weighted_loss3 = lambda3 * val_loss3
                val_weighted_loss4 = lambda4 * val_loss4

                # Total Validation Loss
                val_loss += (val_weighted_loss1 + val_weighted_loss2 + val_weighted_loss3 + val_weighted_loss4).item()
                val_running_embedding_loss += val_weighted_loss1.item()
                val_running_toxicity_loss += val_weighted_loss2.item()
                val_running_morgan_loss += val_weighted_loss3.item()
                val_running_filtered_morgan_loss += val_weighted_loss4.item()

        # Store and log validation losses
        val_losses.append(val_loss / len(val_data))
        val_embedding_losses.append(val_running_embedding_loss / len(val_data))
        val_toxicity_losses.append(val_running_toxicity_loss / len(val_data))
        val_morgan_losses.append(val_running_morgan_loss / len(val_data))
        val_filtered_morgan_losses.append(val_running_filtered_morgan_loss / len(val_data))

        wandb.log({
            "epoch": epoch,
            "val_loss": val_losses[-1],
            "val_embedding_loss": val_embedding_losses[-1],
            "val_toxicity_loss": val_toxicity_losses[-1],
            "val_morgan_loss": val_morgan_losses[-1],
            "val_filtered_morgan_loss": val_filtered_morgan_losses[-1]
        })

    # Finish wandb session
    wandb.finish()

    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_embedding_losses": train_embedding_losses,
        "val_embedding_losses": val_embedding_losses,
        "train_toxicity_losses": train_toxicity_losses,
        "val_toxicity_losses": val_toxicity_losses
    }
### =============================================== TENSORS CREATION FUNCTIONS =============================================== ###

# This is our default function, the one we use to prep the data for the encoder that takes us from spectra to ChemNet encodings 
def create_dataset_tensors_1(spectra_dataset, embedding_df, device, start_idx=None, stop_idx=None):
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

    # create tensors of spectra, true embeddings, and spectra indices
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

def create_dataset_tensors_12(spectra_dataset, embedding_df, device, start_idx=None, stop_idx=None):

    spectra = spectra_dataset.iloc[:,start_idx:stop_idx] # prev was [1,-3]

    # create tensors of spectra, true embeddings, true toxicity values, and chemical name encodings for train and val
    chem_labels = list(spectra_dataset['SMILES_spectra'])
    log_tox_tensor = torch.Tensor(spectra_dataset["log_response"].values).unsqueeze(1).to(device)
    embeddings_tensor = torch.Tensor([embedding_df.loc[embedding_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
    spectra_tensor = torch.Tensor(spectra.values).to(device)
    spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

    return embeddings_tensor, log_tox_tensor, spectra_tensor, spectra_indices_tensor 


def create_dataset_tensors_condenc_123e1(spectra_dataset, embedding_df, morgan_df, device, start_idx=None, stop_idx=None):
    """
    Create tensors for the full conditional encoder WITH group information.
    
    Parameters:
    ----------
    spectra_dataset : pd.DataFrame
        DataFrame containing spectral data and chemical labels with Group column
    embedding_df : pd.DataFrame
        DataFrame containing ChemNet embeddings for chemicals
    morgan_df : pd.DataFrame
        DataFrame containing Morgan fingerprints for chemicals
    device : torch.device
        The device (CPU or GPU) on which to store the tensors
    start_idx : int, optional
        Start index for spectral columns
    stop_idx : int, optional
        Stop index for spectral columns
    
    Returns:
    -------
    tuple
        A tuple containing:
        - spectra_with_group_tensor (torch.Tensor): Tensor of spectral data concatenated with one-hot encoded group
        - embeddings_tensor (torch.Tensor): Tensor of true ChemNet embeddings
        - log_tox_tensor (torch.Tensor): Tensor of log toxicity values
        - morgan_tensor (torch.Tensor): Tensor of Morgan fingerprints
        - spectra_indices_tensor (torch.Tensor): Tensor of indices
    """
    # Extract spectral data
    spectra = spectra_dataset.iloc[:, start_idx:stop_idx]
    
    # One-hot encode the Group column
    group_encoded = pd.get_dummies(spectra_dataset['Group'], prefix='group', dtype=int)

    # Concatenate spectra with group encoding
    spectra_with_group = pd.concat([spectra, group_encoded], axis=1)

    # Create chemical labels list
    chem_labels = list(spectra_dataset['SMILES_spectra'])
    
    # Create tensors
    spectra_with_group_tensor = torch.Tensor(spectra_with_group.values).to(device)
    embeddings_tensor = torch.Tensor([embedding_df.loc[embedding_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
    log_tox_tensor = torch.Tensor(spectra_dataset["log_response"].values).unsqueeze(1).to(device)
    morgan_tensor = torch.Tensor([morgan_df.loc[morgan_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
    spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

    return spectra_with_group_tensor, embeddings_tensor, log_tox_tensor, morgan_tensor, spectra_indices_tensor



def create_dataset_tensors_condenc_123e1e2(spectra_dataset, embedding_df, morgan_df, device, start_idx=None, stop_idx=None):
    """
    Create tensors for the full conditional encoder WITH group and collision energy information.
    
    Parameters:
    ----------
    spectra_dataset : pd.DataFrame
        DataFrame containing spectral data and chemical labels with Group and CE_clean columns
    embedding_df : pd.DataFrame
        DataFrame containing ChemNet embeddings for chemicals
    morgan_df : pd.DataFrame
        DataFrame containing Morgan fingerprints for chemicals
    device : torch.device
        The device (CPU or GPU) on which to store the tensors
    start_idx : int, optional
        Start index for spectral columns
    stop_idx : int, optional
        Stop index for spectral columns
    
    Returns:
    -------
    tuple
        A tuple containing:
        - spectra_with_ext_tensor (torch.Tensor): Tensor of spectral data concatenated with one-hot encoded group and collision energy
        - embeddings_tensor (torch.Tensor): Tensor of true ChemNet embeddings
        - log_tox_tensor (torch.Tensor): Tensor of log toxicity values
        - morgan_tensor (torch.Tensor): Tensor of Morgan fingerprints
        - spectra_indices_tensor (torch.Tensor): Tensor of indices
    """
    # Extract spectral data
    spectra = spectra_dataset.iloc[:, start_idx:stop_idx]
    
    # One-hot encode the Group column
    group_encoded = pd.get_dummies(spectra_dataset['Group'], prefix='group', dtype=int)
    
    # One-hot encode the CE_clean column
    ce_encoded = pd.get_dummies(spectra_dataset['CE_clean'], prefix='ce', dtype=int)
    
    # Alternative: Numerical encoding for CE_clean (commented out)
    # ce_mapping = {'NAN': 0, 'low': 5, 'med': 10, 'high': 15}
    # ce_numerical = spectra_dataset['CE_clean'].map(ce_mapping).values.reshape(-1, 1)
    # ce_encoded = pd.DataFrame(ce_numerical, columns=['ce_numerical'], index=spectra_dataset.index)

    # Concatenate spectra with group and collision energy encoding
    spectra_with_ext = pd.concat([spectra, group_encoded, ce_encoded], axis=1)
   
    # Create chemical labels list
    chem_labels = list(spectra_dataset['SMILES_spectra'])
    
    # Create tensors
    spectra_with_ext_tensor = torch.Tensor(spectra_with_ext.values).to(device)
    embeddings_tensor = torch.Tensor([embedding_df.loc[embedding_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
    log_tox_tensor = torch.Tensor(spectra_dataset["log_response"].values).unsqueeze(1).to(device)
    morgan_tensor = torch.Tensor([morgan_df.loc[morgan_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
    spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

    return spectra_with_ext_tensor, embeddings_tensor, log_tox_tensor, morgan_tensor, spectra_indices_tensor


def create_dataset_tensors_123(spectra_dataset, embedding_df, morgan_df, device, start_idx=None, stop_idx=None):
    """
    Create tensors for the conditional encoder WITHOUT group information.
    
    Parameters:
    ----------
    spectra_dataset : pd.DataFrame
        DataFrame containing spectral data and chemical labels
    embedding_df : pd.DataFrame
        DataFrame containing ChemNet embeddings for chemicals
    morgan_df : pd.DataFrame
        DataFrame containing Morgan fingerprints for chemicals
    device : torch.device
        The device (CPU or GPU) on which to store the tensors
    start_idx : int, optional
        Start index for spectral columns
    stop_idx : int, optional
        Stop index for spectral columns
    
    Returns:
    -------
    tuple
        A tuple containing:
        - spectra_tensor (torch.Tensor): Tensor of spectral data only
        - embeddings_tensor (torch.Tensor): Tensor of true ChemNet embeddings
        - log_tox_tensor (torch.Tensor): Tensor of log toxicity values
        - morgan_tensor (torch.Tensor): Tensor of Morgan fingerprints
        - spectra_indices_tensor (torch.Tensor): Tensor of indices
    """
    # Extract spectral data (no group encoding)
    spectra = spectra_dataset.iloc[:, start_idx:stop_idx]
    
    # Create chemical labels list
    chem_labels = list(spectra_dataset['SMILES_spectra'])
    
    # Create tensors
    spectra_tensor = torch.Tensor(spectra.values).to(device)
    embeddings_tensor = torch.Tensor([embedding_df.loc[embedding_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
    log_tox_tensor = torch.Tensor(spectra_dataset["log_response"].values).unsqueeze(1).to(device)
    morgan_tensor = torch.Tensor([morgan_df.loc[morgan_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
    spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

    return spectra_tensor, embeddings_tensor, log_tox_tensor, morgan_tensor, spectra_indices_tensor



def create_dataset_tensors_condenc_1234e1e2(spectra_dataset, embedding_df, morgan_df, filtered_morgan_df, device, start_idx=None, stop_idx=None):
    """
    Create tensors for the full conditional encoder WITH group and collision energy information and filtered Morgan fingerprints.
    
    Parameters:
    ----------
    spectra_dataset : pd.DataFrame
        DataFrame containing spectral data and chemical labels with Group and CE_clean columns
    embedding_df : pd.DataFrame
        DataFrame containing ChemNet embeddings for chemicals
    morgan_df : pd.DataFrame
        DataFrame containing Morgan fingerprints for chemicals
    filtered_morgan_df : pd.DataFrame
        DataFrame containing filtered Morgan fingerprints for chemicals
    device : torch.device
        The device (CPU or GPU) on which to store the tensors
    start_idx : int, optional
        Start index for spectral columns
    stop_idx : int, optional
        Stop index for spectral columns
    
    Returns:
    -------
    tuple
        A tuple containing:
        - spectra_with_ext_tensor (torch.Tensor): Tensor of spectral data concatenated with one-hot encoded group and collision energy
        - embeddings_tensor (torch.Tensor): Tensor of true ChemNet embeddings
        - log_tox_tensor (torch.Tensor): Tensor of log toxicity values
        - morgan_tensor (torch.Tensor): Tensor of Morgan fingerprints
        - filtered_morgan_tensor (torch.Tensor): Tensor of filtered Morgan fingerprints
        - spectra_indices_tensor (torch.Tensor): Tensor of indices
    """
    # Extract spectral data
    spectra = spectra_dataset.iloc[:, start_idx:stop_idx]
    
    # One-hot encode the Group column
    group_encoded = pd.get_dummies(spectra_dataset['Group'], prefix='group', dtype=int)
    
    # One-hot encode the CE_clean column
    ce_encoded = pd.get_dummies(spectra_dataset['CE_clean'], prefix='ce', dtype=int)
    
    # Alternative: Numerical encoding for CE_clean (commented out)
    # ce_mapping = {'NAN': 0, 'low': 5, 'med': 10, 'high': 15}
    # ce_numerical = spectra_dataset['CE_clean'].map(ce_mapping).values.reshape(-1, 1)
    # ce_encoded = pd.DataFrame(ce_numerical, columns=['ce_numerical'], index=spectra_dataset.index)

    # Concatenate spectra with group and collision energy encoding
    spectra_with_ext = pd.concat([spectra, group_encoded, ce_encoded], axis=1)
   
    # Create chemical labels list
    chem_labels = list(spectra_dataset['SMILES_spectra'])
    
    # Create tensors
    spectra_with_ext_tensor = torch.Tensor(spectra_with_ext.values).to(device)
    embeddings_tensor = torch.Tensor([embedding_df.loc[embedding_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
    log_tox_tensor = torch.Tensor(spectra_dataset["log_response"].values).unsqueeze(1).to(device)
    morgan_tensor = torch.Tensor([morgan_df.loc[morgan_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
    filtered_morgan_tensor = torch.Tensor([filtered_morgan_df.loc[filtered_morgan_df['SMILES_spectra'] == chem_name].iloc[0, 1:].values.astype(float) for chem_name in chem_labels]).to(device)
    spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

    return spectra_with_ext_tensor, embeddings_tensor, log_tox_tensor, morgan_tensor, filtered_morgan_tensor, spectra_indices_tensor





### ======================================================= FUNCTIONS ======================================================= ###

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

def pre_filter_and_round_spectrum_mz(df, spectrum_col, min_mz=0, max_mz=1000, round_precision=0.001):
    """
    Pre-filter a DataFrame by removing m/z values outside the specified range AND round m/z values 
    to the nearest specified precision from spectrum strings.
    This reduces memory usage before processing with spectrum_string_to_dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with spectrum data
    spectrum_col : str
        Name of the column containing spectrum strings
    min_mz : float, optional
        Minimum m/z value to keep. Default is 0.
    max_mz : float, optional
        Maximum m/z value to keep. Default is 1000.
    round_precision : float, optional
        Precision to round m/z values to. Default is 0.001.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with filtered and rounded spectrum strings
    """
    def filter_and_round_spectrum_string(spectrum_str, min_mz, max_mz, round_precision):
        if pd.isna(spectrum_str):
            return ""
        
        pairs = spectrum_str.split()
        filtered_pairs = []
        
        for pair in pairs:
            try:
                x, y = pair.split(":")
                mz_value = float(x)
                
                # First filter by range
                if min_mz <= mz_value <= max_mz:
                    # Then round to specified precision
                    rounded_mz = round(mz_value / round_precision) * round_precision
                    # Format to avoid floating point precision issues
                    rounded_mz_str = f"{rounded_mz:.3f}"
                    filtered_pairs.append(f"{rounded_mz_str}:{y}")
            except:
                continue
        
        return " ".join(filtered_pairs)
    
    df_filtered = df.copy()
    print(f"Filtering {len(df)} rows, keeping m/z values between {min_mz} and {max_mz}, rounding to nearest {round_precision}...")
    df_filtered[spectrum_col] = df_filtered[spectrum_col].apply(
        lambda x: filter_and_round_spectrum_string(x, min_mz, max_mz, round_precision)
    )
    print("Filtering and rounding complete.")
    
    return df_filtered

# Filter my spectra into set ranges so that they can be combined after binning is completed. 
def pre_filter_spectrum_by_mz_range(df, spectrum_col, min_mz=0, max_mz=1000):
    """
    Pre-filter a DataFrame by removing m/z values outside the specified range from spectrum strings.
    This reduces memory usage before processing with spectrum_string_to_dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with spectrum data
    spectrum_col : str
        Name of the column containing spectrum strings
    min_mz : float, optional
        Minimum m/z value to keep. Default is 0.
    max_mz : float, optional
        Maximum m/z value to keep. Default is 1000.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with filtered spectrum strings
    """
    def filter_spectrum_string(spectrum_str, min_mz, max_mz):
        if pd.isna(spectrum_str):
            return ""
        
        pairs = spectrum_str.split()
        filtered_pairs = []
        
        for pair in pairs:
            try:
                x, y = pair.split(":")
                mz_value = float(x)
                if min_mz <= mz_value <= max_mz:
                    filtered_pairs.append(pair)
            except:
                continue
        
        return " ".join(filtered_pairs)
    
    df_filtered = df.copy()
    print(f"Filtering {len(df)} rows, keeping m/z values between {min_mz} and {max_mz}...")
    df_filtered[spectrum_col] = df_filtered[spectrum_col].apply(
        lambda x: filter_spectrum_string(x, min_mz, max_mz)
    )
    print("Filtering complete.")
    
    return df_filtered


# Old version without min_mz and max_mz filtering
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
# This is currently specifically for column names
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

# Morgan fingerprint extraction function
def expand_fingerprints_to_matrix(df, smiles_col='SMILES_spectra', fp_col='fp'):
    """
    Convert Morgan fingerprint strings into a matrix format.
    
    Parameters:
    df: DataFrame containing SMILES and fingerprint data
    smiles_col: name of the SMILES column
    fp_col: name of the fingerprint column
    
    Returns:
    DataFrame with SMILES as first column and fingerprint bits as subsequent columns
    """
    import ast
    
    # Remove rows with NA fingerprints (None in the current format)
    df_clean = df.dropna(subset=[fp_col]).copy()
    
    # Convert fingerprint strings to lists of integers
    fingerprints = []
    smiles_list = []
    
    for idx, row in df_clean.iterrows():
        try:
            # Convert string representation of list to actual list
            fp_list = ast.literal_eval(row[fp_col])
            fingerprints.append(fp_list)
            smiles_list.append(row[smiles_col])
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing fingerprint at index {idx}: {e}")
            continue
    
    # Convert to numpy array for easier handling
    fp_array = np.array(fingerprints)
    
    # Create column names for fingerprint bits
    n_bits = fp_array.shape[1]
    fp_columns = [f'bit_{i+1}' for i in range(n_bits)]
    
    # Create the result DataFrame
    result_df = pd.DataFrame(fp_array, columns=fp_columns)
    result_df.insert(0, smiles_col, smiles_list)  # Use smiles_col parameter instead of hardcoded 'SMILES'
    
    print(f"Created matrix with {len(result_df)} rows and {n_bits} fingerprint bits")
    print(f"Shape: {result_df.shape}")
    
    return result_df


# Threshold filter function
def apply_threshold_filter(df, threshold, startindx=1, stopindx=-1):
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
    
    # Get spectral columns (all except start and stop indexes)
    spectral_cols = filtered_df.columns[startindx:stopindx]
    
    # Ensure spectral data is numeric
    # filtered_df[spectral_cols] = filtered_df[spectral_cols].apply(pd.to_numeric, errors='coerce')
    
    # Apply threshold using numpy where - more explicit control
    spectral_data = filtered_df[spectral_cols].values
    spectral_data = np.where(spectral_data > threshold, spectral_data, 0)
    filtered_df[spectral_cols] = spectral_data
    
    # index_id column is preserved unchanged
    return filtered_df

# Uniform binning function
def bin_spectra_by_mz_range(df, bin_size, indx_id_indx, startindx=1, stopindx=-1):
    """
    Bins spectra data by grouping m/z columns into ranges of specified size.
    
    Parameters:
    df: DataFrame with first column as SMILES, last column as index_id, rest as m/z columns (float names)
    bin_size: Float, the size of each bin (e.g., 10 means bins of 0-10, 10-20, etc.)
    
    Returns:
    DataFrame with SMILES column, binned m/z columns named by bin midpoints, and index_id column
    """
    smiles_col = df.columns[0]
    index_col = df.columns[indx_id_indx]  # Preserve the last column (index_id)
    mz_cols = df.columns[startindx:stopindx]  # Exclude first and last columns
    
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
def fill_missing_bins(df, bin_size, indx_id_indx, startindx=1, stopindx=-1):
    """
    Fills in missing bin columns in a binned DataFrame.
    
    Parameters:
    df: DataFrame with first column as SMILES, last column as index_id, rest as binned m/z columns (float names)
    bin_size: Float, the original bin size used for binning
    
    Returns:
    DataFrame with all missing bin midpoints filled in with zeros
    """
    smiles_col = df.columns[0]
    index_col = df.columns[indx_id_indx]  # Preserve the last column (index_id)
    existing_bins = sorted([col for col in df.columns[startindx:stopindx] if isinstance(col, (int, float))])
    
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

def binning_loop(df_spectra, df_original, bin_sizes, thresholds, save_directory, indx_id_indx=-1, startindx=1, stopindx=-1):
    """
    Creates all binned and thresholded datasets for a complete grid search.
    
    Parameters:
    - df_spectra: DataFrame with spectral data (output from spectrum_string_to_dataframe)
    - df_original: Original DataFrame with response data (e.g., df4_QQpos)
    - bin_sizes: List of bin sizes to use
    - thresholds: List of threshold values to use
    - save_directory: Directory path to save the parquet files
    
    Returns:
    - Dictionary with all created datasets keyed by variable names
    """
    import warnings
    import gc  # For garbage collection
    
    created_datasets = {}
    
    def remove_duplicate_columns(df, preserve_cols=None):
        """
        Remove duplicate columns while preserving specified columns.
        
        Parameters:
        - df: DataFrame to process
        - preserve_cols: List of column names to always preserve (first occurrence)
        
        Returns:
        - DataFrame with duplicate columns removed
        """
        if preserve_cols is None:
            preserve_cols = []
        
        # Get columns to check for duplicates (exclude preserved columns)
        cols_to_check = [col for col in df.columns if col not in preserve_cols]
        preserve_mask = [col in preserve_cols for col in df.columns]
        
        # For numerical columns, round to avoid floating point precision issues
        rounded_cols = []
        for col in cols_to_check:
            if isinstance(col, (int, float)):
                rounded_cols.append(round(float(col), 6))  # Round to 6 decimal places
            else:
                rounded_cols.append(col)
        
        # Find duplicates in rounded columns
        seen = set()
        duplicate_mask = []
        
        col_idx = 0
        for i, col in enumerate(df.columns):
            if preserve_mask[i]:
                duplicate_mask.append(False)  # Never mark preserved columns as duplicates
            else:
                rounded_col = rounded_cols[col_idx]
                if rounded_col in seen:
                    duplicate_mask.append(True)  # Mark as duplicate
                else:
                    seen.add(rounded_col)
                    duplicate_mask.append(False)
                col_idx += 1
        
        # Keep only non-duplicate columns
        cols_to_keep = [col for i, col in enumerate(df.columns) if not duplicate_mask[i]]
        return df[cols_to_keep]
    
    # Create ALL binned and thresholded datasets (complete grid search)
    print("Creating all binned and thresholded datasets...")
    df_spectra_original = df_spectra.copy()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
            
        for bin_idx, bin_size in enumerate(bin_sizes):
            for thresh_idx, threshold in enumerate(thresholds):
                
                # Create variable name
                bin_str = str(bin_size).replace('.', '_')
                thresh_str = str(threshold).replace('.', '_')
                var_name = f"bin{bin_str}_thresh{thresh_str}_df_spectra"
                
                print(f"Creating {var_name} ({bin_idx+1}/{len(bin_sizes)}, {thresh_idx+1}/{len(thresholds)})...")
                
                try:
                    # Start with original data - create a fresh copy each iteration
                    current_data = df_spectra_original.copy()
                
                    # Apply threshold filtering first
                    threshold_filtered_data = apply_threshold_filter(current_data, threshold, startindx, stopindx)
                    
                    # Clear intermediate variables
                    del current_data
                    
                    # Then apply binning
                    binned_data = bin_spectra_by_mz_range(threshold_filtered_data, bin_size, indx_id_indx, startindx, stopindx)
                
                    # Clear intermediate variables
                    del threshold_filtered_data
                    
                    # Fill missing bins
                    filled_data = fill_missing_bins(binned_data, bin_size, indx_id_indx, startindx, stopindx)

                    # Clear intermediate variables
                    del binned_data

                    # Remove duplicate columns before adding response data
                    preserve_cols = [filled_data.columns[0], filled_data.columns[indx_id_indx]]  # SMILES and index_id
                    filled_data_clean = remove_duplicate_columns(filled_data, preserve_cols)
                    
                    # Clear intermediate variables
                    del filled_data
                    
                    # Add response and log response values
                    final_data = add_response_and_log_response(filled_data_clean, df_original)
                    
                    # Clear intermediate variables
                    del filled_data_clean
                    
                    # Ensure index_id is preserved from original data
                    if 'index_id' in df_spectra.columns:
                        final_data['index_id'] = df_spectra['index_id'].iloc[:len(final_data)].values
                    
                    # Final duplicate check before saving
                    final_duplicate_check = final_data.columns.duplicated().sum()
                    if final_duplicate_check > 0:
                        print(f"Warning: {final_duplicate_check} duplicates found in {var_name}, removing...")
                        final_data = final_data.loc[:, ~final_data.columns.duplicated()]
                    
                    # Store in created_datasets dictionary (only if needed for return value)
                    # Comment out if you don't need to return the datasets to save memory
                    # created_datasets[var_name] = final_data.copy()
                    
                    # Save to file
                    save_path = f"{save_directory}/{var_name}.parquet"
                    final_data.to_parquet(save_path, index=False)
                    print(f"Saved {var_name} to {save_path} - Shape: {final_data.shape}")
                    
                    # Clear final_data after saving
                    del final_data
                    
                    # Force garbage collection
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error creating {var_name}: {e}")
                    # Force garbage collection even on error
                    gc.collect()
                    continue

    print(f"  - {len(bin_sizes)} bin sizes: {bin_sizes}")
    print(f"  - {len(thresholds)} threshold values: {thresholds}")
    print(f"  - Plus the existing {len(bin_sizes)} thresh0 datasets")

    # Create the missing threshold 0 datasets
    print("Creating binned-only datasets (thresh0)...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for bin_idx, bin_size in enumerate(bin_sizes):
            # Create variable name for thresh0 (no threshold)
            bin_str = str(bin_size).replace('.', '_')
            var_name = f"bin{bin_str}_thresh_zero_df_spectra"
        
            print(f"Creating {var_name} ({bin_idx+1}/{len(bin_sizes)})...")
            
            try:
                # Start with original data (no threshold filtering) - create fresh copy
                current_data = df_spectra_original.copy()
            
                # Binning only
                binned_data = bin_spectra_by_mz_range(current_data, bin_size, indx_id_indx, startindx, stopindx)
            
                # Clear intermediate variables
                del current_data
                
                # Fill missing bins
                filled_data = fill_missing_bins(binned_data, bin_size, indx_id_indx, startindx, stopindx)

                # Clear intermediate variables
                del binned_data

                # Remove duplicate columns before adding response data
                preserve_cols = [filled_data.columns[0], filled_data.columns[indx_id_indx]]  # SMILES and index_id
                filled_data_clean = remove_duplicate_columns(filled_data, preserve_cols)
                
                # Clear intermediate variables
                del filled_data
                
                # Add response and log response values
                final_data = add_response_and_log_response(filled_data_clean, df_original)
                
                # Clear intermediate variables
                del filled_data_clean
                
                # Ensure index_id is preserved from original data
                if 'index_id' in df_spectra.columns:
                    final_data['index_id'] = df_spectra['index_id'].iloc[:len(final_data)].values
                
                # Final duplicate check before saving
                final_duplicate_check = final_data.columns.duplicated().sum()
                if final_duplicate_check > 0:
                    print(f"Warning: {final_duplicate_check} duplicates found in {var_name}, removing...")
                    final_data = final_data.loc[:, ~final_data.columns.duplicated()]
                
                # Store in created_datasets dictionary (only if needed)
                # Comment out if you don't need to return the datasets to save memory
                # created_datasets[var_name] = final_data.copy()
                
                # Save to file
                save_path = f"{save_directory}/{var_name}.parquet"
                final_data.to_parquet(save_path, index=False)
                print(f"Saved {var_name} to {save_path} - Shape: {final_data.shape}")
                
                # Clear final_data after saving
                del final_data
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                print(f"Error creating {var_name}: {e}")
                # Force garbage collection even on error
                gc.collect()
                continue

    print(f"Created {len(bin_sizes)} thresh0 datasets!")
    print(f"Total datasets created: {len(created_datasets)}")
    
    # Final cleanup
    del df_spectra_original
    gc.collect()
    
    return created_datasets

# def binning_loop(df_spectra, df_original, bin_sizes, thresholds, save_directory, indx_id_indx=-1, startindx=1, stopindx=-1):
#     """
#     Creates all binned and thresholded datasets for a complete grid search.
    
#     Parameters:
#     - df_spectra: DataFrame with spectral data (output from spectrum_string_to_dataframe)
#     - df_original: Original DataFrame with response data (e.g., df4_QQpos)
#     - bin_sizes: List of bin sizes to use
#     - thresholds: List of threshold values to use
#     - save_directory: Directory path to save the parquet files
    
#     Returns:
#     - Dictionary with all created datasets keyed by variable names
#     """
#     import warnings
    
#     created_datasets = {}
    
#     def remove_duplicate_columns(df, preserve_cols=None):
#         """
#         Remove duplicate columns while preserving specified columns.
        
#         Parameters:
#         - df: DataFrame to process
#         - preserve_cols: List of column names to always preserve (first occurrence)
        
#         Returns:
#         - DataFrame with duplicate columns removed
#         """
#         if preserve_cols is None:
#             preserve_cols = []
        
#         # Get columns to check for duplicates (exclude preserved columns)
#         cols_to_check = [col for col in df.columns if col not in preserve_cols]
#         preserve_mask = [col in preserve_cols for col in df.columns]
        
#         # For numerical columns, round to avoid floating point precision issues
#         rounded_cols = []
#         for col in cols_to_check:
#             if isinstance(col, (int, float)):
#                 rounded_cols.append(round(float(col), 6))  # Round to 6 decimal places
#             else:
#                 rounded_cols.append(col)
        
#         # Find duplicates in rounded columns
#         seen = set()
#         duplicate_mask = []
        
#         col_idx = 0
#         for i, col in enumerate(df.columns):
#             if preserve_mask[i]:
#                 duplicate_mask.append(False)  # Never mark preserved columns as duplicates
#             else:
#                 rounded_col = rounded_cols[col_idx]
#                 if rounded_col in seen:
#                     duplicate_mask.append(True)  # Mark as duplicate
#                 else:
#                     seen.add(rounded_col)
#                     duplicate_mask.append(False)
#                 col_idx += 1
        
#         # Keep only non-duplicate columns
#         cols_to_keep = [col for i, col in enumerate(df.columns) if not duplicate_mask[i]]
#         return df[cols_to_keep]
    
#     # Create ALL binned and thresholded datasets (complete grid search)
#     print("Creating all binned and thresholded datasets...")
#     df_spectra_original = df_spectra.copy()
    
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
            
#         for bin_size in bin_sizes:
#             for threshold in thresholds:
                
#                 # Create variable name
#                 bin_str = str(bin_size).replace('.', '_')
#                 thresh_str = str(threshold).replace('.', '_')
#                 var_name = f"bin{bin_str}_thresh{thresh_str}_df_spectra"
                
#                 print(f"Creating {var_name}...")
                
#                 try:
#                     # Start with original data
#                     current_data = df_spectra_original.copy()
                
#                     # Apply threshold filtering first
#                     threshold_filtered_data = apply_threshold_filter(current_data, threshold, startindx, stopindx)
                    
#                     # Then apply binning
#                     binned_data = bin_spectra_by_mz_range(threshold_filtered_data, bin_size, indx_id_indx, startindx, stopindx)
                
#                     # Fill missing bins
#                     filled_data = fill_missing_bins(binned_data, bin_size, indx_id_indx, startindx, stopindx)

#                     # Remove duplicate columns before adding response data
#                     preserve_cols = [filled_data.columns[0], filled_data.columns[indx_id_indx]]  # SMILES and index_id
#                     filled_data_clean = remove_duplicate_columns(filled_data, preserve_cols)
                    
#                     # Add response and log response values
#                     final_data = add_response_and_log_response(filled_data_clean, df_original)
                    
#                     # Ensure index_id is preserved from original data
#                     if 'index_id' in df_spectra.columns:
#                         final_data['index_id'] = df_spectra['index_id'].iloc[:len(final_data)].values
                    
#                     # Final duplicate check before saving
#                     final_duplicate_check = final_data.columns.duplicated().sum()
#                     if final_duplicate_check > 0:
#                         print(f"Warning: {final_duplicate_check} duplicates found in {var_name}, removing...")
#                         final_data = final_data.loc[:, ~final_data.columns.duplicated()]
                    
#                     # Store in created_datasets dictionary
#                     created_datasets[var_name] = final_data
                    
#                     # Save to file
#                     save_path = f"{save_directory}/{var_name}.parquet"
#                     final_data.to_parquet(save_path, index=False)
#                     print(f"Saved {var_name} to {save_path} - Shape: {final_data.shape}")
                    
#                 except Exception as e:
#                     print(f"Error creating {var_name}: {e}")
#                     continue

#     print(f"  - {len(bin_sizes)} bin sizes: {bin_sizes}")
#     print(f"  - {len(thresholds)} threshold values: {thresholds}")
#     print(f"  - Plus the existing {len(bin_sizes)} thresh0 datasets")

#     # Create the missing threshold 0 datasets
#     print("Creating binned-only datasets (thresh0)...")
    
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
        
#         for bin_size in bin_sizes:
#             # Create variable name for thresh0 (no threshold)
#             bin_str = str(bin_size).replace('.', '_')
#             var_name = f"bin{bin_str}_thresh_zero_df_spectra"
        
#             print(f"Creating {var_name}...")
            
#             try:
#                 # Start with original data (no threshold filtering)
#                 current_data = df_spectra_original.copy()
            
#                 # Binning only
#                 binned_data = bin_spectra_by_mz_range(current_data, bin_size, indx_id_indx, startindx, stopindx)
            
#                 # Fill missing bins
#                 filled_data = fill_missing_bins(binned_data, bin_size, indx_id_indx, startindx, stopindx)

#                 # Remove duplicate columns before adding response data
#                 preserve_cols = [filled_data.columns[0], filled_data.columns[indx_id_indx]]  # SMILES and index_id
#                 filled_data_clean = remove_duplicate_columns(filled_data, preserve_cols)
                
#                 # Add response and log response values
#                 final_data = add_response_and_log_response(filled_data_clean, df_original)
                
#                 # Ensure index_id is preserved from original data
#                 if 'index_id' in df_spectra.columns:
#                     final_data['index_id'] = df_spectra['index_id'].iloc[:len(final_data)].values
                
#                 # Final duplicate check before saving
#                 final_duplicate_check = final_data.columns.duplicated().sum()
#                 if final_duplicate_check > 0:
#                     print(f"Warning: {final_duplicate_check} duplicates found in {var_name}, removing...")
#                     final_data = final_data.loc[:, ~final_data.columns.duplicated()]
                
#                 # Store in created_datasets dictionary
#                 created_datasets[var_name] = final_data
                
#                 # Save to file
#                 save_path = f"{save_directory}/{var_name}.parquet"
#                 final_data.to_parquet(save_path, index=False)
#                 print(f"Saved {var_name} to {save_path} - Shape: {final_data.shape}")
                
#             except Exception as e:
#                 print(f"Error creating {var_name}: {e}")
#                 continue

#     print(f"Created {len(bin_sizes)} thresh0 datasets!")
#     print(f"Total datasets created: {len(created_datasets)}")
    
#     return created_datasets

# # Old possible broken binning loop
# def binning_loop(df_spectra, df_original, bin_sizes, thresholds, save_directory, indx_id_indx=-1, startindx=1, stopindx=-1):
#     """
#     Creates all binned and thresholded datasets for a complete grid search.
    
#     Parameters:
#     - df_spectra: DataFrame with spectral data (output from spectrum_string_to_dataframe)
#     - df_original: Original DataFrame with response data (e.g., df4_QQpos)
#     - bin_sizes: List of bin sizes to use
#     - thresholds: List of threshold values to use
#     - save_directory: Directory path to save the parquet files
    
#     Returns:
#     - Dictionary with all created datasets keyed by variable names
#     """
#     import warnings
    
#     created_datasets = {}
    
#     # Create ALL binned and thresholded datasets (complete grid search)
#     print("Creating all binned and thresholded datasets...")
#     df_spectra_original = df_spectra.copy()
    
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
            
#         for bin_size in bin_sizes:
#             for threshold in thresholds:
                
#                 # Create variable name
#                 bin_str = str(bin_size).replace('.', '_')
#                 thresh_str = str(threshold).replace('.', '_')
#                 var_name = f"bin{bin_str}_thresh{thresh_str}_df_spectra"
                    
#                 # Start with original data
#                 current_data = df_spectra_original.copy()
            
#                 # Apply threshold filtering first
#                 threshold_filtered_data = apply_threshold_filter(current_data, threshold, startindx, stopindx)
                
#                 # Then apply binning
#                 binned_data = bin_spectra_by_mz_range(threshold_filtered_data, bin_size, indx_id_indx, startindx, stopindx)
            
#                 # Fill missing bins
#                 filled_data = fill_missing_bins(binned_data, bin_size, indx_id_indx, startindx, stopindx)

#                 # Add response and log response values
#                 final_data = add_response_and_log_response(filled_data, df_original)
                
#                 # Ensure index_id is preserved from original data
#                 if 'index_id' in df_spectra.columns:
#                     final_data['index_id'] = df_spectra['index_id'].iloc[:len(final_data)].values
                
#                 # Store in created_datasets dictionary
#                 created_datasets[var_name] = final_data
                
#                 # Save to file
#                 save_path = f"{save_directory}/{var_name}.parquet"
#                 final_data.to_parquet(save_path)
#                 print(f"Saved {var_name} to {save_path} - Shape: {final_data.shape}")

#     print(f"  - {len(bin_sizes)} bin sizes: {bin_sizes}")
#     print(f"  - {len(thresholds)} threshold values: {thresholds}")
#     print(f"  - Plus the existing {len(bin_sizes)} thresh0 datasets")

#     # Create the missing threshold 0 datasets
#     print("Creating binned-only datasets (thresh0)...")
    
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
        
#         for bin_size in bin_sizes:
#             # Create variable name for thresh0 (no threshold)
#             bin_str = str(bin_size).replace('.', '_')
#             var_name = f"bin{bin_str}_thresh_zero_df_spectra"
        
#             print(f"Creating {var_name}...")
        
#             # Start with original data (no threshold filtering)
#             current_data = df_spectra_original.copy()
        
#             # Binning only
#             binned_data = bin_spectra_by_mz_range(current_data, bin_size, indx_id_indx, startindx, stopindx)
        
#             # Fill missing bins
#             filled_data = fill_missing_bins(binned_data, bin_size, indx_id_indx, startindx, stopindx)

#             # Add response and log response values
#             final_data = add_response_and_log_response(filled_data, df_original)
            
#             # Ensure index_id is preserved from original data
#             if 'index_id' in df_spectra.columns:
#                 final_data['index_id'] = df_spectra['index_id'].iloc[:len(final_data)].values
            
#             # Store in created_datasets dictionary
#             created_datasets[var_name] = final_data
            
#             # Save to file
#             save_path = f"{save_directory}/{var_name}.parquet"
#             final_data.to_parquet(save_path)
#             print(f"Saved {var_name} to {save_path}")

#     print(f"Created {len(bin_sizes)} thresh0 datasets!")
#     print(f"Total datasets created: {len(created_datasets)}")
    
#     return created_datasets




### ==================================================== Parsing Datasets ===================================================== ###

def parse_dataset_name(dataset_name, data_suffix='_df_spectra'):
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
        
        thresh_part = parts[1].split(data_suffix)[0]
        threshold = float(thresh_part.replace('_', '.'))
    
    return bin_size, threshold