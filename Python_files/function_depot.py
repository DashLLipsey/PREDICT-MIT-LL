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

### ======================================================= ENCODERS ======================================================= ###
#%%
# ChemNet Encoder
# batch_size = __
# epochs= __
# lr= 0.0001
# criterion = nn.MSELoss()
# output_size = 512
# num_layers = __

class ChemNet_Encoder(nn.Module):
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

# # New structure with sigmoid in the final layer, and just BCELoss
class Morgan_fp_Encoder(nn.Module):
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
        # probs = torch.sigmoid(output)  
        return output

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


# GENERAL TOXICITY MLP
class ToxMLP_Reg(nn.Module):
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

#%%
# ChemNet MLP
# batch_size = __
# epochs = __
# lr = 0.0001
# criterion = nn.MSELoss()
# output_size = 1
# num_layers = __

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

def train_model_MLP(model, train_data, val_data, epochs, learning_rate, criterion, device, config = chemnet_mlp_config):
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
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
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
class Cond_Encoder_chemnet_tox(nn.Module):
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

def train_model_condenc_chemnet_tox(model, train_data, val_data, epochs, learning_rate, criterion1, criterion2, 
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
# batch_size = __
# epochs = __
# lr = 0.0001
# criterion1 = nn.MSELoss()
# criterion2 = nn.MSELoss()
# criterion3 = nn.MSELoss()
# output_size = 513
# num_layers = __
lambda1 = 1
lambda2 = 5
lambda3 = 1

# Conditional Encoder architecture
class Cond_Encoder_chemnet_tox_morgan(nn.Module):
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

def train_model_condenc_chemnet_tox_morgan(model, train_data, val_data, epochs, learning_rate, criterion1, criterion2, criterion3, 
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
                threshold_filtered_data = apply_threshold_filter(current_data, threshold, startindx, stopindx)
                
                # Then apply binning
                binned_data = bin_spectra_by_mz_range(threshold_filtered_data, bin_size, indx_id_indx, startindx, stopindx)
            
                # Fill missing bins
                filled_data = fill_missing_bins(binned_data, bin_size, indx_id_indx, startindx, stopindx)

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



### =============================================== TENSORS CREATION FUNCTIONS =============================================== ###

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



def create_dataset_tensors_fixed(spectra_dataset, embedding_df, device, start_idx=None, stop_idx=None):
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

    start_idx : int, optional
        Start index for spectra columns
        
    stop_idx : int, optional
        Stop index for spectra columns

    Returns:
    -------
    tuple
        A tuple containing:
        - embeddings_tensor (torch.Tensor): Tensor of true embeddings for the chemicals.
        - spectra_tensor (torch.Tensor): Tensor of spectral data.
        - spectra_indices_tensor (torch.Tensor): Tensor of indices corresponding to the spectra.
    """
    # Filter spectra_dataset to only include SMILES that exist in embedding_df
    available_smiles = set(embedding_df['SMILES_spectra'].values)
    spectra_smiles = set(spectra_dataset['SMILES_spectra'].values)
    
    missing_smiles = spectra_smiles - available_smiles
    if missing_smiles:
        print(f"Warning: {len(missing_smiles)} SMILES from spectra dataset not found in embeddings")
        print(f"First few missing: {list(missing_smiles)[:5]}")
        
        # Filter to only include SMILES that have embeddings
        mask = spectra_dataset['SMILES_spectra'].isin(available_smiles)
        spectra_dataset_filtered = spectra_dataset[mask].copy()
        
        print(f"Filtered dataset from {len(spectra_dataset)} to {len(spectra_dataset_filtered)} samples")
    else:
        spectra_dataset_filtered = spectra_dataset
        print("All SMILES found in embeddings dataset")
    
    # Get spectra using filtered dataset
    spectra = spectra_dataset_filtered.iloc[:,start_idx:stop_idx]
    
    # Create tensors using filtered dataset
    chem_labels = list(spectra_dataset_filtered['SMILES_spectra'])
    
    # Create embeddings tensor - now all SMILES should exist in embedding_df
    embeddings_list = []
    for chem_name in chem_labels:
        embedding_row = embedding_df.loc[embedding_df['SMILES_spectra'] == chem_name]
        if len(embedding_row) > 0:
            embeddings_list.append(embedding_row.iloc[0, 1:].values.astype(float))
        else:
            raise ValueError(f"SMILES {chem_name} not found in embeddings after filtering")
    
    embeddings_tensor = torch.Tensor(embeddings_list).to(device)
    spectra_tensor = torch.Tensor(spectra.values).to(device)
    spectra_indices_tensor = torch.Tensor(spectra_dataset_filtered['index'].to_numpy()).to(device)

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

### =============== PLOTTING TRAINING CURVES FUNCTIONS ===============

import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, model_name="Model", 
                        figsize=(10, 6), save_path=None, show_plot=True,
                        title_fontsize=16, label_fontsize=14, legend_fontsize=12):
    """
    Plot training and validation loss curves.
    
    Parameters:
    -----------
    train_losses : list
        List of training losses for each epoch
    val_losses : list  
        List of validation losses for each epoch
    model_name : str, optional
        Name of the model for the plot title. Default is "Model"
    figsize : tuple, optional
        Figure size (width, height). Default is (10, 6)
    save_path : str, optional
        Path to save the plot. If None, plot is not saved. Default is None
    show_plot : bool, optional
        Whether to display the plot. Default is True
    title_fontsize : int, optional
        Font size for the title. Default is 16
    label_fontsize : int, optional
        Font size for axis labels. Default is 14
    legend_fontsize : int, optional
        Font size for the legend. Default is 12
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training and validation losses
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    # Customize the plot
    ax.set_xlabel('Epoch', fontsize=label_fontsize)
    ax.set_ylabel('Loss', fontsize=label_fontsize)
    ax.set_title(f'{model_name}: Training and Validation Loss', fontsize=title_fontsize, fontweight='bold')
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for final losses
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    
    # Add text box with final loss values
    textstr = f'Final Training Loss: {final_train_loss:.6f}\nFinal Validation Loss: {final_val_loss:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Improve layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig, ax

def plot_multiple_training_curves(loss_data_dict, figsize=(12, 8), save_path=None, 
                                 show_plot=True, title_fontsize=16, label_fontsize=14, 
                                 legend_fontsize=10):
    """
    Plot multiple training curves on the same plot for comparison.
    
    Parameters:
    -----------
    loss_data_dict : dict
        Dictionary with model names as keys and tuples of (train_losses, val_losses) as values
        Example: {'ChemNet Encoder': (train_losses1, val_losses1), 
                  'Morgan FP Encoder': (train_losses2, val_losses2)}
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 8)
    save_path : str, optional
        Path to save the plot. If None, plot is not saved. Default is None
    show_plot : bool, optional
        Whether to display the plot. Default is True
    title_fontsize : int, optional
        Font size for the title. Default is 16
    label_fontsize : int, optional
        Font size for axis labels. Default is 14
    legend_fontsize : int, optional
        Font size for the legend. Default is 10
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, (model_name, (train_losses, val_losses)) in enumerate(loss_data_dict.items()):
        color = colors[i % len(colors)]
        epochs = range(1, len(train_losses) + 1)
        
        # Plot with different line styles for train vs validation
        ax.plot(epochs, train_losses, color=color, linestyle='-', 
                label=f'{model_name} (Train)', linewidth=2, alpha=0.8)
        ax.plot(epochs, val_losses, color=color, linestyle='--', 
                label=f'{model_name} (Val)', linewidth=2, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Epoch', fontsize=label_fontsize)
    ax.set_ylabel('Loss', fontsize=label_fontsize)
    ax.set_title('Training and Validation Loss Comparison', fontsize=title_fontsize, fontweight='bold')
    ax.legend(fontsize=legend_fontsize, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig, ax

### ===================================================== ENCODERS W/O WANDB ==================================================== ###

# ChemNet Encoder
# batch_size = __
# epochs= __
# lr= 0.0001
# criterion = nn.MSELoss()
# output_size = 512
# num_layers = __

def train_model_chemnet_encoder_nowandb(model, train_data, val_data, epochs, learning_rate, criterion, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
 
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
        average_train_loss = running_loss / len(train_loader) 

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch, val_true_embeddings, _ in val_data:  
                val_batch = val_batch.to(device)
                val_true_embeddings = val_true_embeddings.to(device)

                val_batch_predicted_embeddings = model(val_batch)

                val_batch_loss = criterion(val_batch_predicted_embeddings, val_true_embeddings)  
                val_loss += val_batch_loss.item() 
        average_val_loss = val_loss / len(val_loader) 

        # Store losses for this epoch
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)

        if epoch % 50 == 0 or epoch == epochs - 1:  
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'   Training loss: {average_train_loss:.6f}')
            print(f'   Validation loss: {average_val_loss:.6f}')
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

def train_model_morgan_fp_encoder_nowandb(model, train_data, val_data, epochs, learning_rate, criterion, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
        # Store losses for this epoch
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)

        if epoch % 50 == 0 or epoch == epochs - 1:  
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'   Training loss: {average_train_loss:.6f}')
            print(f'   Validation loss: {average_val_loss:.6f}')
    return model, train_losses, val_losses

#%%
# Spectra Toxicity MLP
# batch_size = __
# epochs = __
# lr = 0.0001
# criterion = nn.MSELoss()
# output_size = 1
# num_layers = __

def train_model_MLP_spectra_nowandb(model, train_data, val_data, epochs, learning_rate, criterion, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

        # Store losses for this epoch
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)

        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'   Training loss: {average_train_loss:.6f}')
        print(f'   Validation loss: {average_val_loss:.6f}')
    return model, train_losses, val_losses

#%%
# ChemNet MLP
# batch_size = __
# epochs = __
# lr = 0.0001
# criterion = nn.MSELoss()
# output_size = 1
# num_layers = __

def train_model_MLP_nowandb(model, train_data, val_data, epochs, learning_rate, criterion, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

        # Store losses for this epoch
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'   Training loss: {average_train_loss:.6f}')
            print(f'   Validation loss: {average_val_loss:.6f}')
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

def train_model_condenc_chemnet_tox_nowandb(model, train_data, val_data, epochs, learning_rate, criterion1, criterion2, 
                                            lambda1, lambda2, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    return model, train_losses, val_losses, train_embedding_losses, train_toxicity_losses, val_embedding_losses, val_toxicity_losses


#%%
# Conditional encoder (ChemNet + Toxicity + Morgan Fingerprints) 
# batch_size = __
# epochs = __
# lr = 0.0001
# criterion1 = nn.MSELoss()
# criterion2 = nn.MSELoss()
criterion3 = nn.MSELoss()
# output_size = 513
# num_layers = __
lambda1 = 1
lambda2 = 5
lambda3 = 1

def train_model_condenc_chemnet_tox_morgan_nowandb(model, train_data, val_data, epochs, learning_rate, criterion1, criterion2, 
                                                   criterion3, lambda1, lambda2, lambda3, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

        # Store losses for this epoch
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)
        train_embedding_losses.append(average_train_embedding_loss)
        train_toxicity_losses.append(average_train_toxicity_loss)
        train_morgan_losses.append(average_train_morgan_loss)
        val_embedding_losses.append(average_val_embedding_loss)
        val_toxicity_losses.append(average_val_toxicity_loss)
        val_morgan_losses.append(average_val_morgan_loss)

        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'   Training loss: {average_train_loss:.6f}')
        print(f'   Training embedding loss: {average_train_embedding_loss:.6f}')
        print(f'   Training toxicity loss: {average_train_toxicity_loss:.6f}')
        print(f'   Training morgan loss: {average_train_morgan_loss:.6f}')
        print(f'   Validation loss: {average_val_loss:.6f}')
        print(f'   Validation embedding loss: {average_val_embedding_loss:.6f}')
        print(f'   Validation toxicity loss: {average_val_toxicity_loss:.6f}')
        print(f'   Validation morgan loss: {average_val_morgan_loss:.6f}')
    return model, train_losses, val_losses, train_embedding_losses, train_toxicity_losses, train_morgan_losses, val_embedding_losses, val_toxicity_losses, val_morgan_losses


### ========================================================= HEATMAPS ========================================================= ###

# Also create individual larger heatmaps for Morgan fingerprints for better detail
def create_detailed_heatmap_morgan(pivot_data, metric_name, cmap, figsize=(12, 8), vmin=None, vmax=None):
    """Create a detailed heatmap for a single Morgan fingerprint metric"""
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
    
    plt.title(f'Morgan Fingerprint: {metric_name} by Bin Size and Threshold', fontsize=16, fontweight='bold')
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
    plt.savefig(f"/home/dlipsey/MITLincolnLabs/Figures/Morgan_Fingerprint_{metric_name}_by_Bin_Size_and_Threshold")
    plt.show()

# Also create individual larger heatmaps for better detail
def create_detailed_heatmap_spec(pivot_data, metric_name, cmap, figsize=(12, 8), vmin=None, vmax=None):
    """Create a detailed heatmap for a single metric"""
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
    
    plt.title(f'Spectra: {metric_name} by Bin Size and Threshold', fontsize=16, fontweight='bold')
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
    plt.savefig(f"/home/dlipsey/MITLincolnLabs/Figures/Spectra_{metric_name}_by_Bin_Size_and_Threshold")
    plt.show()

def create_detailed_heatmap_chemnet(pivot_data, metric_name, cmap, figsize=(12, 8), vmin=None, vmax=None):
    """Create a detailed heatmap for a single ChemNet metric"""
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
    
    plt.title(f'ChemNet: {metric_name} by Bin Size and Threshold', fontsize=16, fontweight='bold')
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
    plt.savefig(f"/home/dlipsey/MITLincolnLabs/Figures/ChemNet_{metric_name}_by_Bin_Size_and_Threshold")
    plt.show()

def create_detailed_heatmap_cond_enc(pivot_data, metric_name, cmap, figsize=(12, 8), vmin=None, vmax=None):
    """Create a detailed heatmap for a single conditional encoder metric"""
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(pivot_data, 
                annot=True, 
                fmt='.1f', 
                cmap=cmap,
                square=False,
                linewidths=0.5,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': f'Test {metric_name}', 'shrink': 0.8})
    
    plt.title(f'Conditional Encoder: {metric_name} by Bin Size and Threshold', fontsize=16, fontweight='bold')
    plt.xlabel('Threshold Value', fontsize=14)
    plt.ylabel('Bin Size', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Improve readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Add text annotation for best performance
    best_val = pivot_data.min().min()
    plt.text(0.02, 0.98, f'Best {metric_name}: {best_val:.1f}%', 
            transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f"/home/dlipsey/MITLincolnLabs/Figures/Conditional_encoder_{metric_name}_by_Bin_Size_and_Threshold")
    plt.show()