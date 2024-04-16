import sys
import os

import numpy as np
from torch.utils.data import DataLoader

import torch.optim as optim
import torch
import torch.nn as nn

from tqdm import tqdm

from lib.CustomDatasets import SCAEDataset
from lib.Models import StackedConvAutoencoder

# Hyperparameters
lr = 0.001
num_epochs = 100  # Number of epochs to train for

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_scae_model = '../models/SCAE_best.pth'

# Dataset and dataloader 
train_dataset = SCAEDataset(root_dir="data/abp/train")
val_dataset = SCAEDataset(root_dir="data/abp/val")
test_dataset = SCAEDataset(root_dir="data/abp/test")

print(f"Training Dataset size: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4,shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4,shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4,shuffle=True, pin_memory=True)


# Model and training definition
model = StackedConvAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
model.to(device)

best_val_loss = float('inf')

# Training loop
for epoch in tqdm(range(num_epochs)):
    model.train()  # Set the model to training mode
    train_loss = 0.0
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(device).float()
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_loss = 0.0
        for data in val_loader:
            inputs, _ = data
            inputs = inputs.to(device).float()
            
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')
    
    # Check if the current validation loss is the best
    if val_loss < best_val_loss:
        print(f"Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...")
        best_val_loss = val_loss
        # Save model
        torch.save(model.state_dict(), best_scae_model)


