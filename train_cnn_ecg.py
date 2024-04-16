import sys
import os
from torch.utils.data import DataLoader

import torch.optim as optim
import torch
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt

from lib.CustomDatasets import SCAEDataset
from lib.Models import StackedConvAutoencoder, ArtifactCNN
from lib.Utilities import *
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


best_cnn_model='models/CNN_best_model_ECG.pth'

# Hyperparameters
num_classes = 2
num_epochs = 100  # Number of training epochs
lr = 0.0001

# Define variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(6)
best_scae_model = 'models/SCAE_ECG_best.pth'

# Dataset and dataloader 
train_dataset = SCAEDataset(root_dir="data/ecg/train")
val_dataset = SCAEDataset(root_dir="data/ecg/val")
test_dataset = SCAEDataset(root_dir="data/ecg/test")

train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4,shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4,shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4,shuffle=True, pin_memory=True)

# Stacked Convolutional Autoencoder
SCAEModel = StackedConvAutoencoder()

#Load the best SCAE model while training the AE
SCAEModel.load_state_dict(torch.load(best_scae_model))
SCAEModel.to(device)
SCAEModel.eval()

# Define the CNN model
model = ArtifactCNN(num_classes=num_classes)
model.to(device)
# print(model)
train_loss_arr, val_loss_arr = [],[]

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

best_val_loss = float('inf')  # Track the best validation loss

# Training and Validation
for epoch in tqdm(range(num_epochs)):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        inputs_recon = SCAEModel(inputs)
        
        # Forward + backward + optimize
        outputs = model(inputs_recon)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Print average training loss per epoch
    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}')
    
    # Validation
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            inputs_recon = SCAEModel(inputs)

            outputs = model(inputs_recon)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}')
    
    train_loss_arr.append(avg_train_loss)
    val_loss_arr.append(avg_val_loss)
    
    # Save the model if validation loss has decreased
    if avg_val_loss < best_val_loss:
        print(f'Validation Loss Decreased({best_val_loss:.4f}--->{avg_val_loss:.4f}) \t Saving The Model')
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_cnn_model)


# Display loss landscape
# Plotting both training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss_arr, label='Training Loss')
plt.plot(val_loss_arr, label='Validation Loss')

# Labeling the graph
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Load the best performing model
model.load_state_dict(torch.load(best_cnn_model))
# Evaluate the CNN model on test dataset
actual, predicted = evaluate_model(SCAEModel, model, test_loader, device)

# Calculate Accuracy
accuracy = accuracy_score(actual, predicted)

# Calculate F1 Score: F1 Score is the weighted average of Precision and Recall.
f1 = f1_score(actual, predicted)

# Calculate G-Score: G-Score is the geometric mean of Precision and Sensitivity (Recall).
precision = precision_score(actual, predicted)
sensitivity = recall_score(actual, predicted)  # Also known as Recall
g_score = (precision * sensitivity)**0.5

# Calculate Net Prediction: (True Positives - False Positives) / Total Predictions
true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted))
false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted))
net_prediction = (true_positives - false_positives) / len(predicted)

# Calculate Specificity: True Negative Rate
true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted))
specificity = true_negatives / sum(a == 0 for a in actual)

print(f'Accuracy: {accuracy}\n F1 Score: {f1}\n G Score: {g_score}\n Sensitivity: {sensitivity}\n Specificity: {specificity}')