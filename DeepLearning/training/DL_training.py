import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# File paths
DATASET_FILE = "dataset/badminton_stroke_dataset.json"
MODEL_FILE = "DeepLearning/models/dl_optimize.pth"

# Load dataset
print("Loading dataset...")
with open(DATASET_FILE, 'r') as file:
    data = json.load(file)

# Extract IMU_data_sequences and labels
X = []
y = []

for sample in data['samples'][1:]:
    IMU_data_sequences = list(sample['IMU_data_sequences'].values())
    IMU_data_100 = np.array(IMU_data_sequences)[:,50:150]
    X.append(IMU_data_100)
    y.append(sample['label'])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)-1

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Convert to one-hot if not already (assuming y is class labels 0-7)
if y.ndim == 1:
    num_classes = 8
    y_onehot = np.zeros((len(y), num_classes))
    y_onehot[np.arange(len(y)), y] = 1
    y = y_onehot
else:
    num_classes = y.shape[1]

# Split to train and validation dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)

# Reshape for PyTorch: (N, H, W, C) -> (N, C, H, W)
X_train = X_train.reshape(X_train.shape[0], 1, 100, 6)
X_test = X_test.reshape(X_test.shape[0], 1, 100, 6)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the model
class StrokeClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(StrokeClassifier, self).__init__()
        
        # First Conv block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second Conv block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.dropout2 = nn.Dropout(0.1)
        
        # Calculate the flattened size after convolutions
        # Input: (1, 100, 6) -> Conv1: (16, 99, 5) -> Conv2: (32, 98, 4)
        self.flatten_size = 32 * 98 * 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x  # Don't apply softmax here for CrossEntropyLoss


# Initialize model
model = StrokeClassifier(num_classes=num_classes).to(device)

# Print model summary
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Early stopping parameters
patience = 10
best_loss = float('inf')
patience_counter = 0
best_model_state = None


# Training loop
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        optimizer.zero_grad()
        
        outputs = model(inputs)
        # Convert one-hot labels to class indices for CrossEntropyLoss
        labels_idx = torch.argmax(labels, dim=1)
        loss = criterion(outputs, labels_idx)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels_idx).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            labels_idx = torch.argmax(labels, dim=1)
            loss = criterion(outputs, labels_idx)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels_idx).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_idx.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


# Train the model
print("\nStarting training...")
for epoch in range(100):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion)
    
    print(f'Epoch [{epoch+1}/100], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Early stopping logic
    if train_loss < best_loss:
        best_loss = train_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            model.load_state_dict(best_model_state)
            break

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Get predictions on validation set
print("\nEvaluating on validation set...")
_, val_acc, y_pred_validation, y_test_classes = evaluate(model, test_loader, criterion)
cm_validation = confusion_matrix(y_test_classes, y_pred_validation)

# Get predictions on training set
print("Evaluating on training set...")
_, train_acc, y_pred_training, y_train_classes = evaluate(model, train_loader, criterion)
cm_training = confusion_matrix(y_train_classes, y_pred_training)

print("\n" + "="*50)
print("Confusion matrix of validation data")
print(cm_validation)
print(classification_report(y_test_classes, y_pred_validation))

print("\n" + "="*50)
print("Confusion matrix of training data")
print(cm_training)
print(classification_report(y_train_classes, y_pred_training))

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'num_classes': num_classes,
}, MODEL_FILE)

print("\nModel saved to 'dl_optimize.pth'")