import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -----------------------------
# Custom Focal Loss Implementation
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # We use standard CE but disable reduction so we can apply the focal weight per sample
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss) # Get the probability of the true class
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# -----------------------------
# 1. Enhanced Data Setup & Augmentation
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='data/Training', transform=train_transform)
test_dataset = datasets.ImageFolder(root='data/Testing', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# -----------------------------
# 2. Optimized Model Setup (DenseNet121)
# -----------------------------
# Upgraded to DenseNet121 for better feature reuse in medical imaging
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

# DenseNet uses 'classifier' instead of 'fc'
num_ftrs = model.classifier.in_features

# Retain the dropout regularization
model.classifier = nn.Sequential(
    nn.Dropout(0.5), 
    nn.Linear(num_ftrs, len(class_names))
) 
model = model.to(device)

# Swap out CrossEntropyLoss for our custom FocalLoss
criterion = FocalLoss(gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# -----------------------------
# 3. Robust Training Loop
# -----------------------------
epochs = 15
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_loss = float('inf')

total_start_time = time.time()

for epoch in range(epochs):
    epoch_start_time = time.time()
    
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = correct_train / total_train
    
    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    epoch_val_loss = val_loss / len(test_dataset)
    epoch_val_acc = correct_val / total_val
    
    # Save history
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_acc'].append(epoch_val_acc)
    
    # --- Optimization Steps ---
    scheduler.step(epoch_val_loss)
    
    # Model Checkpointing
    save_msg = ""
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        # Updated filename for the new architecture
        torch.save(model.state_dict(), 'best_brain_tumor_densenet.pth')
        save_msg = " -> Model Improved! Saved."

    epoch_end_time = time.time()
    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | "
          f"Time: {(epoch_end_time - epoch_start_time):.1f}s{save_msg}")

print(f"\nTotal Training Time: {(time.time() - total_start_time):.2f} seconds\n")

# -----------------------------
# 4. Data Visualization Functions
# -----------------------------
def plot_training_curves(history):
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss (Focal)')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss (Focal)')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

def plot_confusion_matrix(model_path, test_loader, class_names):
    # Load the BEST model, ensuring it uses DenseNet architecture
    best_model = models.densenet121(weights=None) 
    best_model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(best_model.classifier.in_features, len(class_names)))
    best_model.load_state_dict(torch.load(model_path))
    best_model = best_model.to(device)
    best_model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = best_model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (DenseNet121 + Focal Loss)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

# Generate the plots using the best saved weights
print("Generating visualizations...")
plot_training_curves(history)
plot_confusion_matrix('best_brain_tumor_densenet.pth', test_loader, class_names)
print("Done! Plots have been saved as PNG files in your directory.")