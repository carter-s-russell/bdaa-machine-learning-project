# https://www.kaggle.com/code/asad36302/classify-brain-tumor-mri-cnn-with-merged-dataset#Copy-Dataset-2-Training
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='data/Training', transform=transform)
test_dataset = datasets.ImageFolder(root='data/Testing', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(train_dataset.class_to_idx)

