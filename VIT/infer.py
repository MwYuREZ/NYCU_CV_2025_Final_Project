import os
import csv
import json
import time

import torch
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import RandAugment
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from torchvision.models import resnext50_32x4d, vit_b_16, vit_b_32, vit_l_32, vit_h_14

from PIL import Image
from tqdm import tqdm

# -----------------------------
# Custom Dataset for Cassava
# -----------------------------
class CassavaDataset(Dataset):
    """
    Dataset for cassava leaf images with labels from CSV.
    """
    def __init__(self, csv_file, img_dir, label_map_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # Load label mapping if needed (e.g., to class names)
        with open(label_map_file, 'r') as f:
            self.num_to_name = json.load(f)

        # Read CSV
        self.samples = []  # list of (img_path, label_num)
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row['image_id']
                lbl = int(row['label'])
                path = os.path.join(self.img_dir, f"{img_id}")
                self.samples.append((path, lbl))

        # Build label <-> index mapping
        labels = sorted({lbl for _, lbl in self.samples})
        self.label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
        self.idx_to_label = {idx: lbl for lbl, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # convert numeric label to class index
        cls_idx = self.label_to_idx[lbl]
        return image, cls_idx
    

class TestDataset(Dataset):
    """
    Dataset for cassava test images following sample_submission order.
    """
    def __init__(self, submission_csv, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_ids = []
        # Read image IDs from sample_submission (first column 'image_id')
        with open(submission_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.image_ids.append(row['image_id'])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_filename = f"{img_id}"
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id  # return id without extension for submission

# -----------------------------
# Main Training & Testing Script
# -----------------------------

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparams
    batch_size = 1

    base_dir = '../cassava-leaf-disease-classification'
    train_csv = os.path.join(base_dir, 'train.csv')
    img_dir = os.path.join(base_dir, 'train_images')
    label_map = os.path.join(base_dir, 'label_num_to_disease_map.json')

    # Transforms
    transform_train = transforms.Compose([
        transforms.Resize((800, 600)),
        transforms.RandomHorizontalFlip(p=0.6),
        transforms.RandomAdjustSharpness(3, p=0.45),
        transforms.RandomResizedCrop((518, 518)),
        transforms.ColorJitter(brightness=0.25, contrast=0.22, saturation=0.2, hue=0.15),
        transforms.RandomRotation(35),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((800, 600)),
        transforms.CenterCrop((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transform_val

    # Prepare Dataset and Split
    full_dataset = CassavaDataset(csv_file=train_csv,
                                  img_dir=img_dir,
                                  label_map_file=label_map,
                                  transform=transform_train)


    # Test dataset
    test_csv = os.path.join(base_dir, 'sample_submission.csv')
    test_dir = os.path.join(base_dir, 'test_images')
    test_dataset = TestDataset(submission_csv=test_csv,
                               img_dir=test_dir,
                               transform=transform_test)

    # DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = vit_h_14(weights=None, image_size=518).to(device)
    for param in model.parameters(): param.requires_grad = False
    for param in model.heads.parameters(): param.requires_grad = True

    # Replace the classification head
    in_features = model.heads.head.in_features

    model.heads.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 5)
    ).to(device)


    # After training, reload best model.
    model.load_state_dict(torch.load('best_model_vit.pth'))
    model.eval()
    idx_to_label = full_dataset.idx_to_label

    # Testing & Submission
    submission = []
    with torch.no_grad():
        for images, img_ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            for img_id, pred in zip(img_ids, preds.cpu().numpy()):
                orig_label = idx_to_label[int(pred)]
                submission.append((img_id, orig_label))

    # Write submission.csv
    with open('submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'label'])
        writer.writerows(submission)

    print("Done. Submission saved to submission.csv")

if __name__ == '__main__':
    main()
