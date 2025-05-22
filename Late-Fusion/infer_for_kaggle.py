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

    base_dir = '/kaggle/input/cassava-leaf-disease-classification'
    train_csv = os.path.join(base_dir, 'train.csv')
    img_dir = os.path.join(base_dir, 'train_images')
    label_map = os.path.join(base_dir, 'label_num_to_disease_map.json')
    
    # Transforms vit net 
    transform_val_vit = transforms.Compose([
        transforms.Resize((480, 360)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test_vit = transform_val_vit



    # Transforms efficient net 
    #weights = EfficientNet_B7_Weights.IMAGENET1K_V1
    #default_preprocess = weights.transforms()

    transform_val_effi = transforms.Compose([
        transforms.Resize(720),
        transforms.CenterCrop(640),
        transforms.ToTensor(),
        #default_preprocess
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
    ])
    transform_test_effi = transform_val_effi


    # Prepare Dataset and Split
    full_dataset = CassavaDataset(csv_file=train_csv,
                                  img_dir=img_dir,
                                  label_map_file=label_map)

    # Test dataset
    test_csv = os.path.join(base_dir, 'sample_submission.csv')
    test_dir = os.path.join(base_dir, 'test_images')


    test_dataset_vit = TestDataset(submission_csv=test_csv,
                               img_dir=test_dir,
                               transform=transform_test_vit)
    test_dataset_effi = TestDataset(submission_csv=test_csv,
                               img_dir=test_dir,
                               transform=transform_test_effi)


    # DataLoader
    test_loader_vit = DataLoader(test_dataset_vit, batch_size=batch_size, shuffle=False)
    test_loader_effi = DataLoader(test_dataset_effi, batch_size=batch_size, shuffle=False)


    # vit Model
    model_vit = vit_b_16(pretrained=None).to(device)
    for param in model_vit.parameters(): param.requires_grad = False
    for param in model_vit.heads.parameters(): param.requires_grad = True

    # Replace the classification head
    in_features = model_vit.heads.head.in_features

    # Replace the classification head
    model_vit.heads.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 5)  # Replace 5 with your actual number of classes
    ).to(device)

    # efficient net model 
    model_effi = efficientnet_b7(weights=None).to(device)
    for p in model_effi.parameters():
        p.requires_grad = False
    for p in model_effi.features[6].parameters(): 
        p.requires_grad = True
    for p in model_effi.features[5].parameters():
        p.requires_grad = True

    in_features = model_effi.classifier[1].in_features
    model_effi.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),  
        nn.Dropout(0.4),
        nn.Linear(512, 5)  # 5 classes
    ).to(device)


    # After training, reload best model.
    model_vit.load_state_dict(torch.load('/kaggle/input/vit/pytorch/default/1/ViT_model.pth'))
    model_effi.load_state_dict(torch.load('/kaggle/input/efficientnet/pytorch/default/1/EfficientNet_Model.pth'))


    model_vit.eval()
    model_effi.eval()

    idx_to_label = full_dataset.idx_to_label

    # Testing & Submission
    submission = []
    with torch.no_grad():
        for (images_vit, img_ids_vit), (images_effi, img_ids_effi) in zip(test_loader_vit, test_loader_effi):
            assert img_ids_vit == img_ids_effi, "Mismatch in image IDs"

            images_vit = images_vit.to(device)
            images_effi = images_effi.to(device)

            # Predict softmax probabilities
            outputs_vit = torch.softmax(model_vit(images_vit), dim=1)
            outputs_effi = torch.softmax(model_effi(images_effi), dim=1)

            #print(outputs_effi)
            #print("""""")
            #print(outputs_vit)

            # Average (Late Fusion)
            fused_outputs = (outputs_vit + outputs_effi) / 2

            _, preds = fused_outputs.max(1)

            for img_id, pred in zip(img_ids_vit, preds.cpu().numpy()):
                label = idx_to_label[int(pred)]
                submission.append((img_id, label))


    # Write submission.csv
    with open('submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'label'])
        writer.writerows(submission)

    print("Done. Submission saved to submission.csv")

if __name__ == '__main__':
    main()
