from segmentation_model import *
import os
import numpy as np
import cv2
import albumentations as a
import torch.optim as optim
import torch.nn.functional as f
import matplotlib.pyplot as plt
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from efficientnet_pytorch import EfficientNet
from imblearn.over_sampling import SMOTE


def augmentations():
    return a.Compose([
        a.HorizontalFlip(p=0.5),
        a.VerticalFlip(p=0.5),
        a.Rotate(limit=30, p=0.5),
        a.RandomBrightnessContrast(p=0.2),
        a.GaussNoise(p=0.5),
        a.RandomResizedCrop(size=(128, 128), scale=(0.8, 0.8), p=0.5),
        ToTensorV2(),
    ])


class DiceFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.8, gamma=2, class_weights=None):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        smooth = 1e-6
        y_pred = torch.sigmoid(y_pred)
        y_true = y_true.unsqueeze(1).float()

        intersection = (y_pred * y_true).sum(dim=(2, 3))
        union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
        dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)

        if self.class_weights is not None:
            dice_loss = dice_loss * self.class_weights[1]

        bce = f.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        focal_weight = (1 - torch.exp(-bce)) ** self.gamma
        focal_loss = focal_weight * bce

        if self.class_weights is not None:
            focal_loss = focal_loss * self.class_weights[y_true.long()]

        return self.alpha * dice_loss.mean() + (1 - self.alpha) * focal_loss.mean()


class MultiClassDataset(Dataset):
    def __init__(self, inputs, masks):
        self.inputs = inputs
        self.masks = masks

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image = self.inputs[idx]
        mask = self.masks[idx]

        image = image / 255.0

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask


def segmentation_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy().flatten()
    y_pred_prob = torch.sigmoid(y_pred).cpu().numpy().flatten()
    y_pred_bin = (y_pred_prob > 0.5).astype(np.uint8)

    accuracy = accuracy_score(y_true, y_pred_bin)

    intersection = np.logical_and(y_true, y_pred_bin).sum()
    union = np.logical_or(y_true, y_pred_bin).sum()
    iou = intersection / (union + 1e-6)

    dice = (2 * intersection) / (y_true.sum() + y_pred_bin.sum() + 1e-6)

    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = float('nan')

    return accuracy, iou, dice, auc


def seg_augmented_data(images, masks, num_augmentations=25):
    augmented_images = []
    augmented_masks = []
    augment_fn = augmentations()

    for image, mask in zip(images, masks):
        for _ in range(num_augmentations):
            augmented = augment_fn(image=image.astype(np.uint8), mask=mask.astype(np.uint8))
            augmented_images.append(augmented['image'])
            augmented_masks.append(augmented['mask'])

    return np.array(augmented_images), np.array(augmented_masks)


def calculate_class_weights(masks):
    class_counts = np.bincount(masks.flatten().astype(int))
    frequency = class_counts/np.sum(class_counts)
    class_weights = 1.0 / frequency
    class_weights = class_weights/np.sum(class_weights)
    return torch.tensor(class_weights, dtype=torch.float32)


def preprocess_data(image_dir, mask_dir, lesion_types, target_size=(128, 128)):
    images = []
    masks = []

    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size) / 255.0
        img = img.astype(np.float32)
        images.append(img)

        binary_mask = np.zeros(target_size, dtype=np.uint8)

        for lesion_type in lesion_types:
            mask_filename = filename.replace(".jpg", f"_{lesion_type}.tif")
            mask_path = os.path.join(mask_dir, mask_filename)

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, target_size)
                mask = (mask > 0).astype(np.uint8)
                binary_mask = np.maximum(binary_mask, mask)

        masks.append(binary_mask.astype(np.float32))

    return np.array(images), np.array(masks)


def segmentation_model(batch_size, num_epochs, image_dir='dataset/IDRiD/train/images',
                       masks_dir='dataset/IDRiD/train/masks'):

    lesion_types = {'MA': 1, 'HE': 2, 'EX': 3, 'SE': 4, 'OD': 5}

    images, masks = preprocess_data(image_dir, masks_dir, lesion_types)

    augmented_images, augmented_masks = seg_augmented_data(images, masks)

    images = images.transpose(0, 3, 1, 2)

    all_images = np.concatenate([images, augmented_images])
    all_masks = np.concatenate([masks, augmented_masks])

    class_weights = calculate_class_weights(all_masks)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        print(f"Fold {fold + 1}/{kf.n_splits}")

        train_inputs, val_inputs = all_images[train_idx], all_images[val_idx]
        train_masks, val_masks = all_masks[train_idx], all_masks[val_idx]

        train_dataset = MultiClassDataset(train_inputs, train_masks)
        val_dataset = MultiClassDataset(val_inputs, val_masks)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = UNetWithAttention(in_channels=3, out_channels=1)

        criterion = DiceFocalLoss(alpha=0.8, gamma=2, class_weights=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for image, mask in train_loader:

                outputs = model(image)
                loss = criterion(outputs, mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for image, mask in val_loader:
                    outputs = model(image)
                    loss = criterion(outputs, mask)
                    val_loss += loss.item()

                    all_preds.append(torch.sigmoid(outputs) > 0.5)
                    all_labels.append(mask)

                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)
                accuracy, iou, dice, auc = segmentation_metrics(all_labels, all_preds)

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, "
                  f"Val Loss: {val_loss / len(val_loader):.4f}, Accuracy: {accuracy:.4f}, "
                  f"IoU: {iou:.4f}, Dice: {dice:.4f}, AUC: {auc:.4f}")

        torch.save(model.state_dict(), f"model_fold_{fold + 1}.pth")


def predict_segmentation(lesion_mapping, image_dir='dataset/IDRiD/test/images', mask_dir='dataset/IDRiD/test/masks',
                         model_path='model_fold_1.pth'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetWithAttention(in_channels=3, out_channels=1)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    image_files = [x for x in os.listdir(image_dir) if x.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in the directory.")
        return

    image_name = random.choice(image_files)
    image_path = os.path.join(image_dir, image_name)
    mask_paths = [os.path.join(mask_dir, image_name.replace('.jpg', f'_{lesion}.tif')) for lesion in
                  lesion_mapping.keys()]

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in mask_paths if os.path.exists(mask_path)]
    true_mask = np.sum(masks, axis=0) if masks else None

    image_tensor = a.Compose([a.Resize(256, 256),
                              a.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]), ToTensorV2()])(image=image)['image']
    image_tensor = image_tensor.to(torch.float32).to(device).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

    print(f"Output shapes: {output.shape}")

    predicted_mask = torch.argmax(torch.softmax(output, dim=1), dim=1).cpu().squeeze(0).numpy()

    print(f"predicted masks shapes: {predicted_mask.shape}")

    ttrue_mask = true_mask.astype(np.uint8)

    resized_mask = cv2.resize(ttrue_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    if true_mask is not None and true_mask.dtype != object:
        plt.subplot(1, 4, 2)
        plt.imshow(true_mask, cmap='gray')
        plt.title("True Mask")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(resized_mask, cmap='gray')
        plt.title("Resized Mask")
        plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap='gray')
    plt.axis('off')

    plt.show()

