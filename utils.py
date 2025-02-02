import os
import numpy as np
import cv2
import torch
import albumentations as a
import segmentation_models_pytorch as smp
import torch.optim as optim
import torch.nn.functional as f
import matplotlib.pyplot as plt
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from efficientnet_pytorch import EfficientNet


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


class FundusDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])

        if not img_name.endswith('.jpg'):
            img_name += '.jpg'

        image = Image.open(img_name).convert('RGB')
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


class DiceFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.8, gamma=2, class_weights=None):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        smooth = 1e-6
        y_pred = torch.softmax(y_pred, dim=1)
        y_true_one_hot = f.one_hot(y_true, num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (y_pred * y_true_one_hot).sum(dim=(2, 3))
        union = y_pred.sum(dim=(2, 3)) + y_true_one_hot.sum(dim=(2, 3))
        dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)

        ce_loss = f.cross_entropy(y_pred, y_true, reduction='mean')
        if self.class_weights is not None:
            ce_loss = ce_loss * self.class_weights[y_true]
        focal_loss = (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss

        return self.alpha * dice_loss.mean() + (1 - self.alpha) * focal_loss.mean()


class MultiClassDataset(Dataset):
    def __init__(self, inputs, masks, augmentations=None, rare_classes=None, rare_augmentations=None):
        self.inputs = inputs
        self.masks = masks
        self.augmentations = augmentations
        self.rare_classes = rare_classes
        self.rare_augmentations = rare_augmentations

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image = self.inputs[idx]
        mask = self.masks[idx]

        if self.rare_classes is not None and self.rare_augmentations is not None:
            if any(cls in mask for cls in self.rare_classes):
                augmented = self.rare_augmentations(image=image.astype(np.float32), mask=mask.astype(np.float32))
                image, mask = augmented['image'], augmented['mask']
        elif self.augmentations:
            augmented = self.augmentations(image=image.astype(np.float32), mask=mask.astype(np.float32))
            image, mask = augmented['image'], augmented['mask']

        return image, mask


def get_sampler(labels, num_classes):
    class_weights = classification_class_weights(labels, num_classes)
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def calculate_metrics(predictions, targets, num_classes):
    predictions = torch.argmax(predictions, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()

    dice_scores = []
    iou_scores = []

    for c in range(num_classes):
        intersection = np.logical_and(predictions == c, targets == c).sum()
        union = np.logical_or(predictions == c, targets == c).sum()
        dice = (2 * intersection) / (np.sum(predictions == c) + np.sum(targets == c) + 1e-6)
        iou = intersection / (union + 1e-6)

        dice_scores.append(dice)
        iou_scores.append(iou)

    return np.mean(dice_scores), np.mean(iou_scores)


def check_data_balance_classification(csv_path, num_classes):
    df = pd.read_csv(csv_path)
    class_counts = df['Retinopathy grade'].value_counts().sort_index()

    print("Class distribution for classification:")
    for i in range(num_classes):
        print(f"Severity {i}: {class_counts.get(i, 0)} samples")


def check_data_balance_segmentation(mask_dir, lesion_types, target_size=(128, 128)):
    lesion_pixel_counts = {lesion: 0 for lesion in lesion_types}

    for filename in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, filename)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, target_size)

            for lesion_type in lesion_types:
                if lesion_type in filename:
                    lesion_pixel_counts[lesion_type] += np.sum(mask > 0)

    print("Pixel distribution for segmentation:")
    for lesion, count in lesion_pixel_counts.items():
        print(f"{lesion}: {count} pixels")


def classification_class_weights(labels, num_classes):
    class_counts = np.bincount(labels, minlength=num_classes)
    total_count = len(labels)
    class_weights = total_count / (class_counts + 1e-6)
    return class_weights / class_weights.sum()


def preprocess_data(image_dir, mask_dir, lesion_types, target_size=(128, 128)):
    images = []
    masks = []

    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size) / 255.0
        images.append(img)
        mask_stack = []

        for lesion_type in lesion_types:
            mask_filename = filename.replace(".jpg", f"_{lesion_type}.tif")
            mask_path = os.path.join(mask_dir, mask_filename)

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, target_size)

                mask = mask / 255.0
                mask = mask.astype(np.float32)
            else:
                mask = np.zeros(target_size)

            mask_stack.append(mask)

        mask_stack = np.stack(mask_stack, axis=-1)
        mask_combined = np.argmax(mask_stack, axis=-1)
        masks.append(mask_combined)

    return np.array(images), np.array(masks)


def segmentation_model(image_dir, mask_dir, lesion_types, num_epochs, batch_size, rare_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(lesion_types)

    images, masks = preprocess_data(image_dir, mask_dir, lesion_types)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    model = smp.Unet(
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        decoder_attention_type="scse"
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    class_weights = torch.tensor([1.0] * num_classes).to(device)
    loss_fn = DiceFocalLoss(class_weights=class_weights)

    for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
        print(f"Fold {fold + 1}/{5}")

        train_dataset = MultiClassDataset(images[train_idx], masks[train_idx], augmentations=augmentations(),
                                          rare_classes=rare_classes)
        val_dataset = MultiClassDataset(images[val_idx], masks[val_idx], augmentations=a.Compose(
            [a.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
                                        rare_classes=rare_classes)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0

            for images_batch, masks_batch in train_loader:
                images_batch = images_batch.to(torch.float32).to(device)
                masks_batch = masks_batch.to(torch.long).to(device)
                optimizer.zero_grad()
                outputs = model(images_batch)
                loss = loss_fn(outputs, masks_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            val_loss, dice_score, iou_score, auc_score = 0, 0, 0, 0

            model.eval()
            with torch.no_grad():
                for images_batch, masks_batch in val_loader:
                    images_batch = images_batch.to(torch.float32).to(device)
                    masks_batch = masks_batch.to(torch.long).to(device)
                    outputs = model(images_batch)
                    loss = loss_fn(outputs, masks_batch)
                    val_loss += loss.item()

                    y_pred = torch.softmax(outputs, dim=1).cpu().numpy()
                    y_pred = y_pred.reshape(-1, num_classes)
                    y_true = masks_batch.cpu().numpy().flatten()
                    y_true = label_binarize(y_true, classes=np.arange(num_classes))

                    auc_score += roc_auc_score(y_true, y_pred, multi_class='ovr')

                    dice, iou = calculate_metrics(outputs, masks_batch, num_classes)
                    dice_score += dice
                    iou_score += iou

                scheduler.step(val_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}, "
                  f"Val Loss: {val_loss / len(val_loader):.4f}, Dice Score: {dice_score / len(val_loader):.4f}, "
                  f"IoU: {iou_score / len(val_loader):.4f}, AUC: {auc_score / len(val_loader):.4f}")

    torch.save(model.state_dict(), 'saved_models/lesion_model_multiclass.pth')


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def classification_model(image_dir, num_classes, num_epochs, batch_size, csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(csv_path)

    label_encoder = LabelEncoder()
    data['Retinopathy grade'] = label_encoder.fit_transform(data['Retinopathy grade'])

    check_data_balance_classification(csv_path, num_classes)
    class_weights = classification_class_weights(data['Retinopathy grade'].values, num_classes)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FundusDataset(train_data, image_dir, transform=transform)
    val_dataset = FundusDataset(val_data, image_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = EfficientNet.from_pretrained(
        'efficientnet-b0',
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0, 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += calculate_accuracy(outputs, labels)


        val_loss, val_acc = 0, 0
        model.eval()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += calculate_accuracy(outputs, labels)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc / len(val_loader):.4f}")

        torch.save(model.state_dict(), "efficientnet_fundus_classification.pth")


def predict(image_dir, mask_dir, lesion_types, num_classes, model_path='saved_models/lesion_model_multiclass.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        decoder_attention_type="scse"
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    images, masks = preprocess_data(image_dir, mask_dir, lesion_types)

    idx = random.randint(0, len(images) - 1)
    image = images[idx]
    true_mask = masks[idx]

    augmentations = a.Compose([
        a.Normalize(),
        ToTensorV2(),
    ])
    augmented = augmentations(image=image.astype(np.float32))
    image_tensor = augmented['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Immagine Originale")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Maschera Vera")
    plt.imshow(true_mask, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Maschera Predetta")
    plt.imshow(pred_mask, cmap='jet')
    plt.axis('off')

    plt.show()
