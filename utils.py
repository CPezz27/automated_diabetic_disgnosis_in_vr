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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from efficientnet_pytorch import EfficientNet
from random import randint


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


class FundusSegmentationDataset(Dataset):
    def __init__(self, dataframe, image_dir, mask_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_dir, row['Image name'] + '.jpg')
        mask_path = os.path.join(self.mask_dir, row['Image name'] + '_mask.npy')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if os.path.exists(mask_path):
            mask = np.load(mask_path)
        else:
            mask = np.zeros((128, 128), dtype=np.uint8)

        mask_one_hot = np.eye(2)[mask]

        if self.transform:
            image = self.transform(image)

        mask_tensor = torch.tensor(mask_one_hot, dtype=torch.float32).permute(2, 0, 1)

        label = torch.tensor(row['Retinopathy grade'], dtype=torch.long)

        return image, mask_tensor, label


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


class ClassificationWithSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationWithSegmentation, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.backbone._fc = nn.Identity()

        self.conv1x1 = nn.Conv2d(2, 3, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(self.backbone._fc.in_features + 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, mask):
        mask_features = self.conv1x1(mask)
        combined_input = image + mask_features

        features = self.backbone(combined_input)
        global_features = torch.cat([features, mask.mean(dim=(2, 3))], dim=1)
        output = self.fc(global_features)

        return output


def segmentation_metrics(predictions, targets, num_classes):
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


def calculate_class_weights(masks, num_classes):
    flat_masks = masks.flatten().astype(int)
    unique_classes = np.unique(flat_masks)

    if not np.all(np.isin(unique_classes, np.arange(num_classes))):
        raise ValueError("Le maschere contengono valori di classe non validi")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=flat_masks
    )

    full_class_weights = np.zeros(num_classes, dtype=np.float32)
    for cls, weight in zip(unique_classes, class_weights):
        full_class_weights[cls] = weight

    return torch.tensor(full_class_weights, dtype=torch.float32)


def preprocess_data(image_dir, mask_dir, lesion_types, target_size=(128, 128)):
    images = []
    masks = []

    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size) / 255.0
        images.append(img)

        binary_mask = np.zeros(target_size, dtype=np.uint8)

        for lesion_type in lesion_types:
            mask_filename = filename.replace(".jpg", f"_{lesion_type}.tif")
            mask_path = os.path.join(mask_dir, mask_filename)

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, target_size)
                mask = (mask > 0).astype(np.float32)
                binary_mask = np.maximum(binary_mask, mask)

        masks.append(binary_mask)

    return np.array(images), np.array(masks)


def segmentation_model(num_epochs, batch_size, image_dir='dataset/IDRiD/train/images',
                       mask_dir='dataset/IDRiD/train/masks'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lesion_types = ["MA", "HE", "EX", "SE", "OD"]
    num_classes = 2

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

    class_weights = calculate_class_weights(masks, num_classes).to(device)

    print("class_weights: ", class_weights)

    loss_fn = DiceFocalLoss(class_weights=class_weights)

    for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
        print(f"Fold {fold + 1}/{5}")

        train_dataset = MultiClassDataset(images[train_idx], masks[train_idx], augmentations=augmentations())
        val_dataset = MultiClassDataset(images[val_idx], masks[val_idx], augmentations=a.Compose(
            [a.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]))

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

            val_loss, dice_score, iou_score, auc_score, accuracy = 0, 0, 0, 0, 0
            model.eval()
            with torch.no_grad():
                for images_batch, masks_batch in val_loader:
                    images_batch = images_batch.to(torch.float32).to(device)
                    masks_batch = masks_batch.to(torch.long).to(device)
                    outputs = model(images_batch)
                    loss = loss_fn(outputs, masks_batch)
                    val_loss += loss.item()

                    y_pred = torch.softmax(outputs, dim=1).cpu().numpy()
                    y_pred = np.argmax(y_pred, axis=1).flatten()
                    y_true = masks_batch.cpu().numpy().flatten()

                    accuracy += accuracy_score(y_true, y_pred)
                    auc_score += roc_auc_score(label_binarize
                                               (y_true, classes=np.arange(num_classes)),
                                               label_binarize(y_pred, classes=np.arange(num_classes)),
                                               multi_class='ovr')
                    dice, iou = segmentation_metrics(outputs, masks_batch, num_classes)
                    dice_score += dice
                    iou_score += iou

                scheduler.step(val_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}, "
                  f"Val Loss: {val_loss / len(val_loader):.4f}, Dice Score: {dice_score / len(val_loader):.4f}, "
                  f"IoU: {iou_score / len(val_loader):.4f}, AUC: {auc_score / len(val_loader):.4f}, "
                  f"Accuracy: {accuracy / len(val_loader):.4f}")

    torch.save(model.state_dict(), 'saved_models/lesion_model_binary.pth')


def classification_metrics(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    labels = labels.cpu().numpy()

    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    return precision, recall, f1


def classification_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return (preds == labels).float().mean()


def classification_model(
        num_classes, num_epochs, batch_size, image_dir='dataset/IDRiD/train/images',
        csv_path='dataset/IDRiD/DiseaseGrading/Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(csv_path)

    label_encoder = LabelEncoder()
    data['Retinopathy grade'] = label_encoder.fit_transform(data['Retinopathy grade'])

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

    class_counts = train_data['Retinopathy grade'].value_counts().sort_index().values
    class_weights = 1. / class_counts+1e-6
    sample_weights = class_weights[train_data['Retinopathy grade'].values]

    print("class_weights: ", class_weights)
    print("sample_weights: ", sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FundusDataset(train_data, image_dir, transform=transform)
    val_dataset = FundusDataset(val_data, image_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = EfficientNet.from_pretrained(
        'efficientnet-b4',
        num_classes=num_classes
    ).to(device)

    model._fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model._fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    best_val_loss = float('inf')
    patience = 10
    counter = 0

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
            train_acc += classification_accuracy(outputs, labels)

        val_loss, val_acc, val_precision, val_recall, val_f1 = 0, 0, 0, 0, 0
        model.eval()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += classification_accuracy(outputs, labels)

                precision, recall, f1 = classification_metrics(outputs, labels)
                val_precision += precision
                val_recall += recall
                val_f1 += f1

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "best_model.pth")
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping triggered")
                        break

                scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc / len(val_loader):.4f}, "
              f"Val Precision: {val_precision / len(val_loader):.4f}, Val Recall: {val_recall / len(val_loader):.4f}, "
              f"Val F1: {val_f1 / len(val_loader):.4f}")

    torch.save(model.state_dict(), "saved_models/efficientnet_fundus_classification.pth")


def predict_segmentation(lesion_mapping, image_dir='dataset/IDRiD/test/images', mask_dir='dataset/IDRiD/test/masks',
                         model_path='saved_models/lesion_model_binary.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        decoder_attention_type="scse"
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in the directory.")
        return
    image_name = random.choice(image_files)
    image_path = os.path.join(image_dir, image_name)
    mask_paths = [os.path.join(mask_dir, image_name.replace('.jpg', f'_{lesion}.tif')) for lesion in
                  lesion_mapping.keys()]
    print("mask_paths: ", mask_paths)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in mask_paths if os.path.exists(mask_path)]
    true_mask = np.sum(masks, axis=0) if masks else None

    image_tensor = a.Compose([a.Resize(256, 256), a.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]), ToTensorV2()])(image=image)['image']
    image_tensor = image_tensor.to(torch.float32).to(device).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

    predicted_mask = torch.argmax(torch.softmax(output, dim=1), dim=1).cpu().squeeze(0).numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    if true_mask is not None and true_mask.dtype != object:
        plt.subplot(1, 3, 2)
        plt.imshow(true_mask, cmap='gray')
        plt.title("True Mask")
        plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap='gray')
    plt.axis('off')

    plt.show()


def predict_classification(image_dir, num_classes, model_path='efficientnet_fundus_classification.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i = 0

    model = EfficientNet.from_name("efficientnet-b0", num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for x in image_files:
        i += 1

    if not image_files:
        raise ValueError("No images found in the directory.")

    idx = randint(1, i)

    image_path = os.path.join(image_dir, image_files[idx])

    print(image_path)

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    predicted_class = torch.argmax(output, dim=1).item()

    return predicted_class


def train_classification_with_segmentation(num_classes, num_epochs, batch_size, image_dir, mask_dir, csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(csv_path)
    label_encoder = LabelEncoder()
    data['Retinopathy grade'] = label_encoder.fit_transform(data['Retinopathy grade'])

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FundusSegmentationDataset(train_data, image_dir, mask_dir, transform=transform)
    val_dataset = FundusSegmentationDataset(val_data, image_dir, mask_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ClassificationWithSegmentation(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0, 0

        for images, masks, labels in train_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item
            train_acc += classification_accuracy(outputs, labels)

            val_loss, val_acc = 0, 0
            model.eval()
            with torch.no_grad():
                for images, masks, labels in val_loader:
                    images, masks, labels = images.to(device), masks.to(device), labels.to(device)
                    outputs = model(images, masks)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_acc += classification_accuracy(outputs, labels)

            scheduler.step(val_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, "
                  f"Train Acc: {train_acc / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, "
                  f"Val Acc: {val_acc / len(val_loader):.4f}")

        torch.save(model.state_dict(), "saved_models/classification_with_segmentation.pth")
