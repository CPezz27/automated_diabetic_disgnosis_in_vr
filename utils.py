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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold, train_test_split
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


def get_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.Grayscale(num_output_channels=3),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class FundusDataset(Dataset):
    def __init__(self, data, image_dir, tabular_data, transform=None):
        self.data = data
        self.image_dir = image_dir
        self.tabular_data = tabular_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def load_and_augment_data(self):
        images = []
        tabular_data = []

        for idx, row in self.data.iterrows():
            img_path = os.path.join(self.image_dir, row['Image name'] + '.jpg')
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Impossibile leggere l'immagine: {img_path}")
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (128, 128))
                tab_data = self.tabular_features[idx]

            images.append(image)
            tabular_data.append(tab_data)

        augmented_images, augmented_tabular = cls_augmented_data(images, tabular_data, self.num_augmentations)

        all_images = images + augmented_images
        all_tabular = tabular_data + augmented_tabular

        return all_images, all_tabular


    def __getitem__(self, idx):
        image = self.images[idx]
        tabular_data = self.tabular_data[idx]

        if self.transform is not None and not isinstance(image, torch.Tensor):
            image = self.transform(image)

            if isinstance(image, torch.Tensor):
                pass
            else:
                image = transforms.functional.to_tensor(image)

        image = image.clone().detach().permute(2, 0, 1).to(dtype=torch.float32) / 255.0
        tabular_data = torch.tensor(tabular_data, dtype=torch.float32)

        label = torch.tensor(self.data.iloc[idx]['Retinopathy grade'], dtype=torch.long)
        return image, tabular_data, label


class MultiModalModel(nn.Module):
    def __init__(self, num_tabular_features, num_classes=5):
        super(MultiModalModel, self).__init__()
        self.cnn = EfficientNet.from_pretrained('efficientnet-b0')
        self.cnn._fc = nn.Linear(self.cnn._fc.in_features, 256)

        self.mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.7),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, tabular_data):
        img_features = self.cnn(image)
        tab_features = self.mlp(tabular_data)
        combined = torch.cat((img_features, tab_features), dim=1)
        output = self.classifier(combined)
        return output


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


def classification_metrics(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    labels = labels.cpu().numpy()

    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    return precision, recall, f1


def classification_accuracy(labels, outputs):
    if outputs.ndim > 1 and outputs.shape[1] > 1:
        outputs = outputs.argmax(dim=1)

    if labels.ndim > 1 and labels.shape[1] > 1:
        labels = labels.argmax(dim=1)

    correct = (outputs == labels).sum().item()
    total = labels.numel()

    return correct / total


def cls_augmented_data(images, tabular_data, num_augmentations=25):
    augmented_images = []
    augmented_tabular = []
    augment_fn = get_transforms()

    for image, tab_data in zip(images, tabular_data):
        image_pil = Image.fromarray(image.astype(np.uint8))
        augmented_images.append(transforms.ToTensor()(image_pil))
        augmented_tabular.append(tab_data)

        for _ in range(num_augmentations):
            augmented = augment_fn(image_pil)
            augmented_images.append(augmented)
            augmented_tabular.append(tab_data)

    return torch.stack(augmented_images), np.array(augmented_tabular)


def classification_model(
        batch_size, num_epochs, image_dir='dataset/IDRiD/DiseaseGrading/OriginalImages/a. Training Set',
        csv_path='dataset/IDRiD/DiseaseGrading/Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(csv_path)

    data['Image name'] = data['Image name'].astype(str)
    augmented_names = pd.Series("aug_" + data.index.astype(str))
    data['Image name'] = data['Image name'].fillna(augmented_names)

    label_encoder = LabelEncoder()
    data['Retinopathy grade'] = label_encoder.fit_transform(data['Retinopathy grade'])

    tabular_features = data.drop(columns=['Retinopathy grade', 'Image name', 'Unnamed: 11'])
    tabular_features = tabular_features.dropna(axis=1, how='all')
    column_names = tabular_features.columns

    scaler = StandardScaler()
    tabular_features = scaler.fit_transform(tabular_features)

    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(tabular_features, data['Retinopathy grade'])

    train_data_resampled = pd.DataFrame(x_resampled, columns=column_names)
    train_data_resampled['Retinopathy grade'] = y_resampled
    train_data_resampled['Image name'] = data['Image name'].iloc[:len(y_resampled)]

    print(train_data_resampled['Image name'])

    train_data, val_data = train_test_split(train_data_resampled, test_size=0.2, random_state=42, shuffle=True)

    nan_indices = train_data['Image name'].isna()
    count = 1
    for idx in train_data[nan_indices].index:
        train_data.at[idx, 'Image name'] = f'IDRiD_{count}_aug'
        count += 1

    nan_indices = val_data['Image name'].isna()
    for idx in val_data[nan_indices].index:
        val_data.at[idx, 'Image name'] = f'IDRiD_{count}_aug'
        count += 1

    class_counts = train_data['Retinopathy grade'].value_counts().sort_index().values
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_data['Retinopathy grade'].values]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    transform = get_transforms()

    train_dataset = FundusDataset(train_data, image_dir, x_resampled, transform=transform)
    val_dataset = FundusDataset(val_data, image_dir, tabular_features, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MultiModalModel(num_tabular_features=tabular_features.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    best_val_loss = float('inf')
    patience, counter = 10, 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0, 0

        for images, tabular_data, labels in train_loader:
            images = images.permute(0, 2, 1, 3)
            images = images.to(device)
            tabular_data = tabular_data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, tabular_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += classification_accuracy(outputs, labels)

        val_loss, val_acc, val_precision, val_recall, val_f1 = 0, 0, 0, 0, 0
        model.eval()

        with torch.no_grad():
            for images, tabular_data, labels in val_loader:
                images = images.permute(0, 2, 1, 3)
                images = images.to(device)
                tabular_data = tabular_data.to(device)
                labels = labels.to(device)

                outputs = model(images, tabular_data)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += classification_accuracy(outputs, labels)

                precision, recall, f1 = classification_metrics(outputs, labels)
                val_precision += precision
                val_recall += recall
                val_f1 += f1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "saved_models/classification/best_classification_model.pth")
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

    torch.save(model, "saved_models/classification/efficientnet_fundus_classification.pth")


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


def predict_classification(image_path, csv_path, model_path="saved_models/efficientnet_fundus_classification.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path, map_location=device)
    model.eval()

    data = pd.read_csv(csv_path)
    tabular_features = data.drop(columns=['Retinopathy grade', 'Image name', 'Unnamed: 11'], errors='ignore')
    tabular_features = tabular_features.dropna(axis=1, how='all')

    scaler = StandardScaler()
    tabular_features = scaler.fit_transform(tabular_features)
    tabular_tensor = torch.tensor(tabular_features, dtype=torch.float32).to(device)

    transform = get_transforms()

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    print(f"Image features shape: {image.shape}")
    print(f"Tabular features shape: {tabular_features.shape}")

    with torch.no_grad():
        output = model(image, tabular_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return prediction
