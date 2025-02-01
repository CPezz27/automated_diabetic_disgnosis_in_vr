import os
import numpy as np
import cv2
import torch
import albumentations as A
import segmentation_models_pytorch as smp
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder


class DiceFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.8, gamma=2, class_weights=None):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        smooth = 1e-6
        y_pred = torch.softmax(y_pred, dim=1)
        y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (y_pred * y_true_one_hot).sum(dim=(2, 3))
        union = y_pred.sum(dim=(2, 3)) + y_true_one_hot.sum(dim=(2, 3))
        dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)

        ce_loss = F.cross_entropy(y_pred, y_true, reduction='mean')
        focal_loss = (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss

        return self.alpha * dice_loss.mean() + (1 - self.alpha) * focal_loss.mean()


class MultiClassDataset(Dataset):
    def __init__(self, inputs, masks, augmentations=None):
        self.inputs = inputs
        self.masks = masks
        self.augmentations = augmentations

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image = self.inputs[idx]
        mask = self.masks[idx]

        if self.augmentations:
            augmented = self.augmentations(image=image.astype(np.float32), mask=mask.astype(np.float32))
            image, mask = augmented['image'], augmented['mask']

        return image, mask


augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ElasticTransform(p=0.3),
    A.CoarseDropout(p=0.3),
    A.GaussianBlur(p=0.2),
    A.CLAHE(p=0.2),
    A.Normalize(),
    A.GridDistortion(),
    A.OpticalDistortion(),
    ToTensorV2(),
])


def read_dr_severity_labels(csv_path):
    df = pd.read_csv(csv_path)
    image_names = df['Image name'].values
    labels = df['Retinopathy grade'].values
    return image_names, labels


def refine_mask_morphology(mask, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
    return refined_mask


def one_hot_encode_labels(labels, num_classes):
    return label_binarize(labels, classes=range(num_classes))


def split_data(image_names, labels, test_size=0.2, random_state=42):
    return train_test_split(image_names, labels, test_size=test_size, random_state=random_state)


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


def iou_score(y_pred, y_true, smooth=1e-6):
    y_true = (y_true > 0.5).float()

    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()


def calculate_class_weights(loader, num_classes):
    class_counts = torch.zeros(num_classes, dtype=torch.float32)

    for _, masks_batch in loader:
        masks_batch = masks_batch.view(-1)
        for c in range(num_classes):
            class_counts[c] += (masks_batch == c).sum().item()

    total_count = class_counts.sum()
    class_weights = total_count / (class_counts + 1e-6)
    print(f"class_weights: {class_weights}")
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


def train_segmentation_model(image_dir, mask_dir, lesion_types, num_classes, num_epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        train_dataset = MultiClassDataset(images[train_idx], masks[train_idx], augmentations=augmentations)
        val_dataset = MultiClassDataset(images[val_idx], masks[val_idx], augmentations=augmentations)

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
                    # print(f"y_pred prima: {y_pred.shape}") #8, 5, 128, 128
                    y_pred = y_pred.reshape(-1, num_classes)
                    # print(f"y_pred dopo: {y_pred.shape}") #131072, 5
                    y_true = masks_batch.cpu().numpy().flatten()
                    # print(f"y_true prima:  {y_true.shape}") #131072

                    y_true = label_binarize(y_true, classes=np.arange(num_classes))  # (131072, 5)
                    # print(f"y_true shape dopo: {y_true.shape}")

                    auc_score += roc_auc_score(y_true, y_pred, multi_class='ovr')

                    dice, iou = calculate_metrics(outputs, masks_batch, num_classes)
                    dice_score += dice
                    iou_score += iou

            scheduler.step(val_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}, "
                  f"Val Loss: {val_loss / len(val_loader):.4f}, Dice Score: {dice_score / len(val_loader):.4f}, "
                  f"IoU: {iou_score / len(val_loader):.4f}, AUC: {auc_score / len(val_loader):.4f}")

    torch.save(model.state_dict(), 'saved_models/lesion_model_multiclass.pth')


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

    augmentations = A.Compose([
        A.Normalize(),
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
