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
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
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
    def __init__(self, inputs, masks, augmentations=None):
        self.inputs = inputs
        self.masks = masks
        self.augmentations = augmentations

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image = self.inputs[idx]
        mask = self.masks[idx]

        image = image / 255.0

        augmented = self.augmentations(image=image.astype(np.float32), mask=mask.astype(np.float32))
        image, mask = augmented['image'], augmented['mask']

        image = image.clone().detach().to(torch.float32)
        mask = mask.clone().detach().to(torch.float32)

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


def calculate_class_weights(masks):
    class_counts = np.bincount(masks.flatten())
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

        masks.append(binary_mask)

    return np.array(images), np.array(masks)


def segmentation_model(batch_size, num_epochs, image_dir='dataset/IDRiD/train/images',
                       masks_dir='dataset/IDRiD/train/masks'):

    lesion_types = {'MA': 1, 'HE': 2, 'EX': 3, 'SE': 4, 'OD': 5}

    images, masks = preprocess_data(image_dir, masks_dir, lesion_types)
    class_weights = calculate_class_weights(masks)

    print(f"class_weights: {class_weights}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
        print(f"Fold {fold + 1}/{5}")
        print(f"Train indices: {train_idx}")
        print(f"Validation indices: {val_idx}")

        print(f"train idx: {max(train_idx)}")
        print(f"val_idx idx: {max(val_idx)}")
        print(f"val_idx idx: {len(images)}")

        assert max(train_idx) < len(images), "Train indices out of bounds"
        assert max(val_idx) < len(images), "Validation indices out of bounds"

        train_inputs, val_inputs = images[train_idx], images[val_idx]
        train_masks, val_masks = masks[train_idx], masks[val_idx]

        train_dataset = MultiClassDataset(train_inputs, train_masks, augmentations=augmentations())
        val_dataset = MultiClassDataset(val_inputs, val_masks, augmentations=augmentations())

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


def classification_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return (preds == labels).float().mean()


def classification_model(
        num_classes, num_epochs, batch_size, image_dir='dataset/IDRiD/DiseaseGrading/OriginalImages/a. Training Set',
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

    torch.save(model, "saved_models/efficientnet_fundus_classification.pth")


def predict_segmentation(lesion_mapping, image_dir='dataset/IDRiD/test/images', mask_dir='dataset/IDRiD/test/masks',
                         model_path='model_fold_1.pth'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetWithAttention(in_channels=3, out_channels=1)

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

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in mask_paths if os.path.exists(mask_path)]
    true_mask = np.sum(masks, axis=0) if masks else None

    image_tensor = a.Compose([a.Resize(256, 256), a.Normalize(mean=[0.485, 0.456, 0.406],
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


def preprocess_grad(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocess_input(img_array)

    img_tensor = torch.tensor(img_array).float()
    img_tensor = img_tensor.permute(0, 3, 1, 2)
    return img_tensor


def load_classification_model(weights_path="best_model.pth", num_classes=5, device="cuda"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    map_location = torch.device(device if torch.cuda.is_available() else "cpu")

    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)

    model._fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model._fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    model.load_state_dict(torch.load(weights_path, map_location=map_location))
    model.to(map_location)
    model.eval()

    print(model)

    return model


def grad_cam(model, image, layer_name):
    model.eval()
    gradients = []
    activations = []

    def hook_function(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook_function(module, inp, out):
        activations.append(out)

    layer = dict([*model.named_modules()])[layer_name]
    forward_hook = layer.register_forward_hook(forward_hook_function)
    backward_hook = layer.register_backward_hook(hook_function)

    output = model(image)
    class_idx = output.argmax(dim=1).item()
    model.zero_grad()
    output[:, class_idx].backward()

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * act, axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image.shape[-1], image.shape[-2]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    forward_hook.remove()
    backward_hook.remove()

    return cam


def superimpose_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img


def visualize_grad_cam(model, img_path, preprocess_fn, target_layer='_conv_head'):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = preprocess_fn
    img_tensor = transform(img).unsqueeze(0)

    heatmap = grad_cam(model, img_tensor, target_layer)
    superimposed_img = superimpose_heatmap(img_path, heatmap)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Immagine originale")

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title("Grad-CAM Overlay")
    plt.show()


def overlay_grad_cam(heatmap, original_image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    if original_image.shape[-1] == 4:
        alpha_channel = np.ones_like(heatmap_color[..., 0]) * 255
        heatmap_color = np.dstack([heatmap_color, alpha_channel])
    elif original_image.shape[-1] == 3 and heatmap_color.shape[-1] == 4:
        heatmap_color = heatmap[..., :3]

    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
