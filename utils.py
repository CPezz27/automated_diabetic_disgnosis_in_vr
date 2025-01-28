import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import segmentation_models_pytorch as smp
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MultiClassDataset(Dataset):
    def __init__(self, inputs, masks):
        self.inputs = inputs
        self.masks = masks

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32).permute(2, 0, 1)
        mask_tensor = torch.tensor(self.masks[idx], dtype=torch.long)
        return input_tensor, mask_tensor


def augment_image(image, mask):

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ElasticTransform(p=0.3),
        A.CoarseDropout(p=0.3),
        A.Normalize(),
        A.GridDistortion(),
        A.OpticalDistortion(),
        ToTensorV2(),
    ])

    augmented = transform(image=image, mask=mask)
    return augmented['image'], augmented['mask']


# aoooooooooooooooooooooooooooooooooooooooooooooooooooo
def combine_image_with_mask(image, mask):
    result = np.concatenate([image, np.dstack(mask)], axis=-1)
    return result


def tversky_loss(y_pred, y_true, alpha=0.6, beta=0.4, smooth=1e-6, class_weights=None):
    y_pred = torch.softmax(y_pred, dim=1)
    y_true_one_hot = torch.nn.functional.one_hot(y_true, num_classes=y_pred.size(1))
    y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2).float()

    true_pos = torch.sum(y_pred * y_true_one_hot, dim=(2, 3))
    false_neg = torch.sum(y_true_one_hot * (1 - y_pred), dim=(2, 3))
    false_pos = torch.sum((1 - y_true_one_hot) * y_pred, dim=(2, 3))

    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

    if class_weights is not None:
        tversky_index *= class_weights

    return 1 - tversky_index.mean()


'''y_pred = torch.sigmoid(y_pred)
    y_true = y_true.float()

    print(f"y_pred: {y_pred.shape}")
    print(f"y_true: {y_true.shape}")

    tp = (y_pred * y_true).sum(dim=(2, 3))
    fp = ((1 - y_true) * y_pred).sum(dim=(2, 3))
    fn = (y_true * (1 - y_pred)).sum(dim=(2, 3))
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

    if class_weights is not None:
        tversky_index *= class_weights

    return 1 - tversky_index.mean()
'''

def dice_score(y_pred, y_true, smooth=1e-6):
    y_true = (y_true > 0.5).float()

    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.mean().item()


def calculate_metrics(predictions, targets, num_classes):
    predictions = torch.argmax(predictions, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()

    dice_scores = []
    iou_scores = []

    for c in range(num_classes):
        intersection = np.sum((predictions == c) & (targets == c))
        union = np.sum((predictions == c) | (targets == c))
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


# non so boh, vorrei aggiungere
def visualize_predictions(images_batch, masks_batch, outputs):
    outputs = torch.argmax(outputs, dim=1).cpu().numpy()
    masks_batch = masks_batch.cpu().numpy()

    for i in range(len(images_batch)):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title('Input Image')
        plt.imshow(images_batch[i].cpu().numpy().transpose(1, 2, 0))
        plt.subplot(1, 3, 2)
        plt.title('True Mask')
        plt.imshow(masks_batch[i])
        plt.subplot(1, 3, 3)
        plt.title('Predicted Mask')
        plt.imshow(outputs[i])
        plt.show()


def calculate_class_weights(loader, num_classes):
    class_counts = torch.zeros(num_classes, dtype=torch.float32)

    for _, masks_batch in loader:
        masks_batch = masks_batch.view(-1)
        for c in range(num_classes):
            class_counts[c] += (masks_batch == c).sum().item()

    total_count = class_counts.sum()
    class_weights = total_count / (class_counts + 1e-6)
    return class_weights / class_weights.sum()


# anche qui non so come muovermi
def visualize_activations(model, images_batch):
    model.eval()
    with torch.no_grad():
        activations = model(images_batch)
        activations = torch.softmax(activations, dim=1)
        activations = activations.cpu().numpy()

    for i in range(len(images_batch)):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.title('Input Image')

        image = images_batch[i].cpu().numpy().transpose(1, 2, 0)
        if image.shape[2] > 3:
            image = image[:, :, :3]

        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.title('Activation Map')
        plt.imshow(activations[i, 1], cmap='hot')
        plt.show()


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
            else:
                mask = np.zeros(target_size)

            mask_stack.append(mask)

        mask_stack = np.stack(mask_stack, axis=-1)
        mask_combined = np.argmax(mask_stack, axis=-1)
        masks.append(mask_combined)

    return np.array(images), np.array(masks)


def train_and_validate(image_dir, mask_dir, lesion_types, num_classes, num_epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, masks = preprocess_data(image_dir, mask_dir, lesion_types)

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        classes=num_classes,
        activation=None
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
        print(f"Fold {fold + 1}/{k}")

        train_images, val_images = images[train_idx], images[val_idx]
        train_masks, val_masks = masks[train_idx], masks[val_idx]

        train_dataset = MultiClassDataset(train_images, train_masks)
        val_dataset = MultiClassDataset(val_images, val_masks)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        class_weights = calculate_class_weights(train_loader, num_classes).to(device)
        loss_fn = tversky_loss

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0

            for images_batch, masks_batch in train_loader:
                images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
                optimizer.zero_grad()
                outputs = model(images_batch)
                loss = loss_fn(outputs, masks_batch, class_weights=class_weights)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            val_loss = 0
            dice_score = 0
            iou_score = 0

            model.eval()
            with torch.no_grad():
                for images_batch, masks_batch in val_loader:
                    images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
                    outputs = model(images_batch)
                    loss = loss_fn(outputs, masks_batch)
                    val_loss += loss.item()

                    dice, iou = calculate_metrics(outputs, masks_batch, num_classes)
                    dice_score += dice
                    iou_score += iou

            scheduler.step(val_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}, "
                  f"Val Loss: {val_loss / len(val_loader):.4f}, Dice Score: {dice_score / len(val_loader):.4f}, "
                  f"IoU: {iou_score / len(val_loader):.4f}")

    torch.save(model.state_dict(), 'saved_models/lesion_model_multiclass.pth')
