import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir = 'dataset/IDRiD/train/images'
mask_dir = 'dataset/IDRiD/train/masks'

lesion_mask_types = ["EX", "MA", "HE", "SE"]
optic_disk_type = ["OD"]


class CombinedDataset(Dataset):
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


def combine_image_with_mask(image, mask):
    return np.concatenate([image, np.dstack(mask)], axis=-1)


def tversky_loss(y_pred, y_true, alpha=0.7, beta=0.3, smooth=1e-6, class_weights=None):

    y_true = (y_true > 0.5).float()

    y_true = y_true.permute(0, 3, 1, 2)

    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)

    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    tp = (y_pred * y_true).sum(dim=(2, 3))
    fp = ((1 - y_true) * y_pred).sum(dim=(2, 3))
    fn = (y_true * (1 - y_pred)).sum(dim=(2, 3))

    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

    if class_weights is not None:
        tversky_index *= class_weights

    return 1 - tversky_index.mean()


def dice_score(y_pred, y_true, smooth=1e-6):
    y_true = y_true.permute(0, 3, 1, 2)

    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(y_pred, y_true, smooth=1e-6):
    y_true = y_true.permute(0, 3, 1, 2)

    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def calculate_class_weights(loader, num_classes):
    class_counts = torch.zeros(num_classes, dtype=torch.float32)

    for _, masks_batch in loader:
        masks_batch = masks_batch.view(-1)
        for c in range(num_classes):
            class_counts[c] += (masks_batch == c).sum()

    total_count = class_counts.sum()
    class_weights = total_count / (class_counts + 1e-6)
    return class_weights / class_weights.sum()


def preprocess_data(image_dir, mask_dir, lesion_types, optic_disk_type, target_size=(128, 128)):
    combined_inputs = []
    combined_masks = []

    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size) / 255.0
        mask_stack = []

        for lesion_type in lesion_types:
            mask_filename = filename.replace(".jpg", f"_{lesion_type}.tif")
            mask_path = os.path.join(mask_dir, mask_filename)

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, target_size) / 255.0
            else:
                mask = np.zeros(target_size)

            mask_stack.append(mask)

        optic_disk_filename = filename.replace(".jpg", f"_{optic_disk_type[0]}.tif")
        optic_disk_path = os.path.join(mask_dir, optic_disk_filename)

        if os.path.exists(optic_disk_path):
            optic_disk_mask = cv2.imread(optic_disk_path, cv2.IMREAD_GRAYSCALE)
            optic_disk_mask = cv2.resize(optic_disk_mask, target_size) / 255.0
        else:
            optic_disk_mask = np.zeros(target_size)

        combined_input = np.concatenate([img, np.dstack(mask_stack)], axis=-1)
        combined_inputs.append(combined_input)

        combined_mask = np.dstack(mask_stack + [optic_disk_mask])
        combined_masks.append(combined_mask)

    return np.array(combined_inputs), np.array(combined_masks)


images, combined_masks = preprocess_data(image_dir, mask_dir, lesion_mask_types, optic_disk_type)

train_images, val_images, train_combined_masks, val_combined_mask = train_test_split(images, combined_masks,
                                                                                     test_size=0.2, random_state=42)

k = 5
num_classes = len(lesion_mask_types) + len(optic_disk_type)
kf = KFold(n_splits=k, shuffle=True, random_state=42)

lesion_model = smp.DeepLabV3Plus(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    in_channels=3 + len(lesion_mask_types) + len(optic_disk_type),
    classes=len(lesion_mask_types) + 1,
    activation=None
).to(device)

modules_to_modify = []

for name, module in lesion_model.decoder.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        modules_to_modify.append((name, module))

for name, module in modules_to_modify:
    setattr(lesion_model.decoder, name, torch.nn.Sequential(
        module, torch.nn.Dropout2d(p=0.5)
    ))

optimizer = optim.Adam(lesion_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

num_epochs = 50
for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
    print(f"Fold {fold + 1}/{k}")

    train_images, val_images = images[train_idx], images[val_idx]
    train_combined_masks, val_combined_masks = combined_masks[train_idx], combined_masks[val_idx]

    train_dataset = CombinedDataset(train_images, train_combined_masks)
    val_dataset = CombinedDataset(val_images, val_combined_masks)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    class_weights = calculate_class_weights(train_loader, num_classes).to(device)
    print(f"Class Weights for Fold {fold + 1}: {class_weights}")

    for epoch in range(num_epochs):
        lesion_model.train()
        epoch_loss = 0
        for images_batch, masks_batch in train_loader:
            images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
            optimizer.zero_grad()
            outputs = lesion_model(images_batch)
            outputs = torch.softmax(outputs, dim=1)
            loss = tversky_loss(outputs, masks_batch, class_weights=class_weights)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_dice = 0
        val_iou = 0
        val_roc_auc = 0

        lesion_model.eval()
        val_loss = 0
        with torch.no_grad():
            for images_batch, masks_batch in val_loader:
                images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
                outputs = lesion_model(images_batch)
                probs = torch.softmax(outputs, dim=1)
                true_labels = masks_batch.numpy()

                try:
                    auc = roc_auc_score(true_labels.ravel(), probs.cpu().numpy().ravel(), average='macro',
                                        multi_class='ovr')
                    val_roc_auc += auc
                except ValueError:
                    pass

                val_dice += dice_score(outputs, masks_batch)
                val_iou += iou_score(outputs, masks_batch)

                loss = tversky_loss(probs, masks_batch, class_weights=class_weights)
                val_loss += loss.item()

        scheduler.step(val_loss)

        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        val_roc_auc /= len(val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, Dice Score: {val_dice:.4f}, IoU: {val_iou:.4f}")

os.makedirs('saved_models', exist_ok=True)
torch.save(lesion_model.state_dict(), 'saved_models/lesion_model_with_combined_masks.pth')
