import os
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir = 'dataset/IDRiD/train/images'
mask_dir = 'dataset/IDRiD/train/masks'

lesion_mask_types = ["EX", "MA", "HE", "SE", "OD"]


class CombinedDataset(Dataset):
    def __init__(self, inputs, masks):
        self.inputs = inputs
        self.masks = masks

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32).permute(2, 0, 1)
        mask_tensor = torch.tensor(self.masks[idx], dtype=torch.float32).unsqueeze(0)
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


def combine_image_with_mask(image, mask):
    result = np.concatenate([image, np.dstack(mask)], axis=-1)
    return result


def calculate_metrics(y_pred, y_true):
    y_true = (y_true > 0.5).float()
    y_pred = (y_pred > 0.5).float()

    precision = precision_score(y_true.cpu().numpy().ravel(), y_pred.cpu().numpy().ravel())
    recall = recall_score(y_true.cpu().numpy().ravel(), y_pred.cpu().numpy().ravel())
    f1 = f1_score(y_true.cpu().numpy().ravel(), y_pred.cpu().numpy().ravel())

    return precision, recall, f1


def tversky_loss(y_pred, y_true, alpha=0.6, beta=0.4, smooth=1e-6, class_weights=None):

    print(f"y_pred: {y_pred.shape}")

    print(f"y_true: {y_true.shape}")

    y_true = (y_true > 0.5).float()

    print(f"y_pred: {y_pred.shape}")

    print(f"y_true: {y_true.shape}")

    tp = (y_pred * y_true).sum(dim=(2, 3))
    fp = ((1 - y_true) * y_pred).sum(dim=(2, 3))
    fn = (y_true * (1 - y_pred)).sum(dim=(2, 3))
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

    if class_weights is not None:
        tversky_index *= class_weights

    return 1 - tversky_index.mean()


def dice_score(y_pred, y_true, smooth=1e-6):
    y_true = (y_true > 0.5).float()

    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.mean().item()


def iou_score(y_pred, y_true, smooth=1e-6):
    y_true = (y_true > 0.5).float()

    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()


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
        masks_batch = masks_batch.view(masks_batch.size(0), -1, num_classes)
        for c in range(num_classes):
            class_counts[c] += (masks_batch == c).sum()

    total_count = class_counts.sum()

    print(f"Class counts: {class_counts}")

    class_weights = total_count / (class_counts + 1e-6)
    return class_weights / class_weights.sum()


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

    combined_inputs = []
    images = []
    combined_masks = []
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
                mask = (cv2.resize(mask, target_size) > 0).astype(np.float32)
            else:
                mask = np.zeros(target_size)

            mask_stack.append(mask)

        combined_input = combine_image_with_mask(img, mask_stack)
        combined_inputs.append(combined_input)

        masks.append(np.stack(mask_stack, axis=-1))

    for mask_stack in masks:
        combined_mask = np.any(mask_stack, axis=-1).astype(np.float32)
        combined_masks.append(combined_mask)

        # print(f"combined_masks shapes: {combined_masks}")

    return np.array(images), np.array(combined_inputs), np.array(combined_masks)


images, combined_mask_images, combined_masks = preprocess_data(image_dir, mask_dir, lesion_mask_types)

k = 5
num_classes = 2
kf = KFold(n_splits=k, shuffle=True, random_state=42)

lesion_model = smp.DeepLabV3Plus(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    in_channels=3 + len(lesion_mask_types),
    classes=2,
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

num_epochs = 50

optimizer = optim.Adam(lesion_model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

for fold, (train_idx, val_idx) in enumerate(kf.split(combined_mask_images)):
    print(f"Fold {fold + 1}/{k}")

    train_images, val_images = combined_mask_images[train_idx], combined_mask_images[val_idx]
    train_combined_masks, val_combined_masks = combined_masks[train_idx], combined_masks[val_idx]

    print(f"train_images: {len(train_images)}, val_images: {len(val_images)}, "
          f"train_combined_masks: {len(train_combined_masks)}, val_combined_mask: {len(val_combined_masks)}")

# debug
    train_combined_masks_tensor = torch.tensor(train_combined_masks)
    '''
    print(f"Unique classes in train_combined_masks: {torch.unique(train_combined_masks_tensor)}")

    print(f"Shape of train_combined_masks: {train_combined_masks.shape}")
    '''
# fine debug

    train_dataset = CombinedDataset(train_images, train_combined_masks)
    val_dataset = CombinedDataset(val_images, val_combined_masks)

    print(f"Dataset length: {len(train_dataset)}")

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
            # print(f"Output range before attivation: min={outputs.min().item()}, max={outputs.max().item()}")

            outputs = torch.softmax(outputs, dim=1)

            # print(f"Output range after attivation: min={outputs.min().item()}, max={outputs.max().item()}")

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

                # print(f"mask batch 1: {masks_batch.shape}")

                outputs = lesion_model(images_batch)
                probs = torch.softmax(outputs, dim=1)
                true_labels = (masks_batch.cpu().numpy() > 0).astype(int)

                try:
                    auc = roc_auc_score(true_labels.ravel(), probs.cpu().numpy().ravel(), average='macro',
                                        multi_class='ovr')
                    val_roc_auc += auc
                except ValueError:
                    pass

                val_dice += dice_score(outputs, masks_batch)
                val_iou += iou_score(outputs, masks_batch)

                # print(f"mask batch after 2: {masks_batch.shape}")

                loss = tversky_loss(probs, masks_batch, class_weights=class_weights)
                val_loss += loss.item()
                # precision, recall, f1 = calculate_metrics(outputs, masks_batch)

                visualize_activations(lesion_model, images_batch)

        scheduler.step(val_loss)

        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        val_roc_auc /= len(val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, Dice Score: {val_dice:.4f}, IoU: {val_iou:.4f}")

os.makedirs('saved_models', exist_ok=True)
torch.save(lesion_model.state_dict(), 'saved_models/lesion_model_with_combined_masks.pth')
