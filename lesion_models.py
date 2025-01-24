import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_losses_lesion, val_losses_lesion, val_dice_lesion = [], [], []
train_losses_disc, val_losses_disc, val_dice_disc = [], [], []


def preprocess_masks(masks, num_classes):

    masks_tensor = torch.tensor(masks, dtype=torch.int64)
    print("Before one-hot encoding:", masks_tensor.shape)
    masks_one_hot = torch.nn.functional.one_hot(masks_tensor, num_classes=num_classes)

    print("After one-hot encoding:", masks_one_hot.shape)  # Debug per vedere l'output della one-hot encoding

    if masks_one_hot.shape[2] == 1:
        masks_one_hot = masks_one_hot.squeeze(2)

    masks_one_hot = masks_one_hot.permute(0, 3, 1, 2).float()

    print("After permute and squeeze:", masks_one_hot.min(), masks_one_hot.max())

    return masks_one_hot


def dice_score(y_true, y_pred, num_classes=None):
    if num_classes is None:
        num_classes = y_true.shape[1] if y_true.ndim == 4 else 1

    if y_pred.ndim == 3:
        if torch.max(y_pred) >= num_classes:
            raise ValueError(f"y_pred contiene valori >= num_classes ({num_classes}).")
        y_pred = torch.nn.functional.one_hot(y_pred.long(), num_classes=num_classes).float()
    elif y_pred.ndim == 4 and y_pred.shape[1] != num_classes:
        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = torch.nn.functional.one_hot(y_pred, num_classes).float()

    if y_pred.shape[2:] != y_true.shape[2:]:
        y_pred = F.interpolate(y_pred, size=y_true.shape[2:], mode='bilinear', align_corners=False)
        raise ValueError(f"Dimension mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    y_pred.permute(0, 3, 1, 2)

    y_true_f = y_true.reshape(y_true.size(0), -1)
    y_pred_f = y_pred.reshape(y_pred.size(0), -1)

    intersection = (y_true_f * y_pred_f).sum(dim=1)
    dice = (2. * intersection) / (y_true_f.sum(dim=1) + y_pred_f.sum(dim=1))
    return dice.mean()


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, class_weight=None):

    if y_true.shape[2:] != y_pred.shape[2:]:
        y_pred = y_pred.permute(0, 2, 1, 3)

    tp = torch.sum(y_true * y_pred, dim=(2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(2, 3))
    fn = torch.sum(y_true * (1 - y_pred), dim=(2, 3))

    tversky = tp / (tp + alpha * fp + beta * fn + 1e-7)

    loss = 1 - tversky

    if class_weight is not None:
        class_weight = class_weight.view(1, -1, 1, 1)
        loss = loss.unsqueeze(-1).unsqueeze(-1)
        loss *= class_weight
    return loss.mean()


def post_process(mask, threshold=0.5):
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=0)

    processed_mask = np.zeros_like(mask)
    kernel = np.ones((3, 3), np.uint8)

    for c in range(mask.shape[0]):
        class_mask = mask[c, :, :] > threshold
        class_mask = cv2.morphologyEx(class_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
        processed_mask[c, :, :] = class_mask
    return processed_mask


def combine_image_with_mask(image, mask):
    return np.concatenate([image, mask], axis=-1)


def preprocess_data(image_dir, mask_dir, lesion_types, target_size=(128, 128)):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        mask_stack = []

        for lesion_type in lesion_types:
            mask_filename = filename.replace(".jpg", f"_{lesion_type}.tif")
            mask_path = os.path.join(mask_dir, mask_filename)

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, target_size) / 255.0

                if np.sum(mask) == 0:
                    print(f"Empty mask for {mask_filename}")
            else:
                mask = np.zeros(target_size)

            mask_stack.append(mask)

        if not mask_stack:
            print(f"Warning: No masks found for {filename}")
            continue

        mask_stack = np.stack(mask_stack, axis=-1)
        mask_stack = np.clip(np.sum(mask_stack, axis=-1), 0, 1)

        masks.append(mask_stack)

        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size) / 255.0
        images.append(img)
    ''' Debug:

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(mask_stack, cmap="viridis")
        plt.title("Mask")
        plt.axis("off")

        plt.show()'''

    return np.array(images), np.array(masks)


class CustomDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)
        return image, mask


image_dir = 'dataset/IDRiD/train/images'
mask_dir = 'dataset/IDRiD/train/masks'

lesion_mask_types = ["EX", "MA", "HE", "SE"]
optic_disc_type = ["OD"]


images, masks = preprocess_data(image_dir, mask_dir, lesion_mask_types)
images_OD, masks_OD = preprocess_data(image_dir, mask_dir, optic_disc_type)

masks = preprocess_masks(masks, num_classes=len(lesion_mask_types))
masks_OD = preprocess_masks(masks_OD, num_classes=len(optic_disc_type))

for mask in masks:
    print("Mask shape:", mask.shape)  # Verifica la forma della maschera
    print("Mask min:", mask.min(), "Mask max:", mask.max())  # Debug per vedere i valori min/max
    mask_to_show = np.squeeze(mask[0].cpu().numpy())  # Rimuovi la dimensione extra del batch
    mask_to_show = np.clip(mask_to_show, 0, 1)

    mask_class_to_show = mask_to_show[1]

    plt.subplot(1, 1, 1)
    plt.imshow(mask_to_show, cmap='gray')
    plt.title("Maschera preprocessata")
    plt.show()


images, images_val, masks, masks_val = train_test_split(images, masks, test_size=0.2, random_state=42)
images_OD, val_images_OD, masks_OD, val_masks_OD = train_test_split(images_OD, masks_OD, test_size=0.2, random_state=42)

train_lesion_dataset = CustomDataset(images, masks)
val_lesion_dataset = CustomDataset(images_val, masks_val)
train_optic_dataset = CustomDataset(images_OD, masks_OD)
val_OD_dataset = CustomDataset(val_images_OD, val_masks_OD)

train_lesion_loader = DataLoader(train_lesion_dataset, batch_size=8, shuffle=True)
train_optic_loader = DataLoader(train_optic_dataset, batch_size=8, shuffle=True)
val_lesion_loader = DataLoader(val_lesion_dataset, batch_size=8, shuffle=True)
val_optic_loader = DataLoader(val_OD_dataset, batch_size=8, shuffle=True)


lesion_model = smp.DeepLabV3Plus(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=len(lesion_mask_types),
    activation='softmax'
).to(device)

optic_disc_model = smp.DeepLabV3Plus(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=len(optic_disc_type),
    activation='sigmoid'
).to(device)

class_weight = torch.tensor([0.2, 0.3, 0.3, 0.2], device=device)
loss_fn_lesion = smp.losses.DiceLoss(mode="multiclass")
optimizer_lesion = optim.Adam(lesion_model.parameters(), lr=1e-3)

loss_fn_disc = smp.losses.DiceLoss(mode="binary")
optimizer_optic = optim.Adam(optic_disc_model.parameters(), lr=1e-3)

scheduler_lesion = ReduceLROnPlateau(optimizer_lesion, mode='min', factor=0.5, patience=5, verbose=True)

num_epochs = 5
for epoch in range(num_epochs):
    lesion_model.train()
    epoch_loss_lesions = 0
    for images_batch, masks_batch in train_lesion_loader:
        images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)

        optimizer_lesion.zero_grad()
        outputs = lesion_model(images_batch)

        masks_batch.permute(0, 2, 1, 3)

        loss = tversky_loss(outputs, masks_batch, class_weight=class_weight)
        loss.backward()
        optimizer_lesion.step()
        epoch_loss_lesions += loss.item()

    lesion_model.eval()
    val_loss_lesions = 0
    val_dice_lesions = 0
    with torch.no_grad():
        for images_batch, masks_batch in val_lesion_loader:
            images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
            outputs = lesion_model(images_batch)

            masks_batch = masks_batch.permute(0, 2, 1, 3)

            val_loss = tversky_loss(outputs, masks_batch, class_weight=class_weight)
            val_loss_lesions += val_loss.item()

            dice = dice_score(masks_batch, outputs, num_classes=len(lesion_mask_types))
            val_dice_lesions += dice.item()

    optic_disc_model.train()
    epoch_loss_disc = 0
    for images_batch, masks_batch in train_optic_loader:
        images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)

        optimizer_optic.zero_grad()
        outputs = optic_disc_model(images_batch)

        masks_batch = preprocess_masks(masks_batch, num_classes=len(lesion_mask_types))

        loss = tversky_loss(outputs, masks_batch, class_weight=class_weight)
        loss.backward()
        optimizer_optic.step()
        epoch_loss_disc += loss.item()

    optic_disc_model.eval()
    val_loss_disc = 0
    val_dice_sum_disc = 0
    with torch.no_grad():
        for images_batch, masks_batch in val_optic_loader:
            images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
            outputs = optic_disc_model(images_batch)

            masks_batch = masks_batch.permute(0, 2, 1, 3)

            val_loss = tversky_loss(outputs, masks_batch)
            val_loss_disc += val_loss.item()
            outputs = (outputs > 0.5).float()
            dice = dice_score(masks_batch, outputs)
            val_dice_sum_disc += dice.item()

    train_losses_lesion.append(epoch_loss_lesions / len(train_lesion_loader))
    val_losses_lesion.append(val_loss_lesions / len(val_lesion_loader))
    val_dice_lesion.append(val_dice_lesions / len(val_lesion_loader))

    train_losses_disc.append(epoch_loss_disc / len(train_optic_loader))
    val_losses_disc.append(val_loss_disc / len(val_optic_loader))
    val_dice_disc.append(val_dice_sum_disc / len(val_optic_loader))

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Lesion Train Loss: {epoch_loss_lesions / len(train_lesion_loader):.4f}, "
          f"Lesion Val Loss: {val_loss_lesions / len(val_lesion_loader):.4f}, "
          f"Lesion Val Dice: {val_dice_lesions / len(val_lesion_loader):.4f}, "
          f"Disc Train Loss: {epoch_loss_disc / len(train_optic_loader):.4f}, "
          f"Disc Val Loss: {val_loss_disc / len(val_optic_loader):.4f}, "
          f"Disc Val Dice: {val_dice_sum_disc / len(val_optic_loader):.4f}")

    scheduler_lesion.step(val_loss_lesions / len(val_lesion_loader))

os.makedirs('saved_models', exist_ok=True)
torch.save(lesion_model.state_dict(), 'saved_models/lesion_model.pth')
torch.save(optic_disc_model.state_dict(), 'saved_models/optic_disc_model.pth')

lesion_model.eval()
optic_disc_model.eval()


random_index = np.random.randint(0, len(images))
test_image_lesions = torch.tensor(images[random_index], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
with torch.no_grad():
    lesion_prediction = lesion_model(test_image_lesions)
    lesion_prediction = torch.argmax(lesion_prediction, dim=1).cpu().numpy()[0]
    lesion_prediction = post_process(lesion_prediction)

test_image_disc = torch.tensor(images_OD[random_index], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
with torch.no_grad():
    disc_prediction = optic_disc_model(test_image_disc)
    disc_prediction = torch.sigmoid(disc_prediction).cpu().numpy()[0, 0]
    disc_prediction = post_process(disc_prediction)
'''
plt.figure(figsize=(15, 10))
for i, lesion_type in enumerate(lesion_mask_types):

    predicted_class_mask = np.array(lesion_prediction == i, dtype=np.uint8)
    predicted_class_mask = np.squeeze(predicted_class_mask)
    plt.subplot(2, len(lesion_mask_types), i + 1)
    plt.imshow(predicted_class_mask, cmap='gray')
    plt.title(f"Predicted Mask - {lesion_type}")

    true_class_mask = masks[random_index, :, :, i]
    true_class_mask = np.squeeze(true_class_mask)
    plt.subplot(2, len(lesion_mask_types), len(lesion_mask_types) + i + 1)
    plt.imshow(true_class_mask, cmap='gray')
    plt.title(f"True Mask - {lesion_type}")

    plt.tight_layout()
    plt.show()

'''
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(images[random_index])
plt.title("Original Image")

plt.subplot(1, 4, 2)
combined_mask = np.sum(masks[random_index].cpu().numpy(), axis=0)
plt.imshow(combined_mask, cmap='gray')
plt.title("Combined Mask")

plt.subplot(1, 4, 3)
plt.imshow(np.squeeze(lesion_prediction), cmap='gray')
plt.title("Predicted Lesion Mask")

plt.subplot(1, 4, 4)
plt.imshow(np.squeeze(disc_prediction), cmap='gray')
plt.title("Predicted Optic Disc")

plt.show()

plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(range(num_epochs), train_losses_lesion, label='Lesion Train Loss')
plt.plot(range(num_epochs), val_losses_lesion, label='Lesion Val Loss')
plt.plot(range(num_epochs), val_dice_lesion, label='Lesion Val Dice')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.title('Lesion Model Metrics')

plt.subplot(2, 1, 2)
plt.plot(range(num_epochs), train_losses_disc, label='Disc Train Loss')
plt.plot(range(num_epochs), val_losses_disc, label='Disc Val Loss')
plt.plot(range(num_epochs), val_dice_disc, label='Disc Val Dice')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.title('Optic Disc Model Metrics')

plt.tight_layout()
plt.show()
