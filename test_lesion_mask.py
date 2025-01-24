import os
import random

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir = 'dataset/IDRiD/test/images'

output_dir = 'dataset/IDRiD/test/output_masks'
os.makedirs(output_dir, exist_ok=True)

image_filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_filenames:
    raise FileNotFoundError(f"Nessuna immagine trovata nella directory: {image_dir}")


def preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size) / 255.0
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


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


if not os.path.exists('saved_models/lesion_model.pth'):
    raise FileNotFoundError("Il modello lesion_model.pth non esiste.")
if not os.path.exists('saved_models/optic_disc_model.pth'):
    raise FileNotFoundError("Il modello optic_disc_model.pth non esiste.")


lesion_model = smp.DeepLabV3Plus(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=3,
    classes=4,
    activation='softmax'
).to(device)
lesion_model.load_state_dict(torch.load('saved_models/lesion_model.pth', map_location=device, weights_only=True))
lesion_model.eval()

optic_disc_model = smp.DeepLabV3Plus(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation='sigmoid'
).to(device)
optic_disc_model.load_state_dict(torch.load('saved_models/optic_disc_model.pth', map_location=device, weights_only=True))
optic_disc_model.eval()

random_image_filename = random.choice(image_filenames)
test_image_path = os.path.join(image_dir, random_image_filename)

test_image = preprocess_image(test_image_path).to(device)

with torch.no_grad():
    lesion_prediction = lesion_model(test_image)
    lesion_prediction = torch.argmax(lesion_prediction, dim=1).cpu().numpy()[0]
    lesion_prediction = post_process(lesion_prediction)

with torch.no_grad():
    disc_prediction = optic_disc_model(test_image)
    disc_prediction = torch.sigmoid(disc_prediction).cpu().numpy()[0, 0]
    disc_prediction = post_process(disc_prediction)


lesion_mask_path = os.path.join(output_dir, f"{os.path.splitext(random_image_filename)[0]}_lesion_mask.png")
disc_mask_path = os.path.join(output_dir, f"{os.path.splitext(random_image_filename)[0]}_disc_mask.png")

if cv2.imwrite(lesion_mask_path, (lesion_prediction.squeeze() * 255).astype(np.uint8)):
    print(f"Lesion mask saved in: {lesion_mask_path}")
else:
    print(f"Error file: {lesion_mask_path}")

if cv2.imwrite(disc_mask_path, (disc_prediction.squeeze() * 255).astype(np.uint8)):
    print(f"Disc mask saved in: {disc_mask_path}")
else:
    print(f"Errore file: {disc_mask_path}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.imread(test_image_path)[:, :, ::-1])
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(lesion_prediction.squeeze(), cmap='gray')
plt.title("Predicted Lesion Mask")

plt.subplot(1, 3, 3)
plt.imshow(disc_prediction.squeeze(), cmap='gray')
plt.title("Predicted Optic Disc Mask")

plt.tight_layout()
plt.show()
