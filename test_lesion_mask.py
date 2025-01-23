import os
import random

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir = 'dataset/IDRiD/test/images'

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


lesion_model = smp.DeepLabV3Plus(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=3,
    classes=4,
    activation='softmax'
).to(device)
lesion_model.load_state_dict(torch.load('saved_models/lesion_model.pth', map_location=device))
lesion_model.eval()

optic_disc_model = smp.DeepLabV3Plus(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation='sigmoid'
).to(device)
optic_disc_model.load_state_dict(torch.load('saved_models/optic_disc_model.pth', map_location=device))
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
