import random
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.DeepLabV3Plus(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    in_channels=3 + len(["EX", "MA", "HE", "SE"]) + len(["OD"]),
    classes=len(["EX", "MA", "HE", "SE"]) + 1,
    activation=None
).to(device)

model.load_state_dict(torch.load('saved_models/lesion_model_with_combined_masks.pth'), strict=False)

model.eval()

image_dir = 'dataset/IDRiD/test/images'
output_dir = 'dataset/IDRiD/test/output_masks'

os.makedirs(output_dir, exist_ok=True)

image_filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_filenames:
    raise FileNotFoundError(f"Nessuna immagine trovata nella directory: {image_dir}")

image_path = os.path.join(image_dir, random.choice(image_filenames))

image = Image.open(image_path)
image = image.resize((128, 128))

image_array = np.array(image) / 255.0
image_array = np.transpose(image_array, (2, 0, 1))
image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    predictions = model(image_tensor)

predicted_mask = torch.argmax(predictions, dim=1).cpu().numpy()[0]

predicted_mask_image = Image.fromarray(predicted_mask.astype(np.uint8))
predicted_mask_image.save(os.path.join(output_dir, 'output_mask.png'))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Immagine Originale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(predicted_mask, cmap='jet')
plt.title('Maschera di Segmentazione')
plt.axis('off')

plt.show()
