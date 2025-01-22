import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

image_dir = 'dataset/IDRiD/train/images'
mask_dir = 'dataset/IDRiD/train/masks'


def preprocess_data(image_dir, mask_dir, lesion_types, target_size=(128, 128)):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)

        combined_mask = None

        for lesion_type in lesion_types:
            mask_filename = filename.replace(".jpg", f"_{lesion_type}.tif")
            mask_path = os.path.join(mask_dir, mask_filename)

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, target_size) / 255.0
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = np.maximum(combined_mask, mask)

        if combined_mask is None:
            print(f"Warning: No masks found for {filename}")
            continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size) / 255.0

        images.append(img)
        masks.append(combined_mask)

    test_images = np.array(images)
    test_masks = np.expand_dims(np.array(masks), axis=-1)
    return test_images, test_masks


lesion_model = load_model('lesion_mask_model.keras')
optic_disc_model = load_model('optic_disc_model.keras')

lesion_mask_types = ["EX", "MA", "HE", "SE"]

images, masks = preprocess_data(image_dir, mask_dir, lesion_mask_types)

optic_disc_type = ["OD"]

images_OD, masks_OD = preprocess_data(image_dir, mask_dir, optic_disc_type)

random_index = np.random.randint(0, len(images))
lesion_prediction = lesion_model.predict(images[random_index:random_index+1])[0, :, :, 0]
disc_prediction = optic_disc_model.predict(images[random_index:random_index+1])[0, :, :, 0]

print(f"Random index: {random_index}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(images[random_index])
plt.title("Original Image")

plt.subplot(1, 4, 2)
plt.imshow(masks[random_index], cmap='gray')
plt.title("True Lesion Mask")

plt.subplot(1, 4, 3)
plt.imshow(lesion_prediction, cmap='gray')
plt.title("Predicted Lesion Mask")


plt.subplot(1, 4, 4)
plt.imshow(disc_prediction, cmap='gray')
plt.title("Predicted Optic Disc")

plt.show()
