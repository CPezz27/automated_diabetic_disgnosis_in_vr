import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


lesion_mask_model = load_model('lesion_mask_model.keras')

image_path = 'dataset/augmented_resized_V2/test/3/6736_right-600.jpg'
image = Image.open(image_path)


def extract_lesion_mask(image, model, target_size=(128, 128)):

    img_rgb = image.convert("RGB")

    img_resized = cv2.resize(np.array(img_rgb), target_size) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    predicted_mask = model.predict(img_resized)[0]

    print(f"Predicted mask min: {predicted_mask.min()}, max: {predicted_mask.max()}")

    plt.imshow(predicted_mask, cmap='viridis')
    plt.colorbar()
    plt.title("Predicted Mask (Probability Map)")
    plt.show()

    # Controlla i valori della maschera
    print("Unique values in mask:", np.unique(predicted_mask))

    binary_mask = (predicted_mask > 0.007).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    filtered_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 50:
            filtered_mask[labels == i] = 255

    return Image.fromarray(binary_mask.squeeze())


def crop_to_content(image):

    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_image = image.crop((x, y, x + w, y + h))
        return cropped_image
    else:
        return image


def standardize_brightness_clahe(image):
    img_lab = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_lab = cv2.merge((l, a, b))
    img_clahe = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_clahe)


def resize_image(image, size=(128, 128)):
    try:
        return image.resize(size, Image.LANCZOS)
    except AttributeError:
        return image.resize(size, Image.ANTIALIAS)


def preprocess_full_pipeline(img):

    img_resize = resize_image(img)
    img_cropped = crop_to_content(img_resize)
    img_standardize = standardize_brightness_clahe(img_cropped)

    mask = extract_lesion_mask(img_standardize, lesion_mask_model)

    return img_standardize, mask


image2, mask = preprocess_full_pipeline(image)

'''random_index = np.random.randint(0, len(images))'''
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(image2)
plt.title("Preprocessing")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mask, cmap='gray')
plt.title("Mask")
plt.axis("off")

plt.show()
