import os
import numpy as np
from PIL import Image
import cv2

input_base_dir = 'dataset/oversampled_train'
output_base_dir = 'dataset/oversampled_train'

'''
def remove_background(image):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image * 255))

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    output = remove(img_byte_arr)
    img_without_bg = Image.open(io.BytesIO(output))
    img_without_bg = img_without_bg.convert("RGBA")

    return img_without_bg
'''


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

    img = resize_image(img)
    img = crop_to_content(img)
    img = standardize_brightness_clahe(img)

    return img


def preprocess_and_save_images(input_dir, output_dir):

    processed_count = 0
    total_count = sum(len(files) for _, _, files in os.walk(input_dir))
    print(f"Totale immagini da processare: {total_count}")

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path_dir = os.path.join(output_dir, relative_path)
                output_path = os.path.join(output_path_dir, file)

                os.makedirs(output_path_dir, exist_ok=True)

                try:
                    image = Image.open(input_path)
                    processed_image = preprocess_full_pipeline(image)
                    if isinstance(processed_image, np.ndarray):
                        processed_image = Image.fromarray(processed_image.astype(np.uint8))
                    else:
                        processed_image = processed_image.convert("RGBA")
                    processed_image.save(output_path, format="PNG")
                    processed_count += 1
                    print(f"[{processed_count}/{total_count}] Processato: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"Errore nel processare {input_path}: {e}")

    print(f"Preprocessing completato per {input_dir}, salvato in {output_dir}")


preprocess_and_save_images(input_base_dir, output_base_dir)
