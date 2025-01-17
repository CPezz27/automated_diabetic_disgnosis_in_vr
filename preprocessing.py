import os
import io
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove

input_base_dir = 'dataset/oversampled_train'
output_base_dir = 'dataset/oversampled_train'


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


def standardize_brightness(image, target_brightness=8):

    img = image.convert('RGBA')
    img_array = np.array(img)
    brightness = np.mean(np.sqrt(
        0.241 * img_array[:, :, 0]**2 +
        0.691 * img_array[:, :, 1]**2 +
        0.068 * img_array[:, :, 2]**2
    ))
    brightness_factor = target_brightness / brightness
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(brightness_factor)


def resize_image(image, size=(256, 256)):
    try:
        return image.resize(size, Image.LANCZOS)
    except AttributeError:
        return image.resize(size, Image.ANTIALIAS)


def preprocess_full_pipeline(img):

    img = resize_image(img)
    img = remove_background(img)
    img = standardize_brightness(img)

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
