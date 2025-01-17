from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import os
import shutil
import random
import time

train_dir = 'dataset/oversampled_train'


def oversample_dataset(source_dir, output_dir, target_count_per_class):

    os.makedirs(output_dir, exist_ok=True)

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )

    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        if os.path.isdir(class_dir):

            existing_images = [img for img in os.listdir(output_class_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
            class_count = len(existing_images)

            if class_count == 0:
                original_images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if
                                   img.lower().endswith(('png', 'jpg', 'jpeg'))]
                for img_path in original_images:
                    shutil.copy(img_path, output_class_dir)
                class_count += len(original_images)

            if class_count < target_count_per_class:
                print(f"Augmenting class '{class_name}' from {class_count} to {target_count_per_class} images...")
                images = [os.path.join(output_class_dir, img) for img in os.listdir(output_class_dir) if
                          img.lower().endswith(('png', 'jpg', 'jpeg'))]

                while class_count < target_count_per_class:
                    
                    random.shuffle(images)

                    for img_path in images:
                        if class_count >= target_count_per_class:
                            break

                    with Image.open(img_path) as img:
                        if img.mode != 'RGBA':
                            img = img.convert('RGBA')
                        img_array = np.expand_dims(np.array(img), axis=0)

                    unique_prefix = f"aug_{int(time.time())}"

                    print(f"Starting augmentation for class '{class_name}', currently at {class_count}/{target_count_per_class} images.")

                    for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_class_dir,
                                              save_prefix=unique_prefix, save_format='png'):
                        class_count += 1
                        print(f"Generated {class_count}/{target_count_per_class} images for class '{class_name}'")
                        if class_count >= target_count_per_class:
                            break

    print(f"Oversampling completed in {output_dir}")
    for class_name in os.listdir(output_dir):
        class_path = os.path.join(output_dir, class_name)
        if os.path.isdir(class_path):
            print(f"{class_name}: {len(os.listdir(class_path))} images")


target_count = 55500
oversampled_train_dir = "dataset/oversampled_train"
oversample_dataset(train_dir, oversampled_train_dir, target_count)
