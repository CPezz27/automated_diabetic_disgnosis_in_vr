from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import os
import shutil
from collections import Counter

train_dir = 'dataset/oversampled_train'
test_dir = 'dataset/standardize_dataset/test'


def oversample_dataset(source_dir, output_dir, target_count_per_class):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
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
            images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if
                      img.lower().endswith(('png', 'jpg', 'jpeg'))]
            class_count = len(images)

            for img_path in images:
                shutil.copy(img_path, output_class_dir)

            while class_count < target_count_per_class:
                for img_path in images:
                    if class_count >= target_count_per_class:
                        break

                    with Image.open(img_path) as img:
                        if img.mode != 'RGBA':
                            img = img.convert('RGBA')
                        img_array = np.expand_dims(np.array(img), axis=0)

                    for batch in datagen.flow(img_array, batch_size=32, save_to_dir=output_class_dir,
                                              save_prefix='aug', save_format='png'):
                        class_count += len(batch)
                        if class_count >= target_count_per_class:
                            break

    print(f"Oversampling completed in {output_dir}")
    for class_name in os.listdir(output_dir):
        print(f"{class_name}: {len(os.listdir(os.path.join(output_dir, class_name)))} images")


target_count = 55000
oversampled_train_dir = "dataset/oversampled_train"
oversample_dataset(train_dir, oversampled_train_dir, target_count)
