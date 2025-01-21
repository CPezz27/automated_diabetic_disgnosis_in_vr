import os
import numpy as np
import cv2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_dir = 'dataset/IDRiD/train/images'
mask_dir = 'dataset/IDRiD/train/masks'

test_image_dir = 'dataset/IDRiD/test/images'
test_mask_dir = 'dataset/IDRiD/test/masks'


def preprocess_data(image_dir, mask_dir, lesion_types=["MA", "HE", "EX", "SE", "OD"], target_size=(128, 128)):
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


def unet(input_size=(128, 128, 3)):
    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    merge1 = concatenate([conv2, up1], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    merge2 = concatenate([conv1, up2], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge2)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    output = Conv2D(1, 1, activation='sigmoid')(conv5)

    return Model(inputs, output)


model = unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

images, masks = preprocess_data(image_dir, mask_dir)

x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

x_test, y_test = preprocess_data(test_image_dir, test_mask_dir)

image_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

mask_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

seed = 42
image_generator = image_datagen.flow(
    x_train, batch_size=16, seed=seed
)
mask_generator = mask_datagen.flow(
    y_train, batch_size=16, seed=seed
)

train_generator = zip(image_generator, mask_generator)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    train_generator,
    steps_per_epoch=len(x_train),
    validation_data=(x_val, y_val),
    epochs=25,
    batch_size=1,
    callbacks=[early_stopping]
)

model.save('lesion_mask_model.keras')

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

random_index = np.random.randint(0, len(images))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(images[random_index])
plt.title("Image")
plt.subplot(1, 2, 2)
plt.imshow(masks[random_index], cmap='gray')
plt.title("Mask")
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
