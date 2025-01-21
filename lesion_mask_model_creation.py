import os
import numpy as np
import cv2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

image_dir = 'dataset/IDRiD/images'
mask_dir = 'dataset/IDRiD/masks'


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

    return np.array(images), np.array(masks)


def unet(input_size=(128, 128, 3)):
    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # Decoder
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

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=25,
    batch_size=1,
    callbacks=[early_stopping]
)

model.save('lesion_mask_model.keras')


index = 0
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(images[index])
plt.title("Image")
plt.subplot(1, 2, 2)
plt.imshow(masks[index], cmap='gray')
plt.title("Mask")
plt.show()

