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


class RGBADataGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.classes = generator.classes
        self.class_indices = generator.class_indices
        self.filenames = generator.filenames

    def __iter__(self):
        for images, labels in self.generator:
            images = np.array([self.preprocess_image(image) for image in images])
            yield images, labels

    def preprocess_image(self, image):
        image = Image.fromarray((image * 255).astype(np.uint8))  # Convert to PIL
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        return np.array(image) / 255.0

    def reset(self):
        if hasattr(self.generator, 'reset'):
            self.generator.reset()


train_dir = 'dataset/oversampled_train'
test_dir = 'dataset/standardize_dataset/test'

train_datagen = ImageDataGenerator(
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

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_batches = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=True
)

test_batches = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

train_batches = RGBADataGenerator(train_batches)
test_batches = RGBADataGenerator(test_batches)

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(224, 224, 4)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


def warmup_scheduler(epoch, lr):
    return lr * (epoch + 1) / 5 if epoch < 5 else lr


lr_warmup = LearningRateScheduler(warmup_scheduler)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = model.fit(
    train_batches,
    validation_data=test_batches,
    epochs=10,
    callbacks=[early_stopping, lr_warmup, lr_scheduler],
    verbose=1
)

model.save('retinal_cnn_model.keras')

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

plt.tight_layout()
plt.show()

test_batches.reset()
y_true = test_batches.classes
y_pred_prob = model.predict(test_batches)
y_pred = np.argmax(y_pred_prob, axis=1)


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


class_names = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']
plot_confusion_matrix(y_true, y_pred, class_names)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
