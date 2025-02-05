from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.utils import Sequence
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from albumentations import Compose, RandomCrop, HorizontalFlip, VerticalFlip, ElasticTransform, CoarseDropout, Normalize
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import os


def get_augmentation():
    return Compose([
        RandomCrop(width=112, height=112),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ElasticTransform(p=0.5, sigma=10, alpha=1),
        CoarseDropout(p=0.2),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), p=1),
        ToTensorV2()
    ])


class RGBADataGenerator(Sequence):
    def __init__(self, generator, augmentations=None):
        self.generator = generator
        self.classes = generator.classes
        self.class_indices = generator.class_indices
        self.filenames = generator.filenames
        self.augmentations = augmentations

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        images, labels = self.generator[index]
        images = np.array([self.preprocess_image(image) for image in images])
        if self.augmentations:
            images = np.array(
                [self.augmentations(image=image)['image'] for image in images])  # Applica le trasformazioni
        return images, labels

    def preprocess_image(self, image):
        image = Image.fromarray((image * 255).astype(np.float32))
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        return np.array(image) / 255.0

    def on_epoch_end(self):
        if hasattr(self.generator, 'on_epoch_end'):
            self.generator.on_epoch_end()

    def reset(self):
        if hasattr(self.generator, 'reset'):
            self.generator.reset()


def relabel_classes(generator):
    class_mapping = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4
    }

    class_names = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']

    generator.class_indices = {str(i): class_names[i] for i in range(5)}
    generator.classes = np.array([class_mapping[os.path.basename(os.path.dirname(file_name))] for file_name in generator.filenames])

    return generator


train_dir = 'dataset/oversampled_train'
test_dir = 'dataset/standardize_dataset/test'

train_datagen = ImageDataGenerator(rescale=1. / 255)
'''
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
'''

augmentation_pipeline = get_augmentation()

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_batches = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

test_batches = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

train_batches = RGBADataGenerator(train_batches, augmentations=augmentation_pipeline)
test_batches = RGBADataGenerator(test_batches)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(128, 128, 4)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.7),
    Dense(5, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


def warmup_scheduler(epoch, lr):
    return lr * (epoch + 1) / 5 if epoch < 5 else lr


lr_warmup = LearningRateScheduler(warmup_scheduler)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)


def calculate_class_weights(generator):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(generator.classes),
        y=generator.classes
    )
    return dict(zip(np.unique(generator.classes), class_weights))


class_weights = calculate_class_weights(train_batches)

history = model.fit(
    train_batches,
    validation_data=test_batches,
    epochs=10,
    callbacks=[early_stopping, lr_warmup, lr_scheduler],
    verbose=1,
    workers=8,
    class_weight=class_weights
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
