from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K


model = load_model('retinal_cnn_model.keras')
print("Modello caricato correttamente!")

class_names = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}


def grad_cam(model, image, class_index, layer_name="conv2d_2"):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    pooled_grads = tf.expand_dims(tf.expand_dims(pooled_grads, 0), 0)
    weighted_conv_output = conv_output * pooled_grads
    heatmap = tf.reduce_mean(weighted_conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) 
    heatmap /= np.max(heatmap)  

    return heatmap


def overlay_grad_cam(heatmap, original_image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    if original_image.shape[-1] == 4:
        alpha_channel = np.ones_like(heatmap_color[..., 0]) * 255
        heatmap_color = np.dstack([heatmap_color, alpha_channel])
    elif original_image.shape[-1] == 3 and heatmap_color.shape[-1] == 4:
        heatmap_color = heatmap[..., :3]

    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


test_dir = 'dataset/augmented_resized_V2/test'
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_batches = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

image_batch, label_batch = next(test_batches)
random_index = np.random.randint(0, len(image_batch))
image = image_batch[random_index]
label = label_batch[random_index]

image = np.concatenate([image, np.ones((*image.shape[:2], 1))], axis=-1)

predictions = model.predict(np.expand_dims(image, axis=0))
predicted_class = np.argmax(predictions[0])
predicted_probability = predictions[0][predicted_class]*100
predicted_label = class_names[predicted_class]
print(f"Predicted class: {predicted_class}, Probability: {predicted_probability:.2f}%")

heatmap = grad_cam(model, image, predicted_class, layer_name="conv2d_1")
original_image = (image * 255).astype(np.uint8)

overlay = overlay_grad_cam(heatmap, original_image)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap, cmap='jet')
plt.colorbar()
plt.title("Heatmap")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title(f"Grad-CAM Overlay\nPredicted: {predicted_label}")
plt.axis("off")

plt.tight_layout()
plt.show()
