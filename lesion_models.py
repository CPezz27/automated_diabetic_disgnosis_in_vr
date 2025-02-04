from utils import *
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VGG16(weights='imagenet')

mask_dir = 'dataset/IDRiD/train/masks'
img_path_classification = 'dataset/IDRiD/DiseaseGrading/OriginalImages/b. Testing Set'

img_path = img_path_classification+'/IDRiD_001.jpg'

test_img = preprocess_grad(img_path)

last_conv_layer_name = 'block5_conv3'
heatmap = make_gradcam_heatmap(test_img, model, last_conv_layer_name)
superimposed_img = overlay_heatmap(img_path, heatmap)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image.load_img(img_path, target_size=(224, 224)))
plt.title("Immagine Originale")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(superimposed_img)
plt.title("Grad-CAM")
plt.axis('off')
plt.show()

num_classes = 5
num_epochs = 25
batch_size = 8

lesion_mapping = {"MA": 1, "HE": 2, "EX": 3, "SE": 4, "OD": 5}

rare_classes = {'MA': 1, 'SE': 4}


# segmentation_model(num_epochs, batch_size)
classification_model(num_classes, num_epochs, batch_size)

# predict_segmentation(lesion_mapping)

