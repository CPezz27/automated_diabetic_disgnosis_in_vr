from utils import *
from tensorflow.keras.applications import VGG16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VGG16(weights='imagenet')

mask_dir = 'dataset/IDRiD/train/masks'

num_classes = 5
num_epochs = 25
batch_size = 8

lesion_mapping = {"MA": 1, "HE": 2, "EX": 3, "SE": 4, "OD": 5}

rare_classes = {'MA': 1, 'SE': 4}

img_path_classification = 'dataset/IDRiD/DiseaseGrading/OriginalImages/b. Testing Set'
img_path = img_path_classification+'/IDRiD_001.jpg'
print(img_path)
test_img = preprocess_grad(img_path)

# segmentation_model(num_epochs, batch_size)
# classification_model(num_classes, num_epochs, batch_size)

# predict_segmentation(lesion_mapping)


apply_grad_cam(test_img)
