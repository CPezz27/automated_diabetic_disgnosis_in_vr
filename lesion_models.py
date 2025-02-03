from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir_classification = 'dataset/IDRiD/DiseaseGrading/OriginalImages/a. Training Set'
mask_dir = 'dataset/IDRiD/train/masks'

num_classes = 5
num_epochs = 25
batch_size = 8

lesion_mapping = {"MA": 1, "HE": 2, "EX": 3, "SE": 4, "OD": 5}

rare_classes = {'MA': 1, 'SE': 4}


segmentation_model(num_epochs, batch_size)
# classification_model(image_dir_classification, num_classes, num_epochs, batch_size, csv_path)

predict_segmentation(lesion_mapping)
