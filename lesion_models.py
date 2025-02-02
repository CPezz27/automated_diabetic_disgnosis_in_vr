from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir = 'dataset/IDRiD/train/images'
image_dir_classification = 'dataset/IDRiD/DiseaseGrading/OriginalImages/a. Training Set'
mask_dir = 'dataset/IDRiD/train/masks'
csv_path = 'dataset/IDRiD/DiseaseGrading/Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'

lesion_types = ["EX", "MA", "HE", "SE"]

num_classes = 5
num_epochs = 10
batch_size = 8

rare_classes = ['MA', 'SE']

# check_data_balance_classification(csv_path, num_classes=5)
# check_data_balance_segmentation(mask_dir, lesion_types)

# segmentation_model(image_dir, mask_dir, lesion_types, num_epochs, batch_size, rare_classes)
# predict(image_dir, mask_dir, lesion_types, len(lesion_types))
classification_model(image_dir_classification, num_classes, num_epochs, batch_size, csv_path)
