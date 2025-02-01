from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir = 'dataset/IDRiD/train/images'
mask_dir = 'dataset/IDRiD/train/masks'
csv_path = 'dataset/IDRiD/DiseaseGrading/Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'

lesion_types = ["EX", "MA", "HE", "SE", "OD"]

num_classes_seg = len(lesion_types)

num_classes_cls = 5
num_epochs = 10
batch_size = 8

train_and_validate(image_dir, mask_dir, csv_path, lesion_types, num_classes_seg, num_classes_cls, num_epochs,
                   batch_size)
