from utils import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir = 'dataset/IDRiD/train/images'
mask_dir = 'dataset/IDRiD/train/masks'

lesion_types = ["EX", "MA", "HE", "SE", "OD"]

num_classes = len(lesion_types)
num_epochs = 50
batch_size = 8

train_and_validate(image_dir, mask_dir, lesion_types, num_classes, num_epochs, batch_size)
