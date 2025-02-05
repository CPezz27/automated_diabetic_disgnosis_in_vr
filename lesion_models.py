from utils import *

num_epochs = 10
batch_size = 8
lesion_mapping = {"MA": 1, "HE": 2, "EX": 3, "SE": 4, "OD": 5}

segmentation_model(batch_size, num_epochs)
# classification_model(num_classes, num_epochs, batch_size)

predict_segmentation(lesion_mapping)

