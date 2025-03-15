from utils_classification import *
from utils_segmentations import *

num_epochs = 100
batch_size = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
model = load_best_model()

image_path = "dataset/IDRiD/DiseaseGrading/OriginalImages/b. Testing Set/IDRiD_011.jpg"
'''

cls_model(batch_size, num_epochs)
'''
segmentation_model(lesion_type='MA')
segmentation_model(lesion_type='OD')
segmentation_model(lesion_type='EX')
segmentation_model(lesion_type='HE')
segmentation_model(lesion_type='SE')
'''