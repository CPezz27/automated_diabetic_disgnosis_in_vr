from utils import *

num_epochs = 5
batch_size = 8
lesion_mapping = {"MA": 1, "HE": 2, "EX": 3, "SE": 4, "OD": 5}

# segmentation_model(batch_size, num_epochs)
classification_model(batch_size, num_epochs)

# predict_segmentation(lesion_mapping)
result = predict_classification('dataset/IDRiD/DiseaseGrading/OriginalImages/b. Testing Set/IDRiD_006.jpg',
                                'dataset/IDRiD/DiseaseGrading/Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv')
print(f'Predicted class: {result}')
