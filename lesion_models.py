from utils_classification import *

num_epochs = 100
batch_size = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cls_model(batch_size, num_epochs)

feature_extractor = models.efficientnet_b0(pretrained=False)
feature_extractor.classifier = nn.Identity()
feature_extractor.load_state_dict(torch.load("saved_models/feature_extractor.pth", map_location=device))
feature_extractor.to(device)
feature_extractor.eval()

image_path = "dataset/IDRiD/DiseaseGrading/OriginalImages/b. Testing Set/IDRiD_001.jpg"

show_gradcam(feature_extractor, image_path, device=device)
