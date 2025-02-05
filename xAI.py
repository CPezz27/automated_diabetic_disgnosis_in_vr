from utils import *

model = torch.load('saved_models/efficientnet_fundus_classification.pth', map_location=torch.device('cpu'))
model.eval()
print("Modello caricato correttamente!")

class_names = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}

img_path_classification = 'dataset/IDRiD/DiseaseGrading/OriginalImages/b. Testing Set'
img_path = img_path_classification+'/IDRiD_001.jpg'
print(img_path)

test_img = preprocess_grad(img_path)


preprocess_fn = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

visualize_grad_cam(model, img_path, preprocess_fn)
