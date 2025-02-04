from utils import *
from keras.models import load_model

model = load_model('best_model.pth')
print("Modello caricato correttamente!")

class_names = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}

img_path_classification = 'dataset/IDRiD/DiseaseGrading/OriginalImages/b. Testing Set'
img_path = img_path_classification+'/IDRiD_001.jpg'
print(img_path)
test_img = preprocess_grad(img_path)

heatmap = grad_cam(model, test_img, class_names)
overlay = overlay_grad_cam(heatmap, test_img)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(test_img)
plt.title("Original image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap, cmap='jet')
plt.colorbar()
plt.title("Heatmap")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title(f"Grad-CAM Overlay\nPredicted:")
plt.axis("off")

plt.tight_layout()
plt.show()
