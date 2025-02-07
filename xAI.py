from utils import *

model = torch.load('saved_models/efficientnet_fundus_classification.pth', map_location=torch.device('cpu'))
model.eval()
print("Modello caricato correttamente!")

class_names = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}

img_path_classification = 'dataset/IDRiD/DiseaseGrading/OriginalImages/b. Testing Set'
img_path = img_path_classification+'/IDRiD_001.jpg'
print(img_path)

test_img = preprocess_grad(img_path)



def grad_cam(model, image, layer_name):
    model.eval()
    gradients = []
    activations = []

    def hook_function(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook_function(module, inp, out):
        activations.append(out)

    layer = dict([*model.named_modules()])[layer_name]
    forward_hook = layer.register_forward_hook(forward_hook_function)
    backward_hook = layer.register_backward_hook(hook_function)

    output = model(image)
    class_idx = output.argmax(dim=1).item()
    model.zero_grad()
    output[:, class_idx].backward()

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * act, axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image.shape[-1], image.shape[-2]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    forward_hook.remove()
    backward_hook.remove()

    return cam



def load_classification_model(weights_path="best_model.pth", num_classes=5, device="cuda"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    map_location = torch.device(device if torch.cuda.is_available() else "cpu")

    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)

    model._fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model._fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    model.load_state_dict(torch.load(weights_path, map_location=map_location))
    model.to(map_location)
    model.eval()

    print(model)

    return model

def preprocess_grad(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocess_input(img_array)

    img_tensor = torch.tensor(img_array).float()
    img_tensor = img_tensor.permute(0, 3, 1, 2)
    return img_tensor




def superimpose_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img


def visualize_grad_cam(model, img_path, preprocess_fn, target_layer='_conv_head'):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = preprocess_fn
    img_tensor = transform(img).unsqueeze(0)

    heatmap = grad_cam(model, img_tensor, target_layer)
    superimposed_img = superimpose_heatmap(img_path, heatmap)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Immagine originale")

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title("Grad-CAM Overlay")
    plt.show()


def overlay_grad_cam(heatmap, original_image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    if original_image.shape[-1] == 4:
        alpha_channel = np.ones_like(heatmap_color[..., 0]) * 255
        heatmap_color = np.dstack([heatmap_color, alpha_channel])
    elif original_image.shape[-1] == 3 and heatmap_color.shape[-1] == 4:
        heatmap_color = heatmap[..., :3]

    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay




preprocess_fn = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

visualize_grad_cam(model, img_path, preprocess_fn)





