import os
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision import transforms
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from torchvision import models
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def get_effnet_target_layer(model):
    return model.features[-1]


def get_transforms_feat():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def extract_image_features(image_path, model, transform):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Errore nell'aprire l'immagine {image_path}: {e}")
        return None
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.squeeze().cpu().numpy()


def cls_model(batch_size, num_epochs, image_dir='dataset/IDRiD/DiseaseGrading/OriginalImages/a. Training Set',
              csv_path='dataset/IDRiD/DiseaseGrading/Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(csv_path)
    original_images = data['Image name'].tolist()
    original_labels = data['Retinopathy grade'].tolist()

    le = LabelEncoder()
    y_encoded = le.fit_transform(original_labels)

    X_train, X_val, y_train, y_val = train_test_split(
        original_images, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    feature_extractor = models.efficientnet_b0(pretrained=True)
    feature_extractor.classifier = nn.Identity()
    feature_extractor.eval()
    feature_extractor.to(device)

    transform_feat = get_transforms_feat()

    train_image_features = []
    for img_name in X_train:
        img_path = os.path.join(image_dir, img_name.strip() + '.jpg')
        feats = extract_image_features(img_path, feature_extractor, transform_feat)
        if feats is None:
            feats = np.zeros(1280)
        train_image_features.append(feats)
    train_image_features = np.array(train_image_features)
    print("Forma delle feature immagini training:", train_image_features.shape)

    train_data = data[data['Image name'].isin(X_train)]
    train_tabular_features = train_data['Risk of macular edema '].values.reshape(-1, 1)
    print("Forma delle feature tabellari training:", train_tabular_features.shape)

    X_train_combined = np.hstack((train_image_features, train_tabular_features))
    print("Forma delle feature combinate training:", X_train_combined.shape)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)
    print("Dopo SMOTE:")
    print("Forma X_train_resampled:", X_train_resampled.shape)
    print("Distribuzione classi:", np.unique(y_train_resampled, return_counts=True))

    input_dim = X_train_resampled.shape[1]
    classifier = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 5)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-2)

    class CombinedDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = features.astype(np.float32)
            self.labels = labels.astype(np.int64)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

    train_dataset = CombinedDataset(X_train_resampled, y_train_resampled)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_image_features = []
    for img_name in X_val:
        img_path = os.path.join(image_dir, img_name.strip() + '.jpg')
        feats = extract_image_features(img_path, feature_extractor, transform_feat)
        if feats is None:
            feats = np.zeros(1280)
        val_image_features.append(feats)
    val_image_features = np.array(val_image_features)

    val_data = data[data['Image name'].isin(X_val)]
    val_tabular_features = val_data['Risk of macular edema '].values.reshape(-1, 1)
    X_val_combined = np.hstack((val_image_features, val_tabular_features))

    val_dataset = CombinedDataset(X_val_combined, np.array(y_val))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0
        running_corrects = 0
        all_preds_train = []
        all_labels_train = []

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += torch.sum(preds == labels)

            all_preds_train.extend(preds.cpu().numpy())
            all_labels_train.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        precision_train = precision_score(all_labels_train, all_preds_train, average='macro', zero_division=0)
        recall_train = recall_score(all_labels_train, all_preds_train, average='macro', zero_division=0)
        f1_train = f1_score(all_labels_train, all_preds_train, average='macro', zero_division=0)

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} "
              f"- Precision: {precision_train:.4f} - Recall: {recall_train:.4f} - F1: {f1_train:.4f}")

        classifier.eval()
        val_loss = 0.0
        val_corrects = 0
        all_preds_val = []
        all_labels_val = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = classifier(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                preds = outputs.argmax(dim=1)
                val_corrects += torch.sum(preds == labels)

                all_preds_val.extend(preds.cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())
        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        precision_val = precision_score(all_labels_val, all_preds_val, average='macro', zero_division=0)
        recall_val = recall_score(all_labels_val, all_preds_val, average='macro', zero_division=0)
        f1_val = f1_score(all_labels_val, all_preds_val, average='macro', zero_division=0)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(classifier.state_dict(), "saved_models/classifier_best.pth")
            print("Modello salvato con miglior validazione!")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping attivato!")
                break

        print(f"Validation - Loss: {val_loss:.4f} - Acc: {val_acc:.4f} "
              f"- Precision: {precision_val:.4f} - Recall: {recall_val:.4f} - F1: {f1_val:.4f}")

    torch.save(classifier.state_dict(), "saved_models/classification/classifier_combined.pth")
    torch.save(feature_extractor.state_dict(), "saved_models/feature_extractor.pth")


def show_gradcam(model, image_path, target_class=None, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    class_names = {
        0: "No Diabetic Retinopathy",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative"
    }

    target_layer = get_effnet_target_layer(model)

    original_image = Image.open(image_path).convert('RGB')
    input_tensor = test_transform(original_image).unsqueeze(0).to(device)

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)

    if target_class is None:
        target_class = output.argmax(dim=1).item()
        print(f"Target class selezionata: {target_class} - {class_names.get(target_class, 'Unknown')}")
    else:
        if target_class not in class_names:
            raise ValueError(f"La target_class deve essere compresa tra 0 e 4. Valore fornito: {target_class}")
        print(f"Target class fornita: {target_class} - {class_names[target_class]}")

    score = output[0, target_class]
    model.zero_grad()
    score.backward()

    activations_value = activations['value']
    gradients_value = gradients['value']

    weights = gradients_value.mean(dim=(2, 3), keepdim=True)

    cam = torch.sum(weights * activations_value, dim=1)
    cam = F.relu(cam)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()[0]

    cam_resized = cv2.resize(cam, original_image.size)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original_image_np = np.array(original_image)
    superimposed_img = heatmap * 0.4 + original_image_np * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    forward_handle.remove()
    backward_handle.remove()

    plt.figure(figsize=(8, 8))
    plt.imshow(superimposed_img)
    plt.title(f"Grad-CAM - {class_names.get(target_class, 'Unknown')}")
    plt.axis('off')
    plt.show()
