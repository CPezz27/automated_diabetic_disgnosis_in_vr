import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score


def get_effnet_target_layer(model):
    return model.features[-2]


class IDRiDDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        self.images = self.data['Image name'].tolist()
        self.labels = self.data['Retinopathy grade'].tolist()

        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.images[idx].strip() + '.jpg'
        label = self.labels[idx]

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


train_transform = transforms.Compose([
    transforms.Resize((300, 300)), #qua
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


val_transform = transforms.Compose([
    transforms.Resize((300, 300)), #qua2
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def extract_image_features(image_path, model, transform):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Errore nell'aprire l'immagine {image_path}: {e}")
        return None
    image = transform(image).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        features = model(image)
    return features.squeeze().cpu().numpy()


def get_effnet5(pretrained=True):
    model = models.efficientnet_b3(pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = False

    for param in list(model.features[-3:].parameters()):
        param.requires_grad = True

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 5)

    return model


def cls_model(batch_size, num_epochs, image_dir='dataset/IDRiD/DiseaseGrading/OriginalImages/a. Training Set',
              csv_path='dataset/IDRiD/DiseaseGrading/Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(csv_path)
    X = data['Image name'].tolist()
    y = data['Retinopathy grade'].tolist()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    train_df = pd.DataFrame({'Image name': X_train, 'Retinopathy grade': y_train})
    val_df = pd.DataFrame({'Image name': X_val, 'Retinopathy grade': y_val})
    train_csv_path = 'train_temp.csv'
    val_csv_path = 'val_temp.csv'
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    train_dataset = IDRiDDataset(image_dir, train_csv_path, transform=train_transform)
    val_dataset = IDRiDDataset(image_dir, val_csv_path, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = get_effnet5(pretrained=True)
    model = model.to(device)

    class_counts = np.bincount(y_encoded)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)

    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)

    best_val_loss = float('inf')
    early_patience = 10
    counter = 0

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        all_preds_train = []
        all_labels_train = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
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

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        all_preds_val = []
        all_labels_val = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                preds = outputs.argmax(dim=1)
                val_corrects += torch.sum(preds == labels)

                all_preds_val.extend(preds.cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        precision_val = precision_score(all_labels_val, all_preds_val, average='macro', zero_division=0)
        recall_val = recall_score(all_labels_val, all_preds_val, average='macro', zero_division=0)
        f1_val = f1_score(all_labels_val, all_preds_val, average='macro', zero_division=0)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "saved_models/classification/classifier_best.pth")
            print("Miglior modello salvato!")
        else:
            counter += 1
            if counter >= early_patience:
                print("Early stopping attivato!")
                break

        print(f"Validation - Loss: {val_loss:.4f} - Acc: {val_acc:.4f} "
              f"- Precision: {precision_val:.4f} - Recall: {recall_val:.4f} - F1: {f1_val:.4f}")

    torch.save(model.state_dict(), "saved_models/classification/classifier_final.pth")

    os.remove(train_csv_path)
    os.remove(val_csv_path)

    return model


def load_best_model(checkpoint_path="saved_models/classification/classifier_best.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_effnet5(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def show_gradcam(model, image_path, target_class=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
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

    def forward_hook(module, inp, out):
        activations['value'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)

    if target_class is None:
        target_class = output.argmax(dim=1).item()
        print(f"Target class selezionata: {target_class} - {class_names.get(target_class, 'Unknown')}")
    else:
        if target_class not in class_names:
            raise ValueError(f"La target_class deve essere tra 0 e 4, trovato {target_class}")
        print(f"Target class fornita: {target_class} - {class_names[target_class]}")

    score = output[0, target_class]
    model.zero_grad()
    score.backward()

    act = activations['value']
    grad = gradients['value']

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1)
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

    fwd_handle.remove()
    bwd_handle.remove()

    plt.figure(figsize=(8, 8))
    plt.imshow(superimposed_img)
    plt.title(f"Grad-CAM - {class_names.get(target_class, 'Unknown')}")
    plt.axis('off')
    plt.show()
