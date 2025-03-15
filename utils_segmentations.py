from segmentation_model import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        if inputs.shape != targets.shape:
            targets = F.interpolate(targets, size=inputs.shape[2:], mode="nearest")

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return BCE + dice_loss


class DriveDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert("L")

        if index < len(self.mask_paths):
            mask_path = self.mask_paths[index]
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
            else:
                mask = Image.new("L", (256, 256), 0)
        else:
            mask = Image.new("L", (256, 256), 0)

        img = self.transform(img)
        mask = self.transform(mask)

        return img, mask


def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    model.train()

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def segmentation_model(lesion_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50

    train_images = [os.path.join("dataset/IDRiD/train/images", fname) for fname in
                    os.listdir("dataset/IDRiD/train/images")]

    train_masks = [os.path.join("dataset/IDRiD/train/masks", lesion_type, fname) for fname in
                   os.listdir(os.path.join("dataset/IDRiD/train/masks", lesion_type))]

    test_images = [os.path.join("dataset/IDRiD/test/images", fname) for fname in
                   os.listdir("dataset/IDRiD/test/images")]
    test_masks = [os.path.join("dataset/IDRiD/test/masks", lesion_type, fname) for fname in
                  os.listdir(os.path.join("dataset/IDRiD/test/masks", lesion_type))]

    train_dataset = DriveDataset(train_images, train_masks)
    test_dataset = DriveDataset(test_images, test_masks)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, test_loader, loss_fn, device)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    save_dir = f"saved_models/segmentations/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"unet_{lesion_type}_segmentation.pth")
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully!")

    model = UNet().to(device)
    model.load_state_dict(torch.load(f"saved_models/segmentations/unet_{lesion_type}_segmentation.pth"))
    model.eval()
    print("Model loaded successfully!")

    def predict(model, image_path):
        model.eval()

        img = Image.open(image_path).convert("L")
        img = transforms.Resize((256, 256))(img)
        img = transforms.ToTensor()(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            output = torch.sigmoid(output)
            output = output.squeeze().cpu().numpy()

        return output

