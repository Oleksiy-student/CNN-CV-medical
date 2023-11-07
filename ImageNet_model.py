import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm

def extract_features(model, loader):
    features = []
    
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extracting features"):
            # Forward pass through the ResNet model
            features_batch = model(images)
            # Flatten the features
            features_batch = features_batch.view(features_batch.size(0), -1)
            features.append(features_batch)
    
    return torch.cat(features, dim=0)

def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define data transformations
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # load datasets
    dataset2 = datasets.ImageFolder(root='./Prostate Cancer', transform=transform)
    dataset3 = datasets.ImageFolder(root='./Animal Faces', transform=transform)

    # create data loaders (no shuffling - need to match feature vectors with images)
    data_loader2 = DataLoader(dataset2, batch_size=64, shuffle=False, num_workers=4)
    data_loader3 = DataLoader(dataset3, batch_size=64, shuffle=False, num_workers=4)

    # create model pre-trained on ImageNet only
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # remove classification layer
    model = nn.Sequential(*list(model.children())[:-1])

    # set model to testing mode
    model.eval()

    # use multi-GPUs if available and move to selected device
    model = nn.DataParallel(model)
    model = model.to(device)

    dataset2_features = extract_features(model, data_loader2)
    dataset3_features = extract_features(model, data_loader3)

if __name__ == "__main__":
    main()