from torchvision.models import resnet34
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = resnet34(num_classes=3).to(device)
    model.load_state_dict(torch.load("./model.pth", map_location=device))
    model.eval()
    mean = torch.tensor([0.7621, 0.5239, 0.7111])
    std = torch.tensor([0.0066, 0.0096, 0.0063])
    transforms = Compose([ToTensor(), Normalize(mean, std)])

    torch.manual_seed(0)
    # dataset = ImageFolder("/content/Colorectal Cancer/Dataset 1/Colorectal Cancer ", transform=transforms)
    dataset = ImageFolder("./Colorectal Cancer", transform=transforms)
    train_set, validation_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    train_loader = DataLoader(train_set, shuffle=True, batch_size=64)
    validation_loader = DataLoader(validation_set, shuffle=True, batch_size=64)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=64)
    print("Data loaded")
    loss = torch.nn.CrossEntropyLoss()

    average_loss = 0.0
    average_accuracy = 0.0

    # Evaluate
    for features, labels in validation_loader:
        features = features.to(device)
        labels = labels.to(device)
        y_pred = model(features)
        average_loss += loss(y_pred, labels).item()
        average_accuracy += (labels == y_pred.max(1).indices).sum() / labels.size(0)

    average_loss /= len(validation_loader)
    average_accuracy /= len(validation_loader)
    print(f"On validation set, average loss: {average_loss} Accuracy: {average_accuracy}")


if __name__ == "__main__":
    main()
