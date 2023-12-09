### Uncomment to use in Colab: ###
# !gdown 1pMV7-v2icU30mA6-DqEYlmnP65AXdVxA -O "Dataset 1.zip"
# !unzip "Dataset 1.zip" -d "."
# !mv "Dataset 1/Colorectal Cancer " "./Colorectal Cancer"

from torchvision.models import resnet34
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms import v2
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet34(num_classes=3).to(device)
    mean = torch.tensor([0.7621, 0.5239, 0.7111])
    std = torch.tensor([0.0066, 0.0096, 0.0063])
    transforms = v2.Compose([ToTensor(), Normalize(mean, std)])
    # transforms = ToTensor()
    print(f"Device: {device}")

    torch.manual_seed(0)
    dataset = ImageFolder("./Colorectal Cancer", transform=transforms)
    train_set, validation_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    train_loader = DataLoader(train_set, shuffle=True, batch_size=64)
    validation_loader = DataLoader(validation_set, shuffle=True, batch_size=64)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=64)
    print("Data loaded")
    # find_mean(train_loader)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.008, momentum=0.9, weight_decay=0.001)
    loss = torch.nn.CrossEntropyLoss()
    num_epoch = 10

    for epoch in range(num_epoch):
        # Pass an epoch over the training data in batch_size chunks
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            # forward
            y_pred = model(features)
            l = loss(y_pred, labels)
            model.zero_grad()
            # backprop and step
            l.backward()
            optimizer.step()
        print(f"Epoch {epoch}: last batch loss: {l}")
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
    # Save
    torch.save(model.state_dict(), "./model.pth")


# Get per channel mean and std deviation - prints to console
# NOTE: must be run without a Normalize transform on input
def find_mean(train_loader: DataLoader):
  means = torch.zeros(3)
  std = torch.zeros(3)
  for features, _ in train_loader:
    means += features.mean((0,2,3))
  means = means / len(train_loader)
  
  for features, _ in train_loader:
    std += torch.square(features.mean((0,2,3)) - means)
  std = torch.sqrt(std / len(train_loader))
  print(f"Means: {means} Std: {std}")

if __name__ == "__main__":
    main()
