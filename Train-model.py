from torchvision.models import resnet34
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch
import time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet34(num_classes=3).to(device)
    print(f"Device: {device}")
    
    torch.manual_seed(0)
    # dataset = ImageFolder("/content/Colorectal Cancer/Dataset 1/Colorectal Cancer ", transform=ToTensor())
    dataset = ImageFolder("./Colorectal Cancer", transform=ToTensor())
    train_set, validation_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    train_loader = DataLoader(train_set, shuffle=True, batch_size=64)
    validation_loader = DataLoader(validation_set, shuffle=True, batch_size=64)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=64)
    print("Data loaded")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    loss = torch.nn.CrossEntropyLoss()
    num_epoch = 2

    start = time.time()
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
            # print(f"minibatch loss: {l}")
        print(f"Epoch {epoch}: last batch loss: {l}")
    
    print(f"Time taken: {time.time() - start}")


if __name__ == "__main__":
    main()
