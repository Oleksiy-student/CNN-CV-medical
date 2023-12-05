from torchvision.models import resnet34
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
from numpy import ndarray
from matplotlib import pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Load model into evaluation mode
    model = resnet34(num_classes=3).to(device)
    model.load_state_dict(torch.load("./model.pth", map_location=device))
    model.eval()
    mean = torch.tensor([0.7621, 0.5239, 0.7111])
    std = torch.tensor([0.0066, 0.0096, 0.0063])

    transforms = Compose([ToTensor(), Normalize(mean, std)])

    # dataset = ImageFolder("/content/Colorectal Cancer/Dataset 1/Colorectal Cancer ", transform=transforms)
    dataset = ImageFolder("./Colorectal Cancer", transform=transforms)
    print(dataset.class_to_idx.items())

    # Need class labels to properly show confusion matrix. Ensure they're sorted
    # class_to_idx is a dictinary mapping class names to indices.
    labels = [x[0] for x in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])]

    # Create all loaders, with fixed seed, to ensure it stays the same. 
    torch.manual_seed(0)
    train_set, validation_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    train_loader = DataLoader(train_set, shuffle=True, batch_size=64)
    validation_loader = DataLoader(validation_set, shuffle=True, batch_size=64)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=64)
    print("Data loaded")
    loss = torch.nn.CrossEntropyLoss()

    # Evaluate on validation set
    (validation_loss, validation_accuracy, validation_confusion, validation_report) = evaluate(validation_loader, device, model, loss)
    print(f"On validation set, average loss: {validation_loss} Accuracy: {validation_accuracy}")
    print(validation_confusion)

    # Evaluate on test set (in more detail)
    (test_loss, test_accuracy, test_confusion, test_report) = evaluate(test_loader, device, model, loss)
    print(f"On test set, average loss: {test_loss} Accuracy: {test_accuracy}")
    print(test_confusion)
    print(test_report)

    # Show confusion matrix on test set
    disp = ConfusionMatrixDisplay(test_confusion, display_labels=labels)
    print(f"Type of confusion mtx {type(disp)}")
    disp.plot()
    plt.show()


def evaluate(loader: DataLoader, device: str, model, loss) -> tuple[float, float, ndarray]:
    average_loss = 0.0
    average_accuracy = 0.0
    labels_list = []
    predictions_list = []

    # Load all batches, keep track of the ground truth labels and predictions
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        model_output = model(features)
        class_predictions = model_output.max(1).indices

        labels_list.append(labels)
        predictions_list.append(class_predictions)

        average_loss += loss(model_output, labels).item()
        average_accuracy += (labels == class_predictions).sum() / labels.size(0)

    average_loss /= len(loader)
    average_accuracy /= len(loader)
    # Make confusion matrix
    matrix = confusion_matrix(torch.cat(labels_list), torch.cat(predictions_list))
    # Make classification report
    report = classification_report(torch.cat(labels_list), torch.cat(predictions_list))

    return (average_loss, average_accuracy, matrix, report)


if __name__ == "__main__":
    main()
