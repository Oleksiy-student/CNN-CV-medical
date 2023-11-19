from torchvision.models import resnet34
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import torch
from torch.nn import Sequential
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from torch import Tensor
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = resnet34(num_classes=3).to(device)
    model.load_state_dict(torch.load("./model.pth", map_location=device))
    # Resnet has 1 FC layer, remove it to get a feature encoder
    resnet_encoder = Sequential(*list(model.children())[:-1])
    resnet_encoder.eval()

    # Use same mean and std as in training and evaluation
    mean = torch.tensor([0.7621, 0.5239, 0.7111])
    std = torch.tensor([0.0066, 0.0096, 0.0063])
    transforms = Compose([ToTensor(), Normalize(mean, std)])

    # dataset = ImageFolder("./Colorectal Cancer/Dataset 1/Colorectal Cancer ", transform=transforms)
    dataset = ImageFolder("./Colorectal Cancer", transform=transforms)
    # find the labels for each index
    labels_per_index = [x[0] for x in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])]
    # Load entire dataset as one w/o shuffling (to match back to labels)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=64)
    print("Data loaded")

    features, labels = extract_features(resnet_encoder, data_loader, device)
    features = features.to("cpu")
    labels = labels.to("cpu")

    print("Features extracted")

    tsne = TSNE(n_components=2, perplexity=30.0)
    features_tsne = tsne.fit_transform(features)

    fig, ax = plt.subplots()
    scatter = ax.scatter(x=features_tsne[:, 0], y=features_tsne[:, 1], c=labels, alpha=0.35)
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
    ax.add_artist(legend1)
    plt.title("T-SNE Dimension Reduction on Entire Dataset")
    plt.show()

def extract_features(model, loader: DataLoader, device: str) -> tuple[Tensor, Tensor]:
    features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="extracting features"):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass through the ResNet model
            features_batch = model(images)
            # Flatten the features
            features_batch = features_batch.view(features_batch.size(0), -1)
            features.append(features_batch)
            # Store labels as well
            all_labels.append(labels)
    
    return torch.cat(features, dim=0), torch.cat(all_labels, dim=0)


if __name__ == "__main__":
    main()
