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
    # Find labels for each index and make colours
    labels_per_index = [x[0] for x in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])]
    colors_per_index = {0: "#440354", 1: "#21918c", 2: "#fde726"}
    # Load entire dataset
    data_loader = DataLoader(dataset, shuffle=False, batch_size=64)
    print("Data loaded")

    # Extract features and put on CPU for t-sne
    features, labels = extract_features(resnet_encoder, data_loader, device)
    features = features.to("cpu")
    labels = labels.to("cpu")
    print("Features extracted")

    # Use scikit-learn TSNE
    tsne = TSNE(n_components=2, perplexity=30.0)
    features_tsne = tsne.fit_transform(features)

    fig, ax = plt.subplots()
    # Plot 3 classes
    for i in range(0,3):
        this_class = features_tsne[labels==i]
        ax.scatter(x=this_class[:, 0], y=this_class[:, 1], c=colors_per_index[i], 
                    alpha=0.35, label=labels_per_index[i])
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(loc="upper right", title="Classes")
    ax.add_artist(legend1)
    plt.title("T-SNE Dimension Reduction on Entire Dataset")
    plt.show()


def extract_features(encoder, loader: DataLoader, device: str) -> tuple[Tensor, Tensor]:
    """Take the feature encoder and pass the data through it.
    Returns a tuple: (features, labels) where features is shape
    (num_samples, num_features) and where labels is shape
    (num_samples) """
    
    features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="extracting features"):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass through the ResNet model
            features_batch = encoder(images)
            # Flatten features and store
            features_batch = features_batch.view(features_batch.size(0), -1)
            features.append(features_batch)
            # Store labels
            all_labels.append(labels)
    
    return torch.cat(features, dim=0), torch.cat(all_labels, dim=0)


if __name__ == "__main__":
    main()
