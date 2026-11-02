import torch
from torchvision import datasets, transforms

print("--- Initializing Neuromorphic Data Pipeline ---")

# 1. The Translator
# We must convert the raw images into PyTorch Tensors
transform = transforms.ToTensor()

# 2. Download the Dataset (This might take a minute when you run it!)
print("[SYSTEM] Downloading MNIST Dataset into ./data folder...")
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 3. The Conveyor Belt (DataLoader)
# We don't want to feed 60,000 images at once and crash your laptop.
# The DataLoader feeds them in small batches. We will use a batch of 1 for now.
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=True)

# 4. Pull exactly ONE image off the conveyor belt to inspect it
data_iter = iter(train_loader)
image, label = next(data_iter)

print("\n--- Data Successfully Loaded! ---")
print(f"The camera is looking at a handwritten number: {label.item()}")
print(f"The shape of the image tensor is: {image.shape}")
