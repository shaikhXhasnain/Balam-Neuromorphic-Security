import torch
import torch.nn as nn
import snntorch as snn
import snntorch.spikegen as spikegen
from torchvision import datasets, transforms

# 1. Load the Data
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
loader = torch.utils.data.DataLoader(mnist_data, batch_size=1, shuffle=True)
image, label = next(iter(loader))

# 2. Define the Brain (784 pixels IN -> 10 digit choices OUT)
class BalamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10) 
        self.lif1 = snn.Leaky(beta=0.9)

    def forward(self, x):
        mem = self.lif1.init_leaky()
        spk_recording = []
        for step in range(x.size(0)): # Loop through time
            synaptic_current = self.fc1(x[step])
            spk, mem = self.lif1(synaptic_current, mem)
            spk_recording.append(spk)
        return torch.stack(spk_recording)

model = BalamNet()

# 3. The Optic Nerve (Processing)
# Flatten the 2D image [1, 1, 28, 28] into a 1D line of 784 pixels [1, 784]
flat_image = image.view(1, -1) 

# Convert static pixels into a 10ms spike train
spike_data = spikegen.rate(flat_image, num_steps=10)

# 4. Feed it to the Brain!
print(f"--- TRUE LABEL: The image is a {label.item()} ---")
output_spikes = model(spike_data)

# 5. The Brain's Guess
# output_spikes shape is [10 time_steps, 1 batch, 10 neurons]
# We sum them up over time (dim=0) to see which neuron fired the most!
spike_counts = output_spikes.sum(dim=0) 

print("\n--- NEURON SPIKE COUNTS (Digits 0-9) ---")
print(spike_counts)
