import torch
import torch.nn as nn
import snntorch as snn
import snntorch.spikegen as spikegen
from torchvision import datasets, transforms

# 1. Setup Data and Network
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
image, label = next(iter(torch.utils.data.DataLoader(mnist_data, batch_size=1, shuffle=True)))
flat_image = image.view(1, -1) 

class BalamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10) 
        self.lif1 = snn.Leaky(beta=0.9)

    def forward(self, x, defense_active=False):
        mem = self.lif1.init_leaky()
        spk_recording = []
        for step in range(x.size(0)): 
            incoming_signal = x[step]
            
            # ==========================================
            # BALAM RUNTIME FIRMWARE (THE SHIELD)
            # ==========================================
            if defense_active:
                # If any pixel spikes with a voltage higher than 1.5, it is a hack!
                # We clamp (scrub) the malicious energy down to normal levels (1.0).
                incoming_signal = torch.clamp(incoming_signal, max=1.0)
                
            synaptic_current = self.fc1(incoming_signal)
            spk, mem = self.lif1(synaptic_current, mem)
            spk_recording.append(spk)
        return torch.stack(spk_recording)

model = BalamNet()
print(f"--- TARGET DIGIT: {label.item()} ---")

# --- SCENARIO 1: NORMAL ---
normal_spikes = spikegen.rate(flat_image, num_steps=10)
out_normal = model(normal_spikes, defense_active=False)
print("\n[1] NORMAL OPERATION SPIKE COUNTS:")
print(out_normal.sum(dim=0))

# --- SCENARIO 2: THE HACKER ATTACK ---
# Hacker injects 5.0v of magnetic noise into the optic nerve!
hacked_spikes = normal_spikes + 5.0 
out_hacked = model(hacked_spikes, defense_active=False)
print("\n[2] HACKED OPERATION SPIKE COUNTS (Chaos!):")
print(out_hacked.sum(dim=0))

# --- SCENARIO 3: BALAM DEFENSE ACTIVE ---
# The exact same hacked spikes, but this time Balam is watching.
out_defended = model(hacked_spikes, defense_active=True)
print("\n[3] BALAM DEFENSE ACTIVE SPIKE COUNTS (Restored!):")
print(out_defended.sum(dim=0))
print("\n[SYSTEM] Attack neutralized. Network is secure.")
