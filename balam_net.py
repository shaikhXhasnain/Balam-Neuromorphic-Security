import torch
import torch.nn as nn
import snntorch as snn

# 1. Defining the Blueprint of our Empire
class BalamNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # The Hardware Web: Connect 10 sensors to 5 brain neurons
        self.fc1 = nn.Linear(in_features=10, out_features=5)
        
        # The Biology: Our 5 Leaky Integrate-and-Fire neurons
        self.lif1 = snn.Leaky(beta=0.9)

    # 2. The Forward Pass (How time flows through the brain)
    # 'x' is our incoming spike train over time
    def forward(self, x):
        # Initialize the 5 empty water buckets (membrane potentials)
        mem = self.lif1.init_leaky()
        
        # We will record the output spikes to analyze them later
        spk_recording = []
        
        # Loop through every millisecond of time!
        time_steps = x.size(0)
        for step in range(time_steps): 
            current_input = x[step]
            
            # The signal goes through the web...
            synaptic_current = self.fc1(current_input)
            
            # ...and hits the biological neurons to trigger spikes
            spk, mem = self.lif1(synaptic_current, mem)
            
            # Save the spike for this millisecond
            spk_recording.append(spk)
            
        # Stack the list back into a 3D Tensor and return it
        return torch.stack(spk_recording)

# 3. Testing the Castle!
print("--- Initializing BalamNet ---")
model = BalamNet()

# Create a fake 10-millisecond spike train from 10 sensors
# Shape: [Time, Batch, Sensors]
fake_spikes = torch.rand(10, 1, 10)

print("\n--- Feeding Spikes to the Network ---")
output_spikes = model(fake_spikes)

print("Output Spikes Shape:", output_spikes.shape)
print("The Castle is built and functional.")





