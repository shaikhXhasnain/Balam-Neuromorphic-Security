import torch 
import snntorch.spikegen as spikegen 

static_image = torch.tensor([[0.0,0.5,1.0],
                            [0.1,0.8,0.2],
                            [1.0,0.0,0.9]])

print("--- The Static Image (what the computer sees) ---")
print(static_image)
print("shape", static_image.shape)

time_steps = 10
spike_train = spikegen.rate(static_image , num_steps=time_steps)

print(f"\n--- The Spike Train (what the neuromorphic chip sees over{time_steps}ms) ---")
print(spike_train)
print("New Shape:", spike_train.shape)


