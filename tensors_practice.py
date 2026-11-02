import torch

my_first_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

print("--- My First Tensor ---")
print(my_first_tensor)
print("Shape:", my_first_tensor.shape)

time_steps = 5
batch_size = 1
neurons = 3

snn_tensor = torch.rand(time_steps, batch_size, neurons)

print("\n--- My SNN Tensor (Random Noise) ---")
print(snn_tensor)
print("Shape:", snn_tensor.shape)

suppressed_tensor = snn_tensor / 2

print("\n--- Suppressed Tensor ---")
print(suppressed_tensor)
