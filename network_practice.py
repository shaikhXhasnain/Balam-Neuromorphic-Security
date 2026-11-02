import torch 
import torch.nn as nn

my_brain_layer = nn.Linear(in_features = 10 , out_features = 5)

sensors_in = torch.rand(10)

print("--- Incoming Signals from Sensors ---")
print(sensors_in)

brain_out = my_brain_layer(sensors_in)

print("\n--- Output Signal To The 5 Decision Neurons ---")
print(brain_out)

