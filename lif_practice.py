import torch
import snntorch as snn 

beta = 0.9
lif_neuron = snn.Leaky(beta=beta)

mem = lif_neuron.init_leaky()

print("--- Starting Water Level ---")
print("Membrane Potential:",mem)

incoming_pulse = torch.tensor([2.0])

spk,mem = lif_neuron(incoming_pulse,mem)

print("\n---After The Energy pulse ---")
print("Did It Spike? (1=Yes, 0=No):",spk)
print("New Water Level:",mem)

