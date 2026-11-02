import torch 
import snntorch as snn

beta = 0.8
lif_neuron = snn.Leaky(beta=beta)
mem = lif_neuron.init_leaky()

print("--- SYSTEM ONLINE: Monitoring Neuromorphic Chip --- ")

for step in range(10):



    if step == 7:
        incoming_signal = torch.tensor([3.0])
    else:
        incoming_signal = torch.tensor([0.5])



    if incoming_signal.item() > 1.5:
        print(f"\n[ALERT] INTRUSION DETECTED at Step {step}! Signal was {incoming_signal.item()}v")
        print("[ACTION] Scrubbing malicious spike to 0.0v to protect hardware.\n")

        incoming_signal = torch.tensor([0.0])
    
    spk,mem = lif_neuron(incoming_signal, mem)


    print(f"Time {step}ms | Input: {incoming_signal.item()}v | Spike Out: {spk.item()} | Water Level: {mem.item():.2f}")
