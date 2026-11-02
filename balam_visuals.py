import torch
import snntorch as snn
import matplotlib.pyplot as plt

print("--- Generating Balam Defense Visuals ---")

# 1. Setup Three Identical Neurons (beta=0.9)
lif = snn.Leaky(beta=0.9)
mem_normal = lif.init_leaky()
mem_hacked = lif.init_leaky()
mem_defended = lif.init_leaky()

# Lists to store the water levels (membrane potentials) over time
record_normal, record_hacked, record_defended = [], [], []

# 2. Simulate 20 milliseconds of time
for step in range(20):
    
    # A normal brain signal is small (0.5v)
    # But from millisecond 5 to 10, the hacker injects a 5.0v EMFI pulse!
    if 5 <= step <= 10:
        raw_signal = 5.0
    else:
        raw_signal = 0.5
        
    # --- SCENARIO 1: NORMAL (No Hacker) ---
    _, mem_normal = lif(torch.tensor([0.5]), mem_normal)
    record_normal.append(mem_normal.item())
    
    # --- SCENARIO 2: HACKED (Raw signal hits the neuron) ---
    _, mem_hacked = lif(torch.tensor([raw_signal]), mem_hacked)
    record_hacked.append(mem_hacked.item())
    
    # --- SCENARIO 3: BALAM DEFENSE (Firmware active) ---
    # The shield clamps the malicious 5.0v down to a safe 1.0v max
    shielded_signal = torch.clamp(torch.tensor([raw_signal]), max=1.0)
    _, mem_defended = lif(shielded_signal, mem_defended)
    record_defended.append(mem_defended.item())

# 3. Drawing the Graph (matplotlib)
plt.figure(figsize=(10, 6))

# Plot the three timelines
plt.plot(record_normal, label="Normal Operation", color="blue", linestyle="--")
plt.plot(record_hacked, label="Hacked (Digital Seizure)", color="red", linewidth=2)
plt.plot(record_defended, label="Balam Shield Active", color="green", linewidth=2)

# Draw a line showing the Spike Threshold (1.0)
plt.axhline(y=1.0, color='black', linestyle=':', label="Spike Threshold")

# Add text and labels
plt.title("Balam Protocol: Membrane Potential under EMFI Attack", fontsize=14)
plt.xlabel("Time (Milliseconds)", fontsize=12)
plt.ylabel("Membrane Voltage", fontsize=12)
plt.legend()
plt.grid(True)

# Save the picture!
plt.savefig("balam_defense_plot.png")
print("[SYSTEM] Graph successfully saved as 'balam_defense_plot.png'!")
