# Balam: Neuromorphic Runtime Security 🛡️

# The Problem
As computing shifts toward Brain-Computer Interfaces (BCIs) and Spiking Neural Networks (SNNs), a catastrophic blind spot has emerged. Traditional cybersecurity is blind to biological, event-driven spikes. Neuromorphic hardware (medical implants, defense drones) is highly vulnerable to physical Side-Channel Attacks and Electromagnetic Fault Injection (EMFI). 

# The Solution (Balam PoC)
This repository contains the foundational R&D for a hardware-agnostic runtime defense firmware. It mathematically filters malicious analog voltage glitches before they can corrupt the Spiking Neural Network.

# Architecture Demonstrated:
- **Framework:** PyTorch & snnTorch
- **Network:** Leaky Integrate-and-Fire (LIF) neurons
- **Attack Vector:** Simulated 5.0v EMFI injection causing catastrophic digital seizure.
- **Defense Mechanism:** Real-time temporal membrane thresholding (Balam Protocol).

Project Balam is currently in the pre-seed software R&D phase.
