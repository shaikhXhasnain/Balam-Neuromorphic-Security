import torch

x = torch.tensor([2.0], requires_grad=True)
print("---1. The Starting Tensor --- ")
print("x:", x)

y = 3 * (x ** 2)
print("\n---2. The Arrows Landing Spot ---")
print("y = 3 * x^2")
print("y:",y)

y.backward()

print("\n---3. The Correction Gradiant ---")
print("How Much To Adjust x(x.grad):",x.grad)
