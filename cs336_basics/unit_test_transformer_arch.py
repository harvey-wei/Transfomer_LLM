import torch
import torch.nn.init as init


# This code has a few issues:
# 1. Parameters should have requires_grad=True by default for training
# 2. Using add_() modifies tensors in-place which can cause issues with autograd
# 3. No assertions or actual unit tests

# Create parameters properly
w1 = torch.nn.Parameter(torch.zeros(10, 10))
w2 = torch.nn.Parameter(torch.zeros(10, 10)) 
w3 = torch.nn.Parameter(torch.zeros(10, 10))

print("Before initialization:")
for weight in [w1, w2, w3]:
    print(weight)

# Initialize directly with trunc_normal_
for weight in [w1, w2, w3]:
    init.trunc_normal_(weight, mean=0.0, std=1.0, a=-2.0, b=2.0)

print("\nAfter trunc_normal_ initialization:")
for weight in [w1, w2, w3]:
    print(weight)

# The parameters maintain their requires_grad property
print("\nrequires_grad status:")
for i, weight in enumerate([w1, w2, w3]):
    print(f"w{i+1} requires_grad:", weight.requires_grad)



