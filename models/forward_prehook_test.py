import torch

def print_tensor(module, input, output):
    print(f'Input: {input}')
    print(f'Output: {output}')

# create a linear layer
linear = torch.nn.Linear(8, 5)

# register the forward hook
linear.register_forward_hook(print_tensor)

# generate some input data
input = torch.randn(5, 8)

# apply the forward pass
output = linear(input)
