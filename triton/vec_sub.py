import torch 
import triton 
import triton.language as tl 

@triton.jit
def vector_subtract_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis = 0)

    offset = pid * BLOCK_SIZE
    # tl.device_print("offset: ", offset)

    mask = offset + tl.arange(0, BLOCK_SIZE) < n_elements 

    # x = tl.load(x_ptr + offset, mask = mask)
    # y = tl.load(y_ptr + offset, mask = mask)

    x = tl.load(x_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=mask)
    y = tl.load(y_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=mask)

    output = x - y

    # tl.store(output_ptr + offset, output, mask = mask)
    tl.store(output_ptr + offset + tl.arange(0, BLOCK_SIZE), output, mask=mask)

def vector_subtract(x, y):
    assert x.shape == y.shape, "inputs not equal sizes"
    assert x.is_cuda and y.is_cuda, "inputs not on GPU"

    n_elements = x.numel()
    output = torch.empty_like(x)

    BLOCK_SIZE = 1024
    # grid = (triton.cdiv(n_elements, BLOCK_SIZE))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)  # Explicitly set a tuple with at

    vector_subtract_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)

    return output

x = torch.rand(1028, device='cuda')
y = torch.rand(1028, device='cuda')


result = vector_subtract(x, y)

torch_result = x - y
assert torch.allclose(result, torch_result)

print(result)
