import torch 

import triton 
import triton.language as tl 

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit 
def add_kernel( x_ptr, # pointer to first vector,
                y_ptr, # pointer to second vector,
                output_ptr, # pointer to output vector
                n_elements, # number of elements 
                BLOCK_SIZE: tl.constexpr, # number of elements each program should process. in order to be used as a shape value constexpr is used
                ):
    # multiple 'programs' access different parts of the data. we identify which program is accessing
    pid = tl.program_id(axis = 0)
    # if block size is 64, 0:64 is accessed by one pid, 65:128  by another pid ... we offset based on pid value
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # creating a mask to guard memory access operations against out of bound accesses
    mask = offsets < n_elements 

    # loading x and y from DRAM masking out additional accesses in case input shapes are not perfect multiples of block size 
    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    output = x + y

    # write output back to dram 
    tl.store(output_ptr + offsets, output, mask = mask)


# helper function to allocate z tensor and enqueue the kernel with appropriate grid block and sizes 
def add(x: torch.Tensor, y:torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

    n_elements = output.numel()

    # similar to kernel call parameters in cuda, the spmd launch grid contains the 
    # number of kernel instances that run in parallel. 
    # it can be a Tuple[int] 
    # using a 1d grid which has a size of number of blocks 
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # each torch object is implicitly converted into a pointer to its first element 
    # we need to pass meta parameters as keyword arguments 
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE = 1024)
    return output 

torch.manual_seed(0)
size = 98304

x = torch.rand(size, device = DEVICE)
y = torch.rand(size, device = DEVICE)

output_torch = x + y
output_triton = add(x, y)

print(output_torch)
print(output_triton)

print(f"maximum diff b/w torch and triton is: {torch.max(torch.abs(output_torch - output_triton))}")