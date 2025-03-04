import torch 

import triton 
import triton.language as tl 

# DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEVICE = torch.device('cuda:0')

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
    # print(x.device, y.device, output.device)
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

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names = ['size'],                             # names to be used on the x axis
        x_vals = [2 ** i for i in range(12, 30, 1)],    # values to be put on the x axis
        x_log = True,                                   # log axis
        line_arg = 'provider',                          # argument whose value will correspond to a different line in the plot
        line_vals = ['triton', 'torch'],                # possible values for the argument 
        line_names = ['Triton', 'Torch'],               # names for the lines
        styles = [('green', '-'), ('orange', '-')],     # style
        ylabel = 'GB/s',                                # label for y axis
        plot_name = 'vector addition performance',      # plot name
        args = {}                                       # values for function arguments not in x axis or y axis
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device = DEVICE, dtype = torch.float32)
    y = torch.rand(size, device = DEVICE, dtype = torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles = quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles = quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98304

    x = torch.rand(size, device = DEVICE)
    y = torch.rand(size, device = DEVICE)

    output_torch = x + y
    output_triton = add(x, y)

    print(output_torch)
    print(output_triton)

    print(f"maximum diff b/w torch and triton is: {torch.max(torch.abs(output_torch - output_triton))}")

    benchmark.run(print_data = True, save_path = '../plots/')