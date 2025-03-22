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

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(12, 28, 1)], # different values of x_names to benchmark
        x_log = True, # makes x-axis logarithmic
        line_arg='provider', # title of the legend 
        line_vals=['triton', 'torch'], # designators of the different entries in the legend
        line_names=['Triton', 'Torch'], # names to visibly go in the legend
        styles=[('blue', '-'), ('green', '-')], # triton will be blue; pytorch will be green
        ylabel='GB/s', # label name for y-axis
        plot_name='vector-subtract-performance', # also used as file name for saving plot
        args={}, # we'll see how this is used in a later tutorial; need it even if it's empty
    )
)

def benchmark(size, provider):
    # creating our input data
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    # each benchmark runs multiple times and quantiles tells matplotlib what confidence intervals to plot
    quantiles = [0.5, 0.05, 0.95]
    # defining which function this benchmark instance runs
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x - y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vector_subtract(x, y), quantiles=quantiles)
    # turning the raw millisecond measurement into meaninful units
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        # 3 = number of memory operations (2 reads + 1 write)
        # x.numel() = number of elements
        # x.element_size() = bytes per element (4 for float32, 2 for float16)
        # 1e-9 converts bytes to GB
        # 1e-3 converts milliseconds to seconds
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # always run unit-tests
    # test_add_kernel(size=98432)

    x = torch.rand(1028, device='cuda')
    y = torch.rand(1028, device='cuda')


    result = vector_subtract(x, y)

    torch_result = x - y
    assert torch.allclose(result, torch_result)

    print(result)

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)

