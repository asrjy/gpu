import torch 
import triton 
import triton.language as tl 
DEVICE = "cuda"

def naive_softmax(x):
    """
        safe softmax: subtracting max element from all values to avoid numerical overflows from .exp()
    """
    x_max = x.max(dim = 1)[0]
    z = x - x_max[:, None]

    numerator = torch.exp(z)
    denominator = numerator.sum(dim = 1)
    out = numerator/denominator[:, None]

    return out


@triton.jit
def softmax_kernel(
    x_ptr,                      # input pointer
    y_ptr,                      # output pointer
    x_stride,                   # number of elements to skip while moving to next row
    y_stride,                   # number of elements to skip while moving to next row
    n_rows,                     # matrix dimension - x axis
    n_cols,                     # matrix dimension - y axis
    BLOCK_SIZE: tl.constexpr,   # lowest power of 2 thats greater than n_cols
    num_stages: tl.constexpr,
    ):
    """
    the naive implementation has too many memory accesses. 
    we want to read memory only once from the DRAM and does all computations on the GPU
    this is the "fused" part. 

    torch.jit.script and torch.compile can do this to an extent but the custom implementation will do it better.

    each thread will load a strided set of rows, and writes the softmax value to the output
    """
    # each propgram will handle row_step number of rows
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages = num_stages):
        # calculates pointer to current row
        row_start_ptr = x_ptr + row_idx * x_stride
        # creates column offsets: 0 to BLOCK_SIZE -1
        col_offsets = tl.arange(0, BLOCK_SIZE) 
        input_ptrs = row_start_ptr + col_offsets
        # mask handles non power of 2 row lengths
        mask = col_offsets < n_cols
        # uses -inf for masked values
        row = tl.load(input_ptrs, mask=mask, other=float('-inf')) 
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # write output back to DRAM
        output_row_start_ptr = y_ptr + row_idx * y_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


# fetching specifications of gpu before kernel call 
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
# number of streaming multiprocessors
NUM_SM = properties["multiprocessor_count"] 
# number of registers, the fastest memory on the gpu d
NUM_REGS = properties["max_num_regs"] 
# each sm has it's own sram. since each sm will have multiple programs within itself, we can divide this sram within them. 
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] 
# warp is a group of threads that execute together. 
WARP_SIZE = properties["warpSize"]


def softmax(x):
    """
    this will allocate space for the output tensor and enque the kernel call with the appropriate block and grid sizes.
    this is not connected to pytorch graph, meaning it doesnt support backprop. 
    """
    # Ensure input is a 2D tensor
    assert x.ndim == 2
    n_rows, n_cols = x.shape

    # Choose block size as the next power of 2 of the number of columns
    # This ensures efficient memory access patterns
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Set number of warps based on the block size
    # Larger blocks need more warps for better parallelism
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    # Choose number of pipeline stages based on available SRAM
    # More stages can improve performance by hiding memory latency
    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2
    
    # Allocate output tensor with same shape and dtype as input
    y = torch.empty_like(x)
    
    # Warmup the kernel with a dummy run to compile it
    # This also helps with performance by caching the compiled kernel
    kernel = softmax_kernel.warmup(x, y, 
                                    x.stride(0), y.stride(0), 
                                    n_rows, n_cols,
                                    BLOCK_SIZE=BLOCK_SIZE,
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                    grid=(1,))
    
    # Initialize kernel handles to access metadata
    kernel._init_handles()
    
    # Calculate resource usage to determine optimal grid size
    n_regs = kernel.n_regs  # Number of registers used per thread
    sram_needed_per_program = kernel.metadata.shared  # Shared memory needed per program
    
    # Calculate occupancy based on register and SRAM limitations
    reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)  # How many programs fit based on register usage
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program  # How many programs fit based on SRAM usage
    
    # Take the minimum of both constraints to determine programs per SM
    programs_per_sm = min(reg_occupancy, sram_occupancy)
    
    # Calculate total number of programs to launch, capped by input size
    num_programs = min(NUM_SM * programs_per_sm, n_rows)
    
    # Set grid dimensions for kernel launch
    grid = (num_programs, 1, 1)
    
    # Launch the kernel with the calculated grid size
    kernel[grid](
        x, y,  # Input and output tensors
        x.stride(0), y.stride(0),  # Strides for efficient memory access
        n_rows, n_cols,  # Dimensions of the input
    )
    
    return y