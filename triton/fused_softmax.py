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
# number of registers, the fastest memory on the gpu 
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
    assert x.ndim == 2
    n_rows, n_cols = x.shape
