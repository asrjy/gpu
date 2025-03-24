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
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)