import torch
import triton 
import triton.language as tl 
DEVICE = "cuda"





autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]

@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_c_M, stride_c_N, 
    
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, 
):
    PID = tl.program_id(axis=0) 
    
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M) 
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N) 
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    
    group_id = PID // num_PID_in_group 
    
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
        
    PID_N = (PID % num_PID_in_group) // group_size_adj
    
    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)
    
    a_offsets = offsets_M[:, None] * stride_a_M + offsets_K[None, :] * stride_a_K
    b_offsets = offsets_K[:, None] * stride_b_K + offsets_N[None, :] * stride_b_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) 
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        
        mask = offsets_K < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0) 
        b = tl.load(b_ptr + b_offsets, mask=mask[:, None], other=0.0) 
        accumulator = tl.dot(a, b, acc=accumulator)
        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K
    
    accumulator = accumulator.to(tl.float16)
    c_offsets = stride_c_M * offsets_M[:, None] + stride_c_N * offsets_N[None, :]
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N) 
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask) 


def matmul(a, b):
    # checking constraints. 
    # 1. to check if both are matrices and not vectors or tensors. 
    # 2. if matmul compatible dimensions or not 
    assert a.ndim == b.ndim == 2, "input not matrix."
    assert a.shape[1] == b.shape[0], "incompatible matmul dimensions."
    a,b  = a.to(torch.float16), b.to(torch.float16)

    (M, K), (_, N) = a.shape, b.shape

    # allocating output
    c = torch.empty((M, N), device = a.device, dtype = torch.float16)

    # grid launch 
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c