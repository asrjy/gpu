import torch 
import triton 
import triton.language as tl 


@triton.jit
def naive_matmul_kernel(
    a_ptr, b_ptr, c_ptr, # pointers to matrices
    M, N, K, # matrix dimensions
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N
    
    m_range = m_offset + tl.arange(0, BLOCK_M)
    n_range = n_offset + tl.arange(0, BLOCK_N)

    m_mask = m_range < N
    n_mask = n_range < N

    # initialize accumulator to zeros 
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K) * BLOCK_K, BLOCK_K):
        k_range = k + tl.arange(0, BLOCK_K)
        k_mask = k_range < K
        
        a_block_ptr = a_ptr + m_range[:, None] * K + k_range[None, :]
        b_block_ptr = b_ptr + k_range[:, None] * N + n_range[None, :]
        
        a = tl.load(a_block_ptr, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_block_ptr, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # matmul
        acc += tl.dot(a, b)

        c_block_ptr = c_ptr + m_range[:, None] * N + n_range[None, :]
        tl.store(c_block_ptr, acc, mask=m_mask[:, None] & n_mask[None, :])


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_cuda and b.is_cuda, "Inputs must be on GPU"
    
    M, K = a.shape
    K, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    naive_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return c


a = torch.randn(128, 256, device='cuda')
b = torch.randn(256, 128, device='cuda')

c = matmul(a, b)

torch_c = torch.matmul(a, b)
assert torch.allclose(c, torch_c, rtol=1e-3, atol=1e-3)