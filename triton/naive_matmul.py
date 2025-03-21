import torch 
import triton 
import triton.language as tl 


@triton.jit
def naive_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N
    m_range = m_offset + tl.arange(0, BLOCK_M)
    n_range = n_offset + tl.arange(0, BLOCK_N)
    m_mask = m_range < M
    n_mask = n_range < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K) * BLOCK_K, BLOCK_K):
        k_range = k + tl.arange(0, BLOCK_K)
        k_mask = k_range < K
        
        a_block_ptr = a_ptr + (m_range[:, None] * stride_am + k_range[None, :] * stride_ak)
        b_block_ptr = b_ptr + (k_range[:, None] * stride_bk + n_range[None, :] * stride_bn)
        
        a = tl.load(a_block_ptr, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_block_ptr, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        acc += tl.dot(a, b)
    
    acc = acc.to(tl.float16)
    
    c_block_ptr = c_ptr + (m_range[:, None] * stride_cm + n_range[None, :] * stride_cn)
    tl.store(c_block_ptr, acc, mask=m_mask[:, None] & n_mask[None, :])


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_cuda and b.is_cuda, "Inputs must be on GPU"
    
    M, K = a.shape
    K, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    naive_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),  
        b.stride(0), b.stride(1),  
        c.stride(0), c.stride(1),  
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return c



if __name__ == "__main__":
    a = torch.randn(128, 256, device='cuda', dtype=torch.float16)
    b = torch.randn(256, 128, device='cuda', dtype=torch.float16)

    c = matmul(a, b)
    torch_c = torch.matmul(a, b)

    assert torch.allclose(c, torch_c, rtol=1e-3, atol=1e-3)
    print("results match")