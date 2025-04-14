import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,  # dropout probability
    seed,  # random seed
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values where mask is True
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Generate random values
    random = tl.rand(seed, offsets)
    
    # Create dropout mask (True for values to keep)
    x_keep = random > p
    
    # Apply dropout - scale kept values by 1/(1-p)
    output = tl.where(x_keep, x / (1.0 - p), 0.0)
    
    # Store results
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous(), "Input tensor must be contiguous"
    n_elements = x.numel()
    
    # Calculate grid dimensions
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the Triton kernel
    _seeded_dropout[grid](
        x_ptr=x, 
        output_ptr=output, 
        n_elements=n_elements, 
        p=p, 
        seed=seed, 
        BLOCK_SIZE=1024
    )
    return output


x = torch.randn(size=(8,), device=DEVICE)
output1 = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)

print("Input:", x)
print("Output (seed=123):", output1)
print("Output (seed=123, should match):", output2)
print("Output (seed=512, should differ):", output3)
print("\nMatching outputs?", torch.allclose(output1, output2))
