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