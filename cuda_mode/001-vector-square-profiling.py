import torch 

a = torch.tensor([1., 2., 3.])

print(torch.square(a))
print(a * a)
print(a ** 2)

def time_pytorch_func(func, input):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(10):
        func(input)
    
    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

b = torch.randn(10000, 10000).cuda()

def square_2(a):
    return a * a

def square_3(a):
    return a ** 2

time_pytorch_func(torch.square, b)
time_pytorch_func(square_2, b)
time_pytorch_func(square_3, b)


print("=============")
print("Profiling torch.square")
print("=============")

# Now profile each function using pytorch profiler
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a * a")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a ** 2")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


# =============
# Profiling torch.square
# =============
# /home/ubuntu/dev/gpu/cuda_mode/001-vector-square-profiling.py:40: FutureWarning: The attribute `use_cuda` will be deprecated soon, please use ``use_device = 'cuda'`` instead.
#   with torch.autograd.profiler.profile(use_cuda=True) as prof:
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#              aten::square        17.60%     947.326us        40.38%       2.173ms       2.173ms     949.000us        17.41%       5.450ms       5.450ms             1  
#                 aten::pow        22.16%       1.193ms        22.68%       1.221ms       1.221ms       4.489ms        82.37%       4.501ms       4.501ms             1  
#                  aten::to         0.02%       1.293us         0.02%       1.293us       1.293us       7.000us         0.13%       7.000us       7.000us             1  
#         aten::result_type         0.04%       2.145us         0.04%       2.145us       2.145us       5.000us         0.09%       5.000us       5.000us             1  
#           cudaEventRecord         0.41%      21.829us         0.41%      21.829us       2.729us       0.000us         0.00%       0.000us       0.000us             8  
#          cudaLaunchKernel         0.34%      18.087us         0.34%      18.087us      18.087us       0.000us         0.00%       0.000us       0.000us             1  
#     cudaDeviceSynchronize        59.44%       3.200ms        59.44%       3.200ms       3.200ms       0.000us         0.00%       0.000us       0.000us             1  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 5.383ms
# Self CUDA time total: 5.450ms

# =============
# Profiling a * a
# =============
# /home/ubuntu/dev/gpu/cuda_mode/001-vector-square-profiling.py:49: FutureWarning: The attribute `use_cuda` will be deprecated soon, please use ``use_device = 'cuda'`` instead.
#   with torch.autograd.profiler.profile(use_cuda=True) as prof:
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                 aten::mul         1.42%      46.749us         1.99%      65.749us      65.749us       3.355ms       100.00%       3.355ms       3.355ms             1  
#           cudaEventRecord         0.30%       9.940us         0.30%       9.940us       4.970us       0.000us         0.00%       0.000us       0.000us             2  
#          cudaLaunchKernel         0.58%      19.000us         0.58%      19.000us      19.000us       0.000us         0.00%       0.000us       0.000us             1  
#     cudaDeviceSynchronize        97.71%       3.227ms        97.71%       3.227ms       3.227ms       0.000us         0.00%       0.000us       0.000us             1  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 3.303ms
# Self CUDA time total: 3.355ms

# =============
# Profiling a ** 2
# =============
# /home/ubuntu/dev/gpu/cuda_mode/001-vector-square-profiling.py:58: FutureWarning: The attribute `use_cuda` will be deprecated soon, please use ``use_device = 'cuda'`` instead.
#   with torch.autograd.profiler.profile(use_cuda=True) as prof:
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                 aten::pow         1.53%      50.962us         2.22%      73.645us      73.645us       3.347ms        99.73%       3.356ms       3.356ms             1  
#         aten::result_type         0.06%       1.917us         0.06%       1.917us       1.917us       5.000us         0.15%       5.000us       5.000us             1  
#                  aten::to         0.03%       0.870us         0.03%       0.870us       0.870us       4.000us         0.12%       4.000us       4.000us             1  
#           cudaEventRecord         0.44%      14.611us         0.44%      14.611us       2.435us       0.000us         0.00%       0.000us       0.000us             6  
#          cudaLaunchKernel         0.41%      13.732us         0.41%      13.732us      13.732us       0.000us         0.00%       0.000us       0.000us             1  
#     cudaDeviceSynchronize        97.53%       3.239ms        97.53%       3.239ms       3.239ms       0.000us         0.00%       0.000us       0.000us             1  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 3.321ms
# Self CUDA time total: 3.356ms