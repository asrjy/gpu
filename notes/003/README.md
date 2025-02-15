- multiply + add operation is usually called fused multiply add (FMA)
- a warp scheduler is the physical core where instructions are executed within a warp. 
- there are four warp schedulers per multiprocessor. 
- profiling refers to analyzing cuda runtimes, memory access patterns, opengl calls, kernel execution times etc., 
- nvprof (nvidia visual profiler) is a command line profiler that is now considered deprecated. 
- nsys (nvidia nsight systems) is a system wide analysis tool and includes both cpu and gpu activity. recommended for modern cuda profiling. 
- profiling is used to identify bottlenecks and areas for improvement. essential for performance tuning. 
- key metrics to monitor
    - kernel execution time: duration kernel took to execute on the gpu
    - memory transfers bw host and device: need to minimize as much as possible
    - occupancy of SMs: how "occupied" are the streaming multiprocessors in the gpu. sometimes higher occupany leads to better perf. 
    - memory access patterns (coalescing): how efficiently threads in a warp are accessing the global memory. coalesced memory access are where threads access memory that are in contiguous locations. these are much faster than uncoalesced memory accesses. 
    - instruction stats: types and number of instructions executed by the kernel. branch divergence is where threads in a warp take different execution paths. these can reduce performance. 
- `-g` flag during compilation allows profiler to show source code line numbers in the reports, easier to pinpoint bottlenecks. 
- `nvprof ./cuda_executable` will print summary of kernel execution times and memory transfer to console. 
- we can save the output of the profiler to a file using `nvprof -o output.prof ./cuda_executable`
- we can use nvidia visual profiler, `nvvp`  to view the contents of the profile file created. 
- `nsys profile -o output ./cuda_executable` profiles code with nsys. output is stored in `output.nsys-rep` file. 
- `-t cuda, nvtx, orst` argument specifies that cuda runtime, nvtx annotations and os runtime tracers are to be used. 
- `-s push, trace, cuda` specifies how to format the data. 
- `--stats=true` provides terminal output, instead of having to rely only on the gui. 

- a cycle refers to the smallest unit of time in which operations can be executed ie., the fastest op can take 1 cycle. 
- if a gpu runs at 1ghz, then 1 cycle = 1 nanosecond. if a memory access in that gpu takes 400 cycles, it means it takes 400 nano seconds. 
- registers are the fastest (1 cycle), shared memory second fastest (1-2 cycles provided no bank conflicts), global memory (400 - 600 cycles, frequent access should be avoided)
- a bank conflict is when multiple threads try to access the same memory bank in shared memory at the same time, causing serialization and slowing things down. 
- typically there are 32 banks per streaming multiprocessor. 

how l1 and l2 work: 
- cpu/gpu requests a data. 
- core first checks if data is in l1 cache. if it's a hit, l1 gets to it. if it's a miss, proceeds to l2. 
- if data is found in l2, core gets it. slower than l1, but faster than from main memory. if it's a miss, proceed to main memory. 
- core retrieves from main memory. 
- data is copied into both l1 and l2 caches, or sometimes just l2 depending on cache policy. 

- cache hits are critical for performance. so writing cache friendly code is important. 

cache friendly techniques:
- data locality: organize data structures and access patterns so data that is used together is stored closer in memory. so when one piece of data is loaded, related data will also be bought into cache (example: accessing array elements)
- temporal locality: reuse data that has been loaded into cache as much as possible before it gets evicted. (example: using same variable multiple times within code)
- blocking/tiling: divide problem into smaller blocks that fit in the cache. process each block completely and be done with it before moving to the next one. 
- avoid strided access: avoid striding where we skip over multiple elements. try to access contiguosly. 


- `__shfl_sync` is a warp-level shuffle intrinsic. 
- it allows threads within same warp to exchange data and synchronize within the warp. 