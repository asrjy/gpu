## 100 days of GPU Programming!

### Inspiration: 
- [100 days of GPU](https://github.com/hkproj/100-days-of-gpu)
- 🤑

### Progress

|**Day**|**Code**|**Notes**|**Progress**|
|---|----|-----|--------|
|057|[triton: fused softmax](./triton/fused_softmax.py)|[triton](./notes/005/)|gpu specs and kernel call wrapper|
|056|[triton: fused softmax](./triton/fused_softmax.py)|[triton](./notes/005/)|understanding fused implementation|
|055|[triton: fused softmax](./triton/fused_softmax.py)|[triton](./notes/005/)|fused implementation|
|054|[triton: fused softmax](./triton/fused_softmax.py)|[triton](./notes/005/)|ideating fused implementation|
|053|[triton: fused softmax](./triton/fused_softmax.py)|[triton](./notes/005/)|naive implementation|
|052|[triton: vector subtract benchmarking](./triton/vec_sub.py)|[triton](./notes/005/)|added benchmarks for vector subtract|
|051|[triton: naive matmul for turing](./triton/naive_matmul.py)|[triton](./notes/003/)|fixing naive matrix multiplication for turing gpu|
|050|[triton: naive matmul](./triton/naive_matmul.py)|[triton](./notes/003/)|naive matrix multiplication|
|049|[triton: vector subtraction bug fix](./triton/vec_sub.py)|[triton](./notes/003/)|fixing bugs in vec sub|
|048|[triton: vector subtraction](./triton/vec_sub.py)|[triton](./notes/003/)|touching base with basics|
|047|[triton puzzles: flash attention](./triton/puzzles.ipynb)|[triton](./notes/003/)|started flash attention|
|046|[triton puzzles: long softmax v2](./triton/puzzles.ipynb)|[triton](./notes/003/)|softmax on logits, v2|
|045|[triton puzzles: long softmax](./triton/puzzles.ipynb)|[triton](./notes/003/)|softmax on logits|
|044|[triton puzzles: long sum](./triton/puzzles.ipynb)|[triton](./notes/003/)|sum of batch of numbers|
|043|[triton puzzles: matmul + relu](./triton/puzzles.ipynb)|[triton](./notes/003/)|matrix multiplication and relu|
|042|[triton puzzles: fused matmul + relu](./triton/puzzles.ipynb)|[triton](./notes/003/)|fused matrix multiplication and relu|
|041|[triton puzzles: vector addition, row to col](./triton/puzzles.ipynb)|[triton](./notes/003/)|vector addition row and column vectors|
|040|[triton puzzles: vector addition](./triton/puzzles.ipynb)|[triton](./notes/003/)|vector addition|
|039|[triton puzzles: constant addition with varying block sizes](./triton/puzzles.ipynb)|[triton](./notes/003/)|constant addition to vector|
|038|[triton puzzles: constant addition](./triton/puzzles.ipynb)|[triton](./notes/003/)|constant addition|
|037|[triton puzzles: blocks and loading](./triton/puzzles.ipynb)|[triton](./notes/003/)|2d tensor loading as blocks|
|036|[triton puzzles: loading 2d tensors](./triton/puzzles.ipynb)|[triton](./notes/003/)|2d tensor and tl.store|
|035|[triton puzzles](./triton/puzzles.ipynb)|[triton](./notes/003/)|tritonviz + debugging meson + puzzles environment setup|
|034|[benchmarking in triton](./triton/vec_add.py)|[triton](./notes/003/)|triton benchmarking + plots|
|033|[vector addition in triton](./triton/vec_add.py)|[triton](./notes/003/)|triton setup + vector addition|
|032|[knn with vectorized distance computation](./cuda/knnBase.cu)|[float4](./notes/003/)|knn + vectorized distance computation + float4 operations|
|031|[knn with batch distance computation](./cuda/knnBase.cu)|[knn](./notes/003/)|knn + batch distance computation|
|030|[knn with thrust for sorting](./cuda/knnBase.cu)|[knn](./notes/003/)|knn + thrust sorting|
|029|[knn with tiled distance computation](./cuda/knnBase.cu)|[knn](./notes/003/)|knn + tiling|
|028|[knn with shared memory distance calculation](./cuda/knnBase.cu)|[knn](./notes/003/)|knn + shared memory|
|027|[baseline gpu knn](./cuda/knnBase.cu)|[knn](./notes/003/)|knn|
|026|[bitonic sort with shared memory](./cuda/bitonicSortSharedMemory.cu)|[sorting](./notes/003/)|bitonic sort with shared memory|
|025|[bitonic sort ](./cuda/bitonicSort.cu)|[sorting](./notes/003/)|bitonic sort|
|024|[histogram with shared memory and atomic add ](./cuda/histogramSharedMemory.cu)|[using shared memory and atomic adds](./notes/003/)|atomic operations, race conditions|
|023|[histogram with atomicadds ](./cuda/histogramAtomicAdd.cu)|[using atomic adds](./notes/003/)|atomic operations, race conditions|
|022|[register pressure and spilling ](./cuda/registerPressureSpilling.cu)|[reducing register pressure](./notes/003/)|spilling, high and low register pressure|
|021|[optimizing warp divergence ](./cuda/warpDivergence.cu)|[optimized warp divergence](./notes/003/)|warp divergence and optimzing for it|
|020|[hillis steele prefix sum (optimized) ](./cuda/prefixSumHillisSteele.cu)|[optimized parallel prefix sum](./notes/003/)|hillis steele with shared memory|
|019|[prefix sum (naive)](./cuda/prefixSumNaive.cu)|[parallel prefix sum](./notes/003/)|prefix sum, parallel scanning|
|018|[parallel reduction (optimized)](./cuda/parallelReductionOptimized.cu)|[optimized parallel reduction](./notes/003/)|shuffle sync with mask and warps|
|017|[parallel reduction (naive)](./cuda/parallelReductionNaive.cu)|[naive parallel reduction](./notes/003/)|parallel reduction with shared memory|
|016|[l1 and l2 cache](./cuda/cacheAccess.cu)|[read about l1, l2 cache and how to write cache friendly code](./notes/003/)|l1, l2 cache|
|015|[matrix multiplication with block tiling](./cuda/matrixMultiplicationBlockTiling.cu)|[optimizing mat mul using block tiling](./notes/003/)|block tiling|
|014|[matrix multiplication sgemm shared memory](./cuda/matrixMultiplicationMemoryBlocking.cu)|[optimizing mat mul using memory blocking](./notes/003/)|shared memory, memory blocking|
|013|[optimizing matrix multiplication](./cuda/matrixMultiplication.cu)|[optimizing mat mul using coalescing](./notes/003/)|coalescing memory and warp scheduling|
|012|[shared memory](./cuda/sharedMemory.cu)|[matrix multiplication and shared memory](./notes/003/)|read about shared memory, registers and warps, bank conflicts, reading matrix multiplication blog by siboehm|
|011|[optimizing matrix multiplication](./cuda/matrixMultiplication.cu)|[matrix multiplication and profiling](./notes/003/)|using nsys and nvprof, reading matrix multiplication blog by siboehm|
|010|[face blur](./cuda/objectBlur/)|[read matrix multiplication blog](./notes/002/)|reading matrix multiplication blog by siboehm, using a compiled kernel in python|
|009|[matrix transpose](./cuda/matrixTranspose.cu)|[matrix transpose and matrix multiplication blog](./notes/002/)|started reading matrix multiplication blog by siboehm, started chapter 4 of PMPP|
|008|[matrix multiply](./cuda/matrixMultiplication.cu) and [helpers](./cuda/helpers.h)|[matrix multiplication, pinned memory and BLAS](./notes/002/)|read about pinned memory, pageable memory and cudaHostAlloc(). finished chapter 3 of PMPP|
|007|[vector multiply](./cuda/vecMultiply.cu) and [helpers](./cuda/helpers.h)|[internal structure of blocks](./notes/002/)|setup gpu env on new server. studied heirarchy of execution within the streaming multiprocessor. created helpers file.|
|006|[gaussianBlurSharedMemory with event times](./cuda/gaussianBlurSharedMemory.cu)|[event times and performance measurement](./notes/002/)|added perf measurement code to gaussian blur with shared memory kernel|
|005|[gaussianBlurSharedMemory](./cuda/gaussianBlurSharedMemory.cu)|[PMPP Chapter 3 & exploration](./notes/002/)|built on top of gaussian blur; learnt about shared memory and implemented it;|
|004|[gaussianBlur](./cuda/gaussianBlur.cu)|[PMPP Chapter 3](./notes/002/)|built on top of image blur; struggling to understand multidimensionality;|
|003|[imageBlur](./cuda/imageBlur.cu)|[PMPP Chapter 3](./notes/002/)|read parts of image blur and about better ways to handle errors, image blurring logic|
|002|[colorToGrayScaleConversion](./cuda/colorToGrayscaleConversion.cu)|[PMPP Chapter 3](./notes/002/)|read half of chapter 2 of pmpp, implemented color to grayscale conversion|
|001|[vecAbsDiff](./cuda/vecAbsDiff.cu)|[PMPP Chapter 2](./notes/001/)|read chapter 2 of pmpp, implemented vector absolute difference kernel|
|000|-|[PMPP](./notes/000/PMPP-Ch1.pdf)|setup environment, lecture 1 of ECE 408, chapter 1 of PMPP|



### Resources:
- Programming Massively Parallel Processors
- [CUDA 120 Days Challenge](https://github.com/AdepojuJeremy/Cuda-120-Days-Challenge)
- [ECE 408](https://www.youtube.com/playlist?list=PL6RdenZrxrw-UKfRL5smPfFFpeqwN3Dsz)
- LLMs


### The Plan:


| **Objective**  | **Topic**                                      | **Task/Implementation** |**Status**|
|----------|----------------------------------------------|-------------------------|-|
| **Phase 1: Foundations** | **Goal:** Understand CUDA fundamentals, memory hierarchy, and write basic optimized kernels. |||
| **1**  | CUDA Setup & First Kernel  | Install CUDA, write a vector addition kernel |✅|
| **2**  | Thread Hierarchy | Grids, blocks, threads, experimenting with configurations |✅|
| **3**  | Memory Model Basics | Global, shared, local memory overview |✅|
| **4**  | Memory Coalescing | Optimize vector addition using shared memory |✅|
| **5**  | Matrix Multiplication (Naïve) | Implement basic matrix multiplication |✅|
| **6**  | Matrix Multiplication (Optimized) | Use shared memory to optimize ||
| **7**  | Profiling Basics | Use `nvprof` and `nsys` to analyze kernels |✅|
| **8**  | L1/L2 Cache Effects | Study cache behavior and memory bandwidth |✅|
| **9**  | Tiled Matrix Multiplication | Further optimize matrix multiplication ||
| **10** | Register Pressure | Optimize register usage and reduce spilling |✅|
| **11** | Warp Execution Model | Avoiding warp divergence |✅|
| **12** | Parallel Reduction (Naïve) | Implement sum/max reductions |✅|
| **13** | Parallel Reduction (Optimized) | Optimize with warp shuffle (`__shfl_sync`) |✅|
| **14** | Code Review & Optimization | Refine and benchmark previous work ||
| **15** | Parallel Scan (Prefix Sum) | Implement parallel scan algorithm |✅|
| **16** | Histogram (Naïve) | Implement histogram using global memory atomics |✅|
| **17** | Histogram (Optimized) | Use shared memory to optimize histogram |✅|
| **18** | Parallel Sorting | Implement bitonic or bucket sort |✅|
| **19** | k-Nearest Neighbors | Implement kNN search using CUDA ||
| **20** | Code Review & Benchmarking | Optimize and compare previous implementations ||
| **Phase 2: ML Operators** | **Goal:** Implement and optimize core ML kernels. |||
| **21** | Dense Matrix-Vector Multiplication | Implement `y = Wx + b` in CUDA ||
| **22** | Fully Connected Layer | Implement dense forward pass ||
| **23** | ReLU & Softmax | Implement activation functions ||
| **24** | Backpropagation | Implement BP for a single layer ||
| **25** | 1D Convolution (Naïve) | Implement 1D convolution ||
| **26** | 1D Convolution (Optimized) | Optimize with shared memory ||
| **27** | Profiling DL Kernels | Compare CUDA vs. PyTorch performance ||
| **28** | 2D Convolution (Naïve) | Implement 2D convolution ||
| **29** | 2D Convolution (Optimized) | Use shared memory for optimization ||
| **30** | Im2Col + GEMM Conv | Implement im2col approach ||
| **31** | Depthwise Separable Conv | Optimize CNN inference workloads ||
| **32** | Batch Norm & Activation Fusion | Optimize BN + activation ||
| **33** | Code Review & Optimization | Refine previous work ||
| **34** | Benchmarking ML Kernels | Compare different CNN implementations ||
| **35** | LayerNorm in CUDA | Implement LayerNorm from scratch ||
| **36** | Efficient Dropout | Optimize dropout for training speed ||
| **37** | Fused MLP Block | Implement fused MLP (`GEMM + activation + dropout`) ||
| **38** | Transformer Attention (Naïve) | Implement self-attention kernel ||
| **39** | Optimized Self-Attention | Optimize self-attention with shared memory ||
| **40** | Benchmark Transformer Layers | Compare against `torch.nn.MultiheadAttention` ||
| **41** | Tensor Cores & FP16 | Implement FP16 computation ||
| **42** | Gradient Accumulation | Optimize training with gradient accumulation ||
| **43** | Mixed Precision Training (AMP) | Implement AMP optimizations ||
| **44** | Optimized Attention (FlashAttention) | Implement FlashAttention concepts ||
| **45** | Fused LayerNorm + Dropout | Optimize memory and performance ||
| **46** | Large-Scale Training Profiling | Analyze memory bottlenecks ||
| **Phase 3: Advanced CUDA & Large-Scale ML** | **Goal:** Optimize LLMs, multi-GPU training, and memory-efficient kernels. |||
| **47** | Multi-GPU Data Parallelism | Implement data parallel training ||
| **48** | Multi-GPU Model Parallelism | Implement model parallel training ||
| **49** | Efficient Multi-GPU Communication | Study NCCL and all-reduce ops ||
| **50** | Large Model Optimization | Optimize large-scale deep learning models ||
| **51** | Rotary Embeddings | Implement rotary embeddings in CUDA ||
| **52** | Fused Transformer Block | Implement fused transformer kernel ||
| **53** | LLM Batch Processing | Optimize inference for large batch sizes ||
| **54** | FlashAttention-Like Kernels | Implement memory-efficient attention ||
| **55** | Memory Optimization for LLMs | Optimize LLM inference footprint ||
| **56** | GPU Benchmarking | Compare performance across GPUs ||
| **57** | Architecture-Specific Optimizations | Tune for Ampere/Hopper GPUs ||
| **58** | CUDA Graphs | Implement CUDA Graphs for execution efficiency ||
| **59** | Memory Fragmentation Optimization | Optimize dynamic allocations ||
| **60** | Benchmarking | Compare PyTorch/TensorFlow vs. your CUDA implementations ||
| **61** | Optimize a Real-World Model | Pick a model (BERT/GPT) and optimize ||
| **62** | Custom CUDA Model Acceleration | Implement a custom CUDA-based model optimization ||