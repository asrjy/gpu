## 100 days of GPU Programming!

### Inspiration: 
- [100 days of GPU](https://github.com/hkproj/100-days-of-gpu)
- 🤑

### Progress

|**Day**|**Code**|**Notes**|**Progress**|
|---|----|-----|--------|
|000|-|[PMPP](./notes/000/PMPP-Ch1.pdf)|setup environment, lecture 1 of ECE 408, chapter 1 of PMPP|
|001|[vecAbsDiff](./kernels/vecAbsDiff.cu)|[PMPP Chapter 2](./notes/001/)|read chapter 2 of pmpp, implemented vector absolute difference kernel|
|002|[colorToGrayScaleConversion](./kernels/colorToGrayscaleConversion.cu)|[PMPP Chapter 3](./notes/002/)|read half of chapter 2 of pmpp, implemented color to grayscale conversion|
|003|[imageBlur](./kernels/imageBlur.cu)|[PMPP Chapter 3](./notes/002/)|read parts of image blur and about better ways to handle errors, image blurring logic|
|004|[gaussianBlur](./kernels/gaussianBlur.cu)|[PMPP Chapter 3](./notes/002/)|built on top of image blur; struggling to understand multidimensionality;|
|005|[gaussianBlurSharedMemory](./kernels/gaussianBlurSharedMemory.cu)|[PMPP Chapter 3 & exploration](./notes/002/)|built on top of gaussian blur; learnt about shared memory and implemented it;|
|006|[gaussianBlurSharedMemory with event times](./kernels/gaussianBlurSharedMemory.cu)|[event times and performance measurement](./notes/002/)|added perf measurement code to gaussian blur with shared memory kernel|
|007|[vector multiply](./kernels/vecMultiply.cu) and [helpers](./kernels/helpers.h)|[internal structure of blocks](./notes/002/)|setup gpu env on new server. studied heirarchy of execution within the streaming multiprocessor. created helpers file.|
|008|[matrix multiply](./kernels/matrixMultiplication.cu) and [helpers](./kernels/helpers.h)|[matrix multiplication, pinned memory and BLAS](./notes/002/)|read about pinned memory, pageable memory and cudaHostAlloc(). finished chapter 3 of PMPP|
|009|[matrix transpose](./kernels/matrixTranspose.cu)|[matrix transpose and matrix multiplication blog](./notes/002/)|started reading matrix multiplication blog by siboehm, started chapter 4 of PMPP|
|010|[face blur](./kernels/objectBlur/)|[read matrix multiplication blog](./notes/002/)|reading matrix multiplication blog by siboehm, using a compiled kernel in python|
|011|[optimizing matrix multiplication](./kernels/matrixMultiplication.cu)|[matrix multiplication and profiling](./notes/003/)|using nsys and nvprof, reading matrix multiplication blog by siboehm|



### Resources:
- Programming Massively Parallel Processors
- [CUDA 120 Days Challenge](https://github.com/AdepojuJeremy/Cuda-120-Days-Challenge)
- [ECE 408](https://www.youtube.com/playlist?list=PL6RdenZrxrw-UKfRL5smPfFFpeqwN3Dsz)
- LLMs


### The Plan:


| **Day**  | **Topic**                                      | **Task/Implementation** |**Status**|
|----------|----------------------------------------------|-------------------------|-|
| **Phase 1: Foundations** | **Goal:** Understand CUDA fundamentals, memory hierarchy, and write basic optimized kernels. |||
| **1**  | CUDA Setup & First Kernel  | Install CUDA, write a vector addition kernel |✅|
| **2**  | Thread Hierarchy | Grids, blocks, threads, experimenting with configurations |✅|
| **3**  | Memory Model Basics | Global, shared, local memory overview |✅|
| **4**  | Memory Coalescing | Optimize vector addition using shared memory |✅|
| **5**  | Matrix Multiplication (Naïve) | Implement basic matrix multiplication |✅|
| **6**  | Matrix Multiplication (Optimized) | Use shared memory to optimize ||
| **7**  | Profiling Basics | Use `nvprof` and `nsys` to analyze kernels |✅|
| **8**  | L1/L2 Cache Effects | Study cache behavior and memory bandwidth ||
| **9**  | Tiled Matrix Multiplication | Further optimize matrix multiplication ||
| **10** | Register Pressure | Optimize register usage and reduce spilling ||
| **11** | Warp Execution Model | Avoiding warp divergence ||
| **12** | Parallel Reduction (Naïve) | Implement sum/max reductions ||
| **13** | Parallel Reduction (Optimized) | Optimize with warp shuffle (`__shfl_sync`) ||
| **14** | Code Review & Optimization | Refine and benchmark previous work ||
| **15** | Parallel Scan (Prefix Sum) | Implement parallel scan algorithm ||
| **16** | Histogram (Naïve) | Implement histogram using global memory atomics ||
| **17** | Histogram (Optimized) | Use shared memory to optimize histogram ||
| **18** | Parallel Sorting | Implement bitonic or bucket sort ||
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