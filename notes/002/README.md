- in a grid, cuda follows SIMD structure ie., single instruction multiple threads. the indices of threads define which portion of the data they work on. 
- the parameters `<<<blocksPerGrid, dimensionsOfBlock>>>` are of the type `dim3`, a vector that can take 3 integer values. we can use fewer than 3 dimensions by setting unused dimension values to 1. example: `dim3 dimGrid(32, 1, 1)`. 
- these can have any name as long as type is dim3. 
- if it's just one dimension, you can just use the value. no need for dim3. the remaining two dimensions automatically take the value of 1.
- `gridDim.x` can take values bw 1 to 2^31 - 1, `gridDim.y` and `gridDim.z` can range from 1 to 2^16 - 1 (65535).         
- `blockDim.x` can range from 0 to `gridDim.x - 1` and so one. 
- max possible total size of block is 1024 threads. they can be distributed in any way within the block (256, 2, 2), (2, 2, 256) etc.,
- a grid dimension can be smaller than a block dimension and vice versa. 
- each `threadIdx` also has three coordinates. 
- the labels of blocks are reversed in dimension as in block(1, 0) means block that has `blockIdx.y = 1` and `blockIdx.x = 0`. 
thread(1, 0, 2) has `threadIdx.z = 1`, `threadIdx.y = 0` and `threadIdx.x = 2`. this is done to help us with mapping thread coordinates for multidimensional data. 
- this reversing of dimensions is because, cuda follows row major layout. so since columns are the fastest varying dimension, threadIdx.x will access these values faster if they are in consecutive memory locations. 
- c requires the number of columns to be known during compile time, but the whole point of using dynamically allocated arrays is that we can use varying size data. hence we usually flatten a dynamically allocated 2d array into an equivalent 1d array. 
- because memory is "flat" in modern computers, all multi dimensional data is flattened. 
- if the data dimension is static, cuda allows us to use higher dimensional indexing. but under the hood, it is still linearized. 
- memory space in a computer is the program's private workspace in computer's memory. its where data and instructions are kept. so when a program needs some data, it takes the starting address and how many bytes are needed to access this data. 
- floats need 4 bytes and doubles need 8 bytes, these multibyte requiring varibles are stored consecutively in memory. 
- row major layout is where all elements of row are consecutively stored in memory. 
- accessing value at jth row and ith column of M, ie., M[j][i], assming there are 4 values in each row is done by j * 4 + i
- column major layour is the transposed form of row major layout. 
- blurring is usually done as a weighted sum of a neighbourhood of the image. it belongs to the convolution pattern. 
- usually weights are given to how far away a pixel is from the current position, this is called gaussian blur. 
- cuda kernel launches are asynchronous. `cudaDeviceSynchrnonize()`  forces the host to wait till gpu is finished executing all preceding cuda calls. this will ensure kernel is completed before any copying is done and catch any errors that might occur during kernel execution. 
- gpu's dram is relatively slower compard to the cuda cores. so everytime a thread needs to fetch memory from there, latency is introduced. 
- shared memory is fast on-chip cache that is shared by all threads within a single block. it's closer to the cuda cores and has significantly low latency and higher bandwidth than global memory. 
- once data has been loaded into the shared memory by a block, all threads in that block will have access to it. 
- gpu memory hierarchy in terms of distance and speed: registers (fastest memory on the gpu; frequently used variables are stored here; practically instantaneous) -> L1 cache (shared by small group of threads or a small group of warps in a streaming multiprocessor) -> shared memory (resides within the shared multiprocessor, the heart and core processing unit of the gpu; lower latency and higher bandwidth than global memory) -> L2 cache (on chip but slower than l1 cache and shared memory; but faster than global memory) -> device memory /dram (largest memory space but located off the chip and connected via a memory bus)
- on chip means things that reside on the same silicon die as the processor's core logic. off chip means things that are not located on the same die, but connected via memory bus or external interfaces. 
- cooperative loading is done when threads load data into the shared memory ie., each thread within block is responsible for loading some data into the shared memory. 
- synchronization: all threads in the block must finish loading data before any thread starts reading from there. `__syncthreads()` will make sure all threads in the block will wait till all threads reach this call. 

multidimensional data
images matrices etc.,

how to orgaize threads efficiently

image processing and comp vision

grids, blocks and threads all can be 3 dimensional 

if 2d image, use 2d threads

grid dim and block dim help us define how big blocks are
there are limits

breakdown datasets to fit into limitations 

using thread id, it can determine what data it works on
ex: find pixel in greyscale

c is designed for flat matrix space 
we have to flatten/linearize them 
cuda c uses row major layout: elements in same row are nex to each other in memory
column major layout: fortran uses it. 

we might have to transpose matrices sometimes 

handling boundary conditions is crucial and dont access non existent data. specific code for this. 

image blurring:

each pixel is output is average of surrounding pixels 
each thread will calculate 1 pixel in blurred output. takes neighbouring of that image and averages it and stores in it. 
each thread has unique id based on position that is used to calcualte where to look for. 
have to include boundary conditions for edges. 

matrix multiplication 
each element will calculate single element of output matrix
thread ids are used for row and column in grid for corresponding input
matrices are stored as 1d array in memory

what if matrix is huge? break down matrix into sub matrices 
where the host code comes in. seperate grids for each sub matrix

have multiple grids working at each time 

another approach is instead of each thread calculating single ouput, calucte multiple 

we can also do dynamic mapping in threads for the data it works on. 