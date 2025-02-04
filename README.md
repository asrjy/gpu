## 100 days of GPU Programming!

### Inspiration: 
- [100 days of GPU](https://github.com/hkproj/100-days-of-gpu)
- 🤑

### Progress

|Day|Code|Notes|Progress|
|---|----|-----|--------|
|000|-|[PMPP](./notes/000/PMPP-Ch1.pdf)|setup environment, lecture 1 of ECE 408, chapter 1 of PMPP|
|001|[vecAbsDiff](./kernels/vecAbsDiff.cu)|[PMPP Chapter 2](./notes/001/)|read chapter 2 of pmpp, implemented vector absolute difference kernel|
|002|[colorToGrayScaleConversion](./kernels/colorToGrayscaleConversion.cu)|[PMPP Chapter 3](./notes/002/)|read half of chapter 2 of pmpp, implemented color to grayscale conversion|
|003|[imageBlur](./kernels/imageBlur.cu)|[PMPP Chapter 3](./notes/002/)|read parts of image blur and about better ways to handle errors, image blurring logic|
|004|[gaussianBlur](./kernels/gaussianBlur.cu)|[PMPP Chapter 3](./notes/002/)|built on top of image blur; struggling to understand multidimensionality;|
|005|[gaussianBlurSharedMemory](./kernels/gaussianBlurSharedMemory.cu)|[PMPP Chapter 3 & exploration](./notes/002/)|built on top of gaussian blur; learnt about shared memory and implemented it;|
|006|[gaussianBlurSharedMemory with event times](./kernels/gaussianBlurSharedMemory.cu)|[event times and performance measurement](./notes/002/)|added perf measurement code to gaussian blur with shared memory kernel|
|007|[vector multiply](./kernels/vecMultiply.cu) and [helpers](./kernels/helpers.h)|[internal structure of blocks](./notes/002/)|setup gpu env on new server. studied heirarchy of execution within the streaming multiprocessor. created helpers file.|



### Resources:
- Programming Massively Parallel Processors
- [CUDA 120 Days Challenge](https://github.com/AdepojuJeremy/Cuda-120-Days-Challenge)
- [ECE 408](https://www.youtube.com/playlist?list=PL6RdenZrxrw-UKfRL5smPfFFpeqwN3Dsz)
- LLMs
