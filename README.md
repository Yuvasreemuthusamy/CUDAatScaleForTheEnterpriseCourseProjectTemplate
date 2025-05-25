### main.cu
```
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include "merge_sort.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;

long* generateRandomLongArray(int numElements) {
    long* arr = new long[numElements];
    for (int i = 0; i < numElements; ++i) {
        arr[i] = rand() % 256;
    }
    return arr;
}

int main(int argc, char** argv) {
    // Default config
    dim3 threadsPerBlock(32);
    dim3 blocksPerGrid(8);
    int numElements = 1024;

    // Parse command-line if needed (omitted here)
    // threadsPerBlock.x = ..., numElements = ..., etc.

    long* data = generateRandomLongArray(numElements);

    std::cout << "Unsorted data:\n";
    for (int i = 0; i < numElements; ++i)
        std::cout << data[i] << " ";
    std::cout << "\n";

    long* sorted = mergesort(data, numElements, threadsPerBlock, blocksPerGrid);

    std::cout << "\nSorted data:\n";
    for (int i = 0; i < numElements; ++i)
        std::cout << sorted[i] << " ";
    std::cout << "\n";

    delete[] data;
    return 0;
}
```
### merge_sort.cu
```
#include "merge_sort.h"

__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] <= source[j])) {
            dest[k] = source[i++];
        } else {
            dest[k] = source[j++];
        }
    }
}

__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x * (x *= threads->z) +
           blockIdx.y * (x *= blocks->z) +
           blockIdx.z * (x *= blocks->y);
}

__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width * idx * slices;

    for (long slice = 0; slice < slices; ++slice) {
        if (start >= size) break;

        long middle = min(start + (width / 2), size);
        long end = min(start + width, size);

        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

__host__ long* mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    long *D_data, *D_swp;
    dim3 *D_threads, *D_blocks;

    long* result = new long[size];
    cudaMalloc(&D_data, sizeof(long) * size);
    cudaMalloc(&D_swp, sizeof(long) * size);
    cudaMalloc(&D_threads, sizeof(dim3));
    cudaMalloc(&D_blocks, sizeof(dim3));

    cudaMemcpy(D_data, data, sizeof(long) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    long* A = D_data;
    long* B = D_swp;
    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    for (long width = 2; width < (size << 1); width <<= 1) {
        long slices = size / (nThreads * width) + 1;
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);
        cudaDeviceSynchronize();
        std::swap(A, B);
    }

    cudaMemcpy(result, A, sizeof(long) * size, cudaMemcpyDeviceToHost);

    cudaFree(D_data);
    cudaFree(D_swp);
    cudaFree(D_threads);
    cudaFree(D_blocks);

    return result;
}
```
