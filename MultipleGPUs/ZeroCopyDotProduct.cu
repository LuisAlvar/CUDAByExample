#include <stdio.h>
#include "./common/book.h"

// nvcc -Llib ./ZeroCopyDotProduct.cu -o ./bin/ZeroCopyDotProduct.exe
// First run 
// Value calculated: 27621697910970467221504.000000
// Time using cudaMalloc: 25.6 ms
// Value calculated: 27621697910970467221504.000000
// Time using cudaMalloc: 22.6 ms

#define imin(a,b) (a<b?a:b)
const int N = 33 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);

float malloc_test(int size);
float cuda_host_alloc_test( int size );

__global__ void dot(int size, float *a, float *b, float *c) {
  __shared__ float cache[threadsPerBlock];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;
  float temp = 0;
  while (tid < size)
  {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;

  }
  
  // set the cache valeus
  cache[cacheIndex] = temp;

  // synchronize threads in this block
  __syncthreads();

  // for reductions, threadsPerBlock must be power of 2
  // because of the following code
  int i = blockDim.x / 2;
  while (i != 0)
  {
    if (cacheIndex < i)
    {
      cache[cacheIndex] += cache[cacheIndex + i];
    }
    __syncthreads();
    i /= 2;
  }

  if (cacheIndex == 0)
  {
    c[blockIdx.x] = cache[0];
  }
}

int main( void ) {

  cudaDeviceProp prop;
  int whichDevice;

  HANDLE_ERROR(cudaGetDevice(&whichDevice));
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));

  if (prop.canMapHostMemory != 1)
  {
    printf("Device cannot map memory. \n");
    return 0;
  }

  // Runtime setting: indicates that we want the device to be allowed to map host 
  // memory. 
  HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
  
  // Perform two tests, display the elapsed time, and exit the application
  float elapsedTime = malloc_test(N);
  printf("Time using cudaMalloc: %3.1f ms\n", elapsedTime);

  elapsedTime = cuda_host_alloc_test(N);
  printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);

}



float malloc_test(int size) {

  cudaEvent_t start, stop;

  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;
  float elapsedTime;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  // allocate memory on the CPU side
  a = (float*)malloc(size * sizeof(float));
  b = (float*)malloc(size * sizeof(float));
  partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

  // allocate the memory on the GPU
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

  // fill in the host memory with data
  for(int i = 0; i < size; ++i) {
    a[i] = i;
    b[i] = i * 2;
  }

  // Start the time event 
  HANDLE_ERROR(cudaEventRecord(start, 0));

  // copy the arrays 'a' and 'b' to the GPU
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice));

  dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

  // copy the array 'c' back from the GPU to the CPU
  HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost));

  // stop our event timer
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

  // finish up on the CPU side
  // we sum our partial results and free our input and output buffers.
  c = 0;
  for (int i = 0; i < blocksPerGrid; i++)
  {
    c += partial_c[i];
  }
  
  // free memory on the GPU side
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_partial_c));

  // free memory on the CPU side
  free(a);
  free(b);
  free(partial_c);

  // free events
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  printf("Value calculated: %f\n", c);

  return elapsedTime;
}


float cuda_host_alloc_test( int size ) {

  cudaEvent_t start, stop;

  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;
  float elapsedTime;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  // allocate the memory on the CPU 
  HANDLE_ERROR(cudaHostAlloc((void**)&a, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));
  HANDLE_ERROR(cudaHostAlloc((void**)&b, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));
  HANDLE_ERROR(cudaHostAlloc((void**)&partial_c, blocksPerGrid * sizeof(float), cudaHostAllocMapped));

  for (int i = 0; i < size; i++)
  {
    a[i] = i;
    b[i] = i * 2;
  }
  
  // CPU pointers to GPU pointers memory mapping 
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0));

  // Start our timer and launch our kernel 
  HANDLE_ERROR(cudaEventRecord(start, 0));
  dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);
  HANDLE_ERROR(cudaThreadSynchronize());

  // After synchronizing, stop our timer and finsh the computation on the CPU as we did before. 
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

  // finish up on the CPU side
  c = 0;
  for (int i = 0; i < blocksPerGrid; i++)
  {
    c += partial_c[i];
  }
  
  // clean up 
  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFreeHost(b));
  HANDLE_ERROR(cudaFreeHost(partial_c));

  // free events
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  printf("Value calculated: %f\n", c);

  return elapsedTime;
}