#include <stdio.h>
#include "./common/book.h"

float malloc_test(int size);
float cuda_host_alloc_test( int size );


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
  dot<<<blocksPerGrids, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);
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