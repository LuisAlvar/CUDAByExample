#include "./common/book.h"

#define SIZE (10*1024*1024)

// nvcc -Llib .\PageLockMemory.cu -o ./bin/PageLockMemory.exe

float cuda_malloc_test(int size, bool up);
float cuda_host_alloc_test(int size, bool up);

int main( void ) {
  float elapsedTime;
  float MB = (float)100*SIZE*sizeof(int)/1024/1024;

  elapsedTime = cuda_malloc_test(SIZE, true);

  printf("Time using cudaMalloc: %3.1f ms\n", elapsedTime);
  printf("\tMB/s during copy up: %3.1f\n", MB/(elapsedTime/1000));

  elapsedTime = cuda_malloc_test(SIZE, false);

  printf("Time using cudaMalloc: %3.1f ms\n", elapsedTime);
  printf("\tMB/s during copy down: %3.1f\n", MB/(elapsedTime/1000));

  elapsedTime = cuda_host_alloc_test(SIZE, true);

  printf("Time using cudaHostMalloc: %3.1f ms\n", elapsedTime);
  printf("\tMB/s during copy up: %3.1f\n", MB/(elapsedTime/1000));

  elapsedTime = cuda_host_alloc_test(SIZE, false);

  printf("Time using cudaHostMalloc: %3.1f ms\n", elapsedTime);
  printf("\tMB/s during copy down: %3.1f\n", MB/(elapsedTime/1000));

}

/// @brief Using pageable memory will transfer 100 copies of the buffer in one direction.
/// @param size The size of the buffer 
/// @param up True means will load data from host to device; False load data from device to host
/// @return The elapsed time taked by the standard pageable host memory logic on tansfering data from either host to device and vise visa.
float cuda_malloc_test(int size, bool up) {
  cudaEvent_t start, stop;
  int *a, *dev_a;
  float elapsedTime;

  // Setting up our start and stop events
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  // CPU allocates standard, pageable host memory
  a = (int*)malloc(size * sizeof(*a));
  HANDLE_NULL(a);

  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));

  // Start the event 
  HANDLE_ERROR(cudaEventRecord(start, 0));

  for (int i = 0; i < 100; i++)
  {
    if (up)
    {
      HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice));
    }
    else {
      HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
    }
  }

  // Stop the event 
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  // Calculate the difference 
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

  free(a);
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  return elapsedTime;
}

/// @brief Using pinned memory will transfer 100 copies of the buffer in one direction 
/// @param size The size of the buffer
/// @param up True means will load data from host to device; False load data from device to host.
/// @return The elapsed time take by the allocating pinned memory within host 
float cuda_host_alloc_test(int size, bool up) {
  cudaEvent_t start, stop;
  int *a, *dev_a;
  float elapsedTime;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  // GPU allocates pinned memory using cudaHostAlloc to allocate a page-locked buffer.
  // cudaHostAllocDefault flag for default page-locked memory behavior.
  HANDLE_ERROR(cudaHostAlloc((void**)&a, size * sizeof(*a), cudaHostAllocDefault));
 
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));

  HANDLE_ERROR(cudaEventRecord(start, 0));
  for (int i = 0; i < 100; i++)
  {
    if (up)
    {
      HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice));
    }
    else {
      HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(*a), cudaMemcpyDeviceToHost));
    }
  }

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  return elapsedTime;
}