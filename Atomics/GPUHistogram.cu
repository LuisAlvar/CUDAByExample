#include "./common/book.h"

#define SIZE (100*1024*1024)

__global__ void histo_kernel(unsigned char* buffer, long size, unsigned int * histo);

int main( void ) {

  // Random input stream data
  unsigned char *buffer = (unsigned char*) big_random_block( SIZE );

  // Initialize events for timing our task
  cudaEvent_t start, stop;
  
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  // allocate memory on the GPU for the file's data
  unsigned char *dev_buffer;
  unsigned int *dev_histo;

  HANDLE_ERROR(cudaMalloc((void**)&dev_buffer, SIZE)); // allocation
  HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice)); // data transfer: CPU -> GPU

  HANDLE_ERROR(cudaMalloc((void**)&dev_histo, 256*sizeof(int))); // allocation
  HANDLE_ERROR(cudaMemset(dev_histo, 0, 256*sizeof(int))); // initialization of GPU buffer

  // TODO: Kernel call function 
  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  int blocks = prop.multiProcessorCount;
  printf("Num of MultiProcessors: %d\n", blocks);
  histo_kernel<<<blocks*2, 256>>>(dev_buffer, SIZE, dev_histo);

  unsigned int histo[256];
  HANDLE_ERROR(cudaMemcpy(histo, dev_histo, 256* sizeof(int), cudaMemcpyDeviceToHost)); // data transfer: GPU -> CPU

  // get stop time, and display the timing result
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  float elapsedTime;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Time to generate: %3.1f ms\n");

  // Unit Test: Verification
  long histoCount = 0;
  for (int i = 0; i < 256; i++)
  {
    histoCount += histo[i];
  }
  printf("Histogram Sum: %ld\n", histoCount);
  
  // verify that we have the same counts via CPU
  for (int i = 0; i < SIZE; i++)
  {
    histo[buffer[i]]--;
  }
  for (int i = 0; i < 256; i++)
  {
    if (histo[i] != 0)
    {
      printf("Failure at %d\n", i);
    }
  }
  
  // Clean up
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  cudaFree(dev_histo);
  cudaFree(dev_buffer);
  free(buffer);
  
  return 0;
}

__global__ void histo_kernel(unsigned char* buffer, long size, unsigned int * histo) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  while (i < size)  
  {
    atomicAdd(&(histo[buffer[i]]), 1);
    i += stride;
  }
  
}
