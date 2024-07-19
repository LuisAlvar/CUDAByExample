#include "./common/book.h"

int main( void ) {

  cudaDeviceProp prop;
  int whichDevice;
  
  HANDLE_ERROR(cudaGetDevice(&whichDevice));
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));

  if (!prop.deviceOverlap)
  {
    printf("Device will not handle overlaps, so no speed up from streams\n");
    return 0;
  }
  
  cudaEvent_t start, stop;
  float elapsedTime;

  // stat the timers
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  // initialize the stream 
  cudaStream_t stream;
  HANDLE_ERROR( cudaStreamCreate(&stream) );

  int *host_a, *host_b,*host_c;
  int *dev_a, *dev_b, *dev_c;

  // allocate the memory on the GPU
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

  // allocate page-locked memory, used to stream
  HANDLE_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));

  for (int i = 0; i < FULL_DATA_SIZE; i++)
  {
    host_a[i] = rand();
    host_b[i] = rand();
  }
  
  // now loop over full data, in bite-sized chunks
  for (int i = 0; i < FULL_DATA_SIZE; i += N)
  {
    // copy the locked memory to the device, async
    HANDLE_ERROR(cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));

    HANDLE_ERROR(cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));

    kernel<<<N/256, 256,0>>>(dev_a, dev_b, dev_c);

    // copy the data from device to locked memory 
    HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
  }
  
}