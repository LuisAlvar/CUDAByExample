#include <stdio.h>
#include "./common/book.h"

#define imin(a,b) (a<b?a:b)
const int N = 33 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);

struct DataStruct {
  int deviceId;
  int offset;
  int size;
  float *a;
  float *b;
  float returnValue;
};


int main ( void ) {

  int deviceCount;
  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2)  
  {
    printf("We need at least two compute 1.0 or greater devices, but only found %d\n", deviceCount);
    return 0;
  }

  cudaDeviceProp prop;
  for (int i = 0; i < deviceCount; i++)
  {
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
    if (prop.canMapHostMemory != 1)
    {
      printf("Device %d cannot map memory. \n", i);
      return 0;
    }
  }

  float *a, *b;
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
  HANDLE_ERROR(cudaHostAlloc((void**)&a, N * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped ));
  HANDLE_ERROR(cudaHostAlloc((void**)&b, N * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped ));

  // fill in the host memory with data
  for( int i = 0; i < N; ++i)
  {
    a[i] = i;
    b[i] = i * 2;
  }
  
  DataStruct data[2];
  data[0].deviceId = 0;
  data[0].offset = 0;
  data[0].size = N/2;
  data[0].a = a;
  data[0].b = b;

  data[1].deviceId = 1;
  data[1].offset = N/2; 
  data[1].size = N/2;
  data[1].a = a;
  data[1].b = b;
  
  CUTThread thread = start_thread( routine, &(data[1]) );
  routine(&(data[0]));

  // before we preoceed, we have the main applicaiton thread wait for the 
  // other thread to finsih by calling end_thread();
  end_thread( thread );

  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFreeHost(b));

  printf("Value calculated: %f\n", data[0].returnValue + data[1].returnValue);

  return 0;

}


void* routine(void *pvoidData) {
  DataStruct *data = (DataStruct*)pvoidData;
  if (data->deviceId != 0)
  {
    HANDLE_ERROR(cudaSetDevice(data->deviceId));
    HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
  }
  
  int size = data->size;
  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;
  
  // allocate memory on the PCPU side
  a = data->a;
  b = data->b;
  partial_c = (float*)malloc(blocksPerGrid*sizeof(float));

  // allcoate memory on the GPU
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));
  HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

  // copy the arrays 'a' and 'b' to the GPU
  dev_a += data->offset;
  dev_b += data->offset;

  dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

  // Copy the array 'c' back from the GPU to the CPU
  HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

  // finish up on the cpu side
  c = 0;
  for (int i = 0; i < blocksPerGrid; i++)
  {
    c += partial_c[i];
  }

  HANDLE_ERROR(cudaFree(dev_partial_c));
  
  // free memory on the CPU side
  free(partial_c);

  data->returnValue = c;
  return 0;
}