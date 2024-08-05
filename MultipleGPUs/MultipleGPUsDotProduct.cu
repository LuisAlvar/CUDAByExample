#include <stdio.h>
#include "./common/book.h"

// nvcc -Llib ./MultipleGPUsDotProduct.cu -o ./bin/MultipleGPUsDotProduct.exe

#define imin(a,b) (a<b?a:b)
const int N = 33 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);

void* routine(void *pvoidData);

struct DataStruct {
  int deviceId;
  int size;
  float *a;
  float *b;
  float returnValue;
};

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

int main ( void ) {

  int deviceCount;
  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2)  
  {
    printf("We need at least two compute 1.0 or greater devices, but only found %d\n", deviceCount);
    return 0;
  }

  float *a = (float*) malloc(sizeof(float) * N);
  HANDLE_NULL(a);
  float *b = (float*) malloc(sizeof(float) * N);
  HANDLE_NULL(b);

  // fill in the host memory with data
  for( int i = 0; i < N; ++i)
  {
    a[i] = i;
    b[i] = i * 2;
  }
  
  DataStruct data[2];
  data[0].deviceId = 0;
  data[0].size = N/2;
  data[0].a = a;
  data[0].b = b;

  data[1].deviceId = 1;
  data[1].size = N/2;
  data[1].a = a + N/2;
  data[1].b = b + N/2;

  CUTThread thread = start_thread(routine, &(data[0]));
  routine(&(data[1]));

  // before we preoceed, we have the main applicaiton thread wait for the 
  // other thread to finsih by calling end_thread();
  end_thread( thread );

  free(a);
  free(b);

  printf("Value calculated: %f\n", data[0].returnValue + data[1].returnValue);

  return 0;
}

void* routine(void *pvoidData) {
  DataStruct *data = (DataStruct*)pvoidData;
  HANDLE_ERROR(cudaSetDevice(data->deviceId));

  int size = data->size;
  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;
  
  // allocate memory on the PCPU side
  a = data->a;
  b = data->b;
  partial_c = (float*)malloc(blocksPerGrid*sizeof(float));

  // allcoate memory on the GPU
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

  // copy the arrays 'a' and 'b' to the GPU
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice));

  dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

  // Copy the array 'c' back from the GPU to the CPU
  HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

  // finish up on the cpu side
  c = 0;
  for (int i = 0; i < blocksPerGrid; i++)
  {
    c += partial_c[i];
  }

  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_partial_c));
  
  // free memory on the CPU side
  free(partial_c);

  data->returnValue = c;
  return 0;
}