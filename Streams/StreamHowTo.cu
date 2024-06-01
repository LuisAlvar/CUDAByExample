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

  //
  
}