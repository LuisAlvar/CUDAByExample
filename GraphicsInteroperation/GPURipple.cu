#include "windows.h" //need to include windows.h before any other GL type file
#include "./common/book.h"
#include "./common/gpu_anim.h"

/*
nvcc -Llib .\GPURipple.cu -o .\bin\GPURipple.exe
*/

#define GL_GLEXT_PROTOTYPES
#define DIM 1024 

void generate_frame(uchar4* pixels, void*, int ticks);
__global__ void kernel(uchar4* ptr, int ticks);


int main( void )
{
  GPUAnimBitmap bitmap(DIM, DIM, NULL);
  bitmap.anim_and_exit( (void (*)(uchar4*,void*,int)) generate_frame, NULL );
}

__global__ void kernel(uchar4* ptr, int ticks)
{
  // map from threadIdx/BlockIdx to pixel position 
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + (y * blockDim.x * gridDim.x);

  // now calculate the value at that position
  float fx = x - DIM/2;
  float fy = y - DIM/2;
  float d = sqrtf(fx*fx + fy*fy);

  unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f));

  ptr[offset].x = grey;
  ptr[offset].y = grey;
  ptr[offset].z = grey;
  ptr[offset].w = 255;
}

void generate_frame(uchar4* pixels, void*, int ticks) {
  dim3 grids(DIM/16, DIM/16);
  dim3 threads(16, 16);
  kernel<<<grids, threads>>>(pixels, ticks);
}


