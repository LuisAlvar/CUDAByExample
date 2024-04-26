#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"
#include "cpu_bitmap.h"

#define DIM 1000

struct cuComplexD
{
    float r;
    float i; 

    __device__ cuComplexD(float x, float y) : r(x), i(y) {}

    __device__ float magnitude2(void) { return r * r + i * i; }
    
    __device__ cuComplexD operator*(const cuComplexD& a)
    {
        return cuComplexD(r * a.r - i * a.i, i*a.r + r*a.i);
    }

    __device__ cuComplexD operator+(const cuComplexD& a)
    {
        return cuComplexD(r + a.r, i + a.i);
    }
};

struct cuComplex
{
    float r;
    float i;

     cuComplex(float x, float y) : r(x), i(y) {}

     float magnitude2(void) { return r * r + i * i; }

     cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

     cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};


int juila(int x, int y)
{
    const float scale = 1.5;

    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (; i < 200; ++i)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
        {
            return 0;
        }
    }

    return 1;
}

__device__ int juilagpu(int x, int y)
{
    const float scale = 1.5;

    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplexD c(-0.8, 0.156);
    cuComplexD a(jx, jy);

    int i = 0;
    for (; i < 200; ++i)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
        {
            return 0;
        }
    }

    return 1;
}

void kernal(unsigned char *p)
{
    for (int row = 0; row < DIM; row++)
    {
        for (int col = 0; col < DIM; col++)
        {
            int offset = col + row * DIM;

            int juilaValue = juila(col, row);

            p[offset * 4 + 0] = 255 * juilaValue;   //RED
            p[offset * 4 + 1] = 0;                  //GREEN
            p[offset * 4 + 2] = 0;                  //BLUE
            p[offset * 4 + 3] = 1;                  //APLHA
         }
    }
}

__global__ void kernalgpu(unsigned char *p)
{
    //x
    int x = blockIdx.x;
    int y = blockIdx.y;

    int offset = x + y * gridDim.x; 

    int juilaValue = juilagpu(x,y);

    p[offset * 4 + 0] = 255 * juilaValue;   //RED
    p[offset * 4 + 1] = 0;                  //GREEN
    p[offset * 4 + 2] = 0;                  //BLUE
    p[offset * 4 + 3] = 1;                  //APLHA
}

int main()
{
    CPUBitmap bitmap(DIM, DIM);
    unsigned char* p = bitmap.get_ptr();
    unsigned char* dev_bitmap;

    cudaMalloc( (void**)&dev_bitmap, bitmap.image_size());

    dim3 grid(DIM, DIM);

    kernalgpu<<<grid, 1 >>>(dev_bitmap);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

    bitmap.display_and_exit();

    cudaFree(dev_bitmap);

    return 0;
}
