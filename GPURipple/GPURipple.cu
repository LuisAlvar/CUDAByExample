#include "./common/cpu_anim.h"
#include "./common/book.h"

#define DIMX 1360
#define DIMY 768

/*
nvcc -Llib ./GPURipple.cu -o ./bin/GPURipple.exe
*/

struct DataBlock {
	unsigned char* dev_bitmap;
	CPUAnimBitmap* bitmap;
};


__global__ void kernel(unsigned char* ptr, int ticks)
{
	//Finding the x,y coordinates
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	//Using (x,y) to determine the linearize index 
	int linear_offset = x + y * blockDim.x * gridDim.x;

	if (linear_offset < (DIMX * DIMY))
	{
		//now caluclate the value at that position 
		float fx = x - DIMX / 2;
		float fy = y - DIMY / 2;
		float d = sqrtf(fx * fx + fy * fy);

		unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

		ptr[linear_offset * 4 + 0] = grey;
		ptr[linear_offset * 4 + 1] = grey;
		ptr[linear_offset * 4 + 2] = grey;
		ptr[linear_offset * 4 + 3] = 255;

	}

}

//This function will be called by the strucutre every time it wants to generate a new frame of the animation.  
void generate_frame(DataBlock *d, int ticks) 
{
	dim3 blocks( (DIMX + 15)/16, (DIMY + 15)/16);
	dim3 threads(16, 16);

	kernel<<< blocks, threads >>>(d->dev_bitmap, ticks);

	cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost);
}

// clean up memory allocated on the GPU
void cleanup(DataBlock *d)
{
	cudaFree(d->dev_bitmap);
}

int main( void )
{
	DataBlock data;
	CPUAnimBitmap bitmap(DIMX,DIMY, &data);
	data.bitmap = &bitmap;

	HANDLE_ERROR(cudaMalloc( (void**)&data.dev_bitmap, bitmap.image_size() ));

	// We pass a function pointer to generate_frame() 
	bitmap.anim_and_exit((void (*)(void*, int))generate_frame, (void (*)(void*))cleanup);

	return 0;
}
