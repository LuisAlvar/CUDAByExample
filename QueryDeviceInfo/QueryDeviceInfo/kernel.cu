
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"
#include <stdio.h>


int main()
{
	cudaDeviceProp prop;

	int count;

	HANDLE_ERROR(cudaGetDeviceCount(&count));

	for (int i = 0; i < count; i++)
	{
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
		printf("--- General Information for device %d ----\n", i + 1);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);

		printf("Device copy overlap:   ");
		if (prop.deviceOverlap)
		{
			printf("Enabled\n");
		}
		else
		{
			printf("Disabled\n");
		}

		printf("Kernel execution timeout: ");
		if (prop.kernelExecTimeoutEnabled)
		{
			printf("Enabled\n");
		}
		else
		{
			printf("Disabled\n");
		}


		printf("--- Memory Information for device %d ----\n", i + 1);

		printf("Total global mem:  %ld\n", prop.totalGlobalMem);
		printf("Total constant mem:   %ld\n", prop.totalConstMem);
		printf("Max mem pitch:   %ld\n", prop.memPitch);
		printf("Texture Aligment:  %ld\n", prop.textureAlignment);

		printf("--- MP Information for device  %d ----\n", i + 1);
		printf("Multiprocessor count:   %d\n", prop.multiProcessorCount);
		printf("Shared mem per mp:   %ld\n", prop.sharedMemPerBlock);
		printf("Threads in warp:   %d\n", prop.warpSize);
		printf("Max threads per block: maxThreadsPerBlock:  %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: maxThreadsDim: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions:  maxGridSize: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");


	}
}
