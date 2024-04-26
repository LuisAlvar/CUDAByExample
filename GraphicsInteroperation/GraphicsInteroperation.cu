#include "cuda.h"
#include "cuda_gl_interop.h"
#include "./common/book.h"
#include "./common/cpu_bitmap.h"
#include "./common/GL/glut.h"

#define GL_GLEXT_PROTOTYPES
#define DIM 512 

// Need two variables that will store different handles to the same buffer
GLuint bufferObj; // will be OpenGL's name for the data
cudaGraphicsResource* resource; //will be the CUDA C name for it 

int main(int argc, char** argv)
{
  cudaDeviceProp prop;
  int dev;

  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 1;
  prop.minor = 0;

  HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
}