#include "windows.h" //need to include windows.h before any other GL type file
#include "cuda.h"
#include "cuda_gl_interop.h"
#include "./common/book.h"
#include "./common/cpu_bitmap.h"
#include "./common/GL/glut.h"
#include <gl/GL.h>

/*
nvcc -Llib .\GraphicsInteroperation.cu -o .\bin\GraphicsInteroperation.exe
*/

#define GL_GLEXT_PROTOTYPES
#define DIM 512 

PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL;
PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData     = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL;

// Need two variables that will store different handles to the same buffer
GLuint bufferObj; // will be OpenGL's name for the data
cudaGraphicsResource* resource; //will be the CUDA C name for it 

__global__ void kernel(uchar4* ptr);
static void draw_func(void);
static void key_func(unsigned char key, int x, int y);

int main(int argc, char** argv)
{
  cudaDeviceProp prop;
  int dev;

  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 1;
  prop.minor = 0;

  HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

  HANDLE_ERROR(cudaGLSetGLDevice( dev ));

  //these GLUT calls need to be made before the other GL calls
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(DIM, DIM);
  glutCreateWindow("bitmap");

  glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
  glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
  glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");
  glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");

  glGenBuffers(1, &bufferObj);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);

  HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));

  uchar4* devPtr;
  size_t size;
  HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));
  HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));

  dim3 grids(DIM/16, DIM/16);
  dim3 threads(16,16);

  kernel<<<grids, threads>>>(devPtr);

  HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));

  // set up GLUT and kick off main loop
  glutKeyboardFunc(key_func);
  glutDisplayFunc(draw_func);
  glutMainLoop();

}

// based on ripple code, but uses uchar4, which is the type of data graphics interop uses
__global__ void kernel(uchar4* ptr)
{
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + (y * blockDim.x * gridDim.x);

  // now calculate the value at that position
  float fx = x/(float)DIM - 0.5f;
  float fy = y/(float)DIM - 0.5f;
  unsigned char green = 128 + 127 * sin(abs(fx*100) - abs(fy*100));

  // accessing uchar4 vs. unsigned char*
  ptr[offset].x = 0;
  ptr[offset].y = green;
  ptr[offset].z = 0;
  ptr[offset].w = 255;
}

static void draw_func(void)
{
  glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glutSwapBuffers();
}

static void key_func(unsigned char key, int x, int y)
{
  switch (key)
  {
    case 27:
      // clean up OpenGl and CUDA
      HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
      glDeleteBuffers(1, &bufferObj);
      exit(0);
  }
}