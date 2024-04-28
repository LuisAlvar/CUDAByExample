#include "cuda.h"
#include "cuda_gl_interop.h"
#include "./common/book.h"
#include "./common/cpu_bitmap.h"
#include "./common/GL/glut.h"

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
