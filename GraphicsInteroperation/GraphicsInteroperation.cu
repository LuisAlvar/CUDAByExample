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

  HANDLE_ERROR(cudaGLSetGLDevice( dev ));

  //these GLUT calls need to be made before the other GL calls
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(DIM, DIM);
  glutCreateWindow("bitmap");

  glGenBuffers(1, &bufferObj);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);

}
