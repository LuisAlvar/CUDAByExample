#include "windows.h" //need to include windows.h before any other GL type file
#include "cuda.h"
#include "cuda_gl_interop.h"
#include "./common/book.h"
#include "./common/gpu_anim.h"
#include "./common/GL/glut.h"
#include <gl/GL.h>

/*
nvcc -Llib .\GraphicsInteroperation.cu -o .\bin\GraphicsInteroperation.exe
*/

#define GL_GLEXT_PROTOTYPES
#define DIM 512 

void generate_frame(uchar4* pixels, void*, int ticks);

int main( void )
{
  GPUAnimBitmap bitmap(DIM, DIM, NULL);

  bitmap.anim_and_exit( (void (*)(uchar4*,void*,int)) generate_frame, NULL );
} 
