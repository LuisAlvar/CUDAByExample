#include "./common/book.h"
#include <time.h>

#define SIZE (100*1024*1024)

/*
nvcc .\CPUHistogram.cu -o .\bin\CPUHistogram.exe
*/


int main( void )
{

  unsigned char* buffer = (unsigned char*) big_random_block(SIZE);
  
  clock_t startTime = clock();


  unsigned int histo[256];

  for (int i = 0; i < 256; i++)
  {
    histo[i] = 0;
  }

  for (int i = 0; i < SIZE; i++)
  {
    histo[buffer[i]]++;
  }

  clock_t endTime = clock();
  double elapsedMilliSeconds = ( (double)(endTime - startTime)) * 1000.0 / CLOCKS_PER_SEC;

  long histoCount = 0;
  for (int i = 0; i < 256; i++)
  {
    histoCount += histo[i];
  }
  
  printf("Histogram Sum: %ld\n", histoCount);
  printf("Elapsed time is %.2lf ms.\n", elapsedMilliSeconds);
  free(buffer);

  return  0;
}