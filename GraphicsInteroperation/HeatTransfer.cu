#include "./common/book.h"

// globals needed by the update routine
struct DataBlock {
    float           *dev_inSrc;
    float           *dev_outSrc;
    float           *dev_constSrc;

    cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;
};

void anim_gpu(uchar4* outputBitmap, DataBlock* d, int ticks);

int main() {}

void anim_gpu(uchar4* outputBitmap, DataBlock* d, int ticks)
{
  HANDLE_ERROR( cudaEventRecord(d->start, 0) );

  dim3 blocks
}
