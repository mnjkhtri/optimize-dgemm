#include <iostream>

//Must be 16 here and N is asseumbed to be 1024
#define BLOCK 16

extern "C" void FindRow_ (double* Input1, double* Input2, double* Output);

void avx(double *X, double *Y, double *Z, size_t N)
{
    for (size_t jj = 0; jj < N; jj += BLOCK)
    for (size_t kk = 0; kk < N; kk += BLOCK)
    //for (size_t ii = 0; ii < N; ii += BLOCK)

    for (size_t i = 0; i < N; i += 1)
    {
        FindRow_((X+i*N+kk), (Y+kk*N+jj), (Z+i*N+jj));
    }
}
