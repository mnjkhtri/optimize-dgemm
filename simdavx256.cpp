#include <iostream>
#include <immintrin.h>

#define mc (4)
#define nc (8)

void simdavx256(double *X, double *Y, double *Z, int N)
{
    for (int i = 0; i < N; i += mc)
    {
        for (int j = 0; j < N; j += nc)
        {
            const int k = 0;
            //Multiply Z[i*N+j] = X[i*N+k] x Y[j*N+k] | Z[mc*nc] = X[mc*N] x Y[N*nc]

            __m256d Zm[mc][(nc/4)] = {};

            //Get one 1*nc slice of Y:
            for (int kk = 0; kk < N; ++kk)
            {
                __m256d y0_reg = _mm256_load_pd(&Y[j*N+k + kk*4]);
                __m256d y1_reg = _mm256_load_pd(&Y[j*N+k + kk*4+N*4]);
                //y0_reg - y1_reg together form a slice of Y

                //With the slice of 1*nc slice of Y update 1*nc slice of Z getting mc*1 slice of X:
                for (int ii = 0; ii < mc; ++ii)
                {
                    __m256d x_reg = _mm256_broadcast_sd(&X[i*N+k + ii*N+kk]);
                    //x_reg is mc*1 slice of X

                    //Zm[ii][0] - Zm[ii][1] form a 1*nc slice of Z:
                    Zm[ii][0] = _mm256_fmadd_pd(x_reg, y0_reg, Zm[ii][0]);
                    Zm[ii][1] = _mm256_fmadd_pd(x_reg, y1_reg, Zm[ii][1]);
                }
            }

            //In each iteration get the 1*nc of Zm into its actual place:
            for (int ii = 0; ii < mc; ++ii)
            {
                __m256d *Z0 = (__m256d*)&Z[i*N+j + ii*N+0]; *Z0 = _mm256_add_pd(Zm[ii][0], *Z0);
                __m256d *Z1 = (__m256d*)&Z[i*N+j + ii*N+4]; *Z1 = _mm256_add_pd(Zm[ii][1], *Z1);

                //Z0-Z1 together is the 1*nc slice of Z which gets its corresponding slice from Zm
            }
        }
    }
}