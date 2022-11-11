#include <iostream>
#include <assert.h>
#include <immintrin.h>

#define BLOCK_X (4)
#define BLOCK_Y (8)

static void find_a_block(double *X, double *Y, double *Z, size_t N);

void finalboss(double *X, double *Y, double *Z, size_t N)
{
    for (size_t i = 0; i < N; i += BLOCK_X)
    {
        for (size_t j = 0; j < N; j += BLOCK_Y)
        {
            //For values of ij will be finding the ijth block:
            //This is done in N updates as follow
            __m256d Zm[BLOCK_X][(BLOCK_Y/4)] = {};

            for (size_t k = 0; k < N; ++k)
            {
                //For values of k will update the ijth block for the kth time
                __m256d yk0_k1_k2_k3_reg = _mm256_load_pd(&Y[N*j+k*4]);
                __m256d yk4_k5_k6_k7_reg = _mm256_load_pd(&Y[N*j+N*4+k*4]);
                
                for (size_t iy = 0; iy < BLOCK_X; iy += 1)
                {
                    //For values of iy will be finding the iy row of ijth block
                    __m256d xik_reg;
                    xik_reg = _mm256_broadcast_sd((double*)&X[(i*N)+iy*N+k]);

                    Zm[iy][0] = _mm256_fmadd_pd(xik_reg, yk0_k1_k2_k3_reg, Zm[iy][0]);
                    Zm[iy][1] = _mm256_fmadd_pd(xik_reg, yk4_k5_k6_k7_reg, Zm[iy][1]);
                }
            }

            for (size_t iy = 0; iy < BLOCK_X; ++iy)
            {
                __m256d *Zi0 = (__m256d*)&Z[(i*N+j)+(iy*N)];
                *Zi0 = _mm256_add_pd(Zm[iy][0], *Zi0);

                __m256d *Zi1 = (__m256d*)&Z[(i*N+j)+(iy*N)+4];
                *Zi1 = _mm256_add_pd(Zm[iy][1], *Zi1);
            }
        }
    }
}