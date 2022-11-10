#include <iostream>
#include <assert.h>
#include <immintrin.h>

//All the other functions need N to refer their elements
static size_t N0;

//Macro for accessing (i,j)th element of a matrix
#define X(i,j) X[(i)*(N0)+(j)]
#define Y(i,j) Y[(i)*(N0)+(j)]
#define Z(i,j) Z[(i)*(N0)+(j)]

//The block size of 128, 256, 512 gives optimum performance
#define BLOCK (256)

/*Function to
    Z[4*4] += X[4*N] x Y[N*4]
            */
static void finddot4by4(double *X, double *Y, double *Z);

//Function for computing Z[BLOCK*BLOCK] += X[BLOCK*BLOCK] x Y[BLOCK*BLOCK]; pass BLOCK as N to it
static void kernel(double *X, double *Y, double *Z);

//Function for computing Z[N*N] = Z[N*N] + X[N*N]*Y[N*N] 
void blocking(double *X, double *Y, double *Z, size_t N)
{

    //True size of the matrix
    N0 = N;

    //Make sure that the memory is aligned
    assert(reinterpret_cast<std::uintptr_t>(X)%32 == 0);
    assert(reinterpret_cast<std::uintptr_t>(Y)%32 == 0);
    assert(reinterpret_cast<std::uintptr_t>(Z)%32 == 0);

    //The ii and jj select the block that will be read to the fullest (which matrix's block is that is decided by the inner loop)
    for (size_t i = 0; i < N; i += BLOCK)
    for (size_t j = 0; j < N; j += BLOCK)

    //As we fix the block, multiplying the submatrices, same size as that block, that use that block
    for (size_t k = 0; k < N; k += BLOCK)
    {
        //Each of these possibilities seem to have same performance
        //kernel(&X(i,k), &Y(k,j), &Z(i,j));
        kernel(&X(k,i), &Y(i,j), &Z(k,j));
        //kernel(&X(i,j), &Y(j,k), &Z(i,k));
    }
}

void kernel(double *X, double *Y, double *Z)
{
    for (size_t i = 0; i < BLOCK; i += 4)
    {
        for (size_t j = 0; j < BLOCK; j += 4)
        {
            finddot4by4(&X(i,0), &Y(0,j), &Z(i,j));
        }
    }
}

//A type alias to accomodate ymm registers
typedef union
{
    __m256d v;
    double d[4];
} v4d_t;

static void finddot4by4(double *X, double *Y, double *Z)
{
    //Registers for Z matrix (four elements in the same row)
    v4d_t
        z00_01_02_03_reg,
        z10_11_12_13_reg,
        z20_21_22_23_reg,
        z30_31_32_33_reg;

    //Initializing to zero
    z00_01_02_03_reg.v = _mm256_setzero_pd();
    z10_11_12_13_reg.v = _mm256_setzero_pd();
    z20_21_22_23_reg.v = _mm256_setzero_pd();
    z30_31_32_33_reg.v = _mm256_setzero_pd();

    //Registers for X matrix (4 elements are to be broadcasted into 4 registers)
    v4d_t x0k_reg, x1k_reg, x2k_reg, x3k_reg;

    //The y's elements are contiguous so one register suffice
    v4d_t yk0_k1_k2_k3_reg;

    for (size_t k = 0; k < BLOCK; ++k)
    {
        /*Broadcast X(0,k), X(1,,x0k_regk), X(2,k), X(3,k) 
                into the registers x0k_reg, x1k_reg, x2k_reg, x3k_reg*/

        x0k_reg.v = _mm256_broadcast_sd((double*)&X(0,k));
        x1k_reg.v = _mm256_broadcast_sd((double*)&X(1,k));
        x2k_reg.v = _mm256_broadcast_sd((double*)&X(2,k));
        x3k_reg.v = _mm256_broadcast_sd((double*)&X(3,k));

        //Get the 4 four elements form matrix Y at once (first address required only)
        yk0_k1_k2_k3_reg.v = _mm256_load_pd(&Y(k,0));

        //z00_01_02_03_reg.v += x0k_reg.v * yk0_k1_k2_k3_reg.v;
        z00_01_02_03_reg.v = _mm256_fmadd_pd(x0k_reg.v, yk0_k1_k2_k3_reg.v, z00_01_02_03_reg.v);
        //z10_11_12_13_reg.v += x1k_reg.v * yk0_k1_k2_k3_reg.v;
        z10_11_12_13_reg.v = _mm256_fmadd_pd(x1k_reg.v, yk0_k1_k2_k3_reg.v, z10_11_12_13_reg.v);
        //z20_21_22_23_reg.v += x2k_reg.v * yk0_k1_k2_k3_reg.v;
        z20_21_22_23_reg.v = _mm256_fmadd_pd(x2k_reg.v, yk0_k1_k2_k3_reg.v, z20_21_22_23_reg.v);
        //z30_31_32_33_reg.v += x3k_reg.v * yk0_k1_k2_k3_reg.v;
        z30_31_32_33_reg.v = _mm256_fmadd_pd(x3k_reg.v, yk0_k1_k2_k3_reg.v, z30_31_32_33_reg.v);
    }

    //Update the memory values
    Z(0,0) += z00_01_02_03_reg.d[0]; 
        Z(0,1) += z00_01_02_03_reg.d[1]; 
            Z(0,2) += z00_01_02_03_reg.d[2];
                Z(0,3) += z00_01_02_03_reg.d[3];

    Z(1,0) += z10_11_12_13_reg.d[0]; 
        Z(1,1) += z10_11_12_13_reg.d[1]; 
            Z(1,2) += z10_11_12_13_reg.d[2];
                Z(1,3) += z10_11_12_13_reg.d[3];

    Z(2,0) += z20_21_22_23_reg.d[0]; 
        Z(2,1) += z20_21_22_23_reg.d[1]; 
            Z(2,2) += z20_21_22_23_reg.d[2];
                Z(2,3) += z20_21_22_23_reg.d[3];

    Z(3,0) += z30_31_32_33_reg.d[0]; 
        Z(3,1) += z30_31_32_33_reg.d[1]; 
            Z(3,2) += z30_31_32_33_reg.d[2];
                Z(3,3) += z30_31_32_33_reg.d[3];
}

//Memory access pattern need optimization