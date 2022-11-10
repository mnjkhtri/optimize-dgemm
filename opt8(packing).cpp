#include <iostream>
#include <assert.h>
#include <immintrin.h>

//All other functions will need N to reference their elements
static size_t N0;

//Macro for accessing (i,j)th element of a matrix
#define X(i,j) X[(i)*(N0)+(j)]
#define Y(i,j) Y[(i)*(N0)+(j)]
#define Z(i,j) Z[(i)*(N0)+(j)]

//The block size of 64, 128, 256, 512 gives optimum performance
#define BLOCK (128)

/*Function to
    Z[4*4] += X[4*N] x Y[N*4]
            */
static void finddot4by4(double *X, double *Y, double *Z);

//Function for computing Z[BLOCK*BLOCK] += X[BLOCK*BLOCK] x Y[BLOCK*BLOCK]; pass BLOCK as N to it
static void kernel(double *X, double *Y, double *Z);

//Function for computing Z[N*N] = Z[N*N] + X[N*N]*Y[N*N] 
void packing(double *X, double *Y, double *Z, size_t N)
{

    //True size of the matrix
    N0 = N;
    
    //Make sure that the memory is aligned
    assert(reinterpret_cast<std::uintptr_t>(X)%32 == 0);
    assert(reinterpret_cast<std::uintptr_t>(Y)%32 == 0);
    assert(reinterpret_cast<std::uintptr_t>(Z)%32 == 0);

    //Before calling kernel, need to confirm that block size is 256 or less else the stack wont handle
    assert(BLOCK < 512);

    //The ii and jj select the block that will be read to the fullest (which matrix's block is that is decided by the inner loop)
    for (size_t i = 0; i < N; i += BLOCK)
    for (size_t j = 0; j < N; j += BLOCK)
    for (size_t k = 0; k < N; k += BLOCK)
    {
        //Each of these possibilities seem to have same performance
        kernel(&X(i,k), &Y(k,j), &Z(i,j));
        //kernel(&X(k,i), &Y(i,j), &Z(k,j));
        //kernel(&X(i,j), &Y(j,k), &Z(i,k));
    }
}

//PackX will be kernel's helper, packs X[4*BLOCK] into Xp (preserving the matrix property, will destroy submatricity)
void PackX(double *X, double *Xp);

//Packs Y[BLOCK*4] into Yp
void PackY(double *Y, double *Yp);

void kernel(double *X, double *Y, double *Z)
{
    //The block matrices are part of large entire matrices so memory accesses are scattered
    double Xp[BLOCK*BLOCK];
    double Yp[BLOCK*BLOCK];
 
    //Pack X[BLOCK*BLOCK] into Xp[BLOCK*BLOCK]; one loop preserves matricity of X[4*BLOCK] while destroys the submatricity
    for (size_t i = 0; i < BLOCK; i += 4)
    {
        PackX(&X(i,0), &Xp[i*BLOCK]);
    }

    //Pack Y[BLOCK*BLOCK] into Yp[BLOCK*BLOCK]; one loop preserves matricity of Y[BLOCK*4] while destroys the submatricity
    for (size_t j = 0; j < BLOCK; j += 4)
    {
        PackY(&Y(0,j), &Yp[BLOCK*j]);
    }

    //Loop over rows
    for (size_t i = 0; i < BLOCK; i += 4)
    {
        //Loop over column
        for (size_t j = 0; j < BLOCK; j += 4)
        {
            finddot4by4(&Xp[i*BLOCK], &Y(0,j), &Z(i,j));
        }
    }
}

void PackX(double *X, double *Xp)
{
    //Point to first element of each row
    double *x0j_pntr = &X(0,0);
    double *x1j_pntr = &X(1,0);
    double *x2j_pntr = &X(2,0);
    double *x3j_pntr = &X(3,0);

    //Offset pointers to packed arrays where the rows ought to go
    double *Xp0 = Xp;
    double *Xp1 = Xp+BLOCK;
    double *Xp2 = Xp+2*BLOCK;
    double *Xp3 = Xp+3*BLOCK;

    for (size_t j = 0; j < BLOCK; j += 1)
    {
        *Xp0++ = *x0j_pntr++;
        *Xp1++ = *x1j_pntr++;
        *Xp2++ = *x2j_pntr++;
        *Xp3++ = *x3j_pntr++;
    }
}

void PackY(double *Y, double *Yp)
{
    //We will move row after row placing the four elements in the row to the packed array
    for (size_t i = 0; i < BLOCK; i += 1)
    {
        double *yij_pntr = &Y(i,0);

        *Yp++ = *yij_pntr++;
        *Yp++ = *yij_pntr++;
        *Yp++ = *yij_pntr++;
        *Yp++ = *yij_pntr++;
    }
}

//The blocks are no longer submatrices of the large matrix
#undef X
#define X(i,j) X[i*BLOCK+j]

// #undef Y
// #define Y(i,j) Y[i*4+j]

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
        /*Broadcast X(0,k), X(1,k), X(2,k), X(3,k) 
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