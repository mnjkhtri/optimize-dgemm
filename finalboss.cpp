#include <iostream>
#include <assert.h>
#include <immintrin.h>

//All other functions will need N to reference their elements
static size_t N0;

//Macro for accessing (i,j)th element of a matrix
#define X(i,j) X[(i)*(N0)+(j)]
#define Y(i,j) Y[(i)*(N0)+(j)]
#define Z(i,j) Z[(i)*(N0)+(j)]

//64 bytes will fit into L2 cache (100 bytes was the limit), 32 (or even 16 to be safest) seems to do the best here
#define BLOCK (16)

#define BLOCK_X (4)
//Currently BLOCK_Y must be 8
#define BLOCK_Y (8)
/*Function to
    Z[BLOCK_X*BLOCK_Y] += X[BLOCK_X*N] x Y[N*BLOCK_Y]
            */
static void finddot4by4(double *X, double *Y, double *Z);

//Function for computing Z[BLOCK*BLOCK] += X[BLOCK*BLOCK] x Y[BLOCK*BLOCK]; pass BLOCK as N to it
static void kernel(double *X, double *Y, double *Z);

//Function for computing Z[N*N] = Z[N*N] + X[N*N]*Y[N*N] 
void finalboss(double *X, double *Y, double *Z, size_t N)
{

    //True size of the matrix
    N0 = N;
    
    //Make sure that the memory is aligned
    assert(reinterpret_cast<std::uintptr_t>(X)%32 == 0);
    assert(reinterpret_cast<std::uintptr_t>(Y)%32 == 0);
    assert(reinterpret_cast<std::uintptr_t>(Z)%32 == 0);

    //Before calling kernel, need to confirm that block size is 256 or less else the stack wont handle
    //assert(BLOCK < 512);

    //The ii and jj select the block that will be read to the fullest (which matrix's block is that is decided by the inner loop)
    for (size_t i = 0; i < N; i += BLOCK)
    for (size_t j = 0; j < N; j += BLOCK)
    for (size_t k = 0; k < N; k += BLOCK)
    {
        //Each of these possibilities seem to have same performance
        //kernel(&X(i,k), &Y(k,j), &Z(i,j));
        //kernel(&X(k,i), &Y(i,j), &Z(k,j));
        kernel(&X(i,j), &Y(j,k), &Z(i,k));
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
    // for (size_t i = 0; i < BLOCK; i += 4)
    // {
    //     PackX(&X(i,0), &Xp[i*BLOCK]);
    // }

    //Pack Y[BLOCK*BLOCK] into Yp[BLOCK*BLOCK]; one loop preserves matricity of Y[BLOCK*4] while destroys the submatricity
    // for (size_t j = 0; j < BLOCK; j += 4)
    // {
    //     PackY(&Y(0,j), &Yp[BLOCK*j]);
    // }

    //Loop over rows
    for (size_t i = 0; i < BLOCK; i += BLOCK_X)
    {
        //Loop over column
        for (size_t j = 0; j < BLOCK; j += BLOCK_Y)
        {
            finddot4by4(&X(i,0), &Y(0,j), &Z(i,j));
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


//We have to multiply Z[BLOCK_X*BLOCK_Y] = X[BLOCK_Y*BLOCK] x Y[BLOCK*BLOCK_Y]
static void finddot4by4(double *X, double *Y, double *Z)
{
    //Need a Zm matrix of ymm's that will fit the whole Z[BLOCK_X*BLOCK_Y]

    //Need Y to be multiple of 4 so fits into one ymm
    assert(BLOCK_Y % 4 == 0);
    __m256d Zm[BLOCK_X][(BLOCK_Y/4)] = {}; //Also initialized to zero

    for (size_t k = 0; k < BLOCK; ++k)
    {
        __m256d yk0_k1_k2_k3_reg;
        yk0_k1_k2_k3_reg = _mm256_load_pd(&Y(k,0));
        __m256d yk4_k5_k6_k7_reg;
        yk4_k5_k6_k7_reg = _mm256_load_pd(&Y(k,4));
        //For any value of k; we are supposed to update Z[BLOCK_X*BLOCK_Y] with one slice of each X and Y in their direction of BLOCK
        for (size_t i = 0; i < BLOCK_X; i += 1)
        {
            __m256d xik_reg;
            xik_reg = _mm256_broadcast_sd((double*)&X(i,k));

            //For certain value of i we need to find the ith row in the given BLOCK_X*BLOCK_Y

            Zm[i][0] = _mm256_fmadd_pd(xik_reg, yk0_k1_k2_k3_reg, Zm[i][0]);
            Zm[i][1] = _mm256_fmadd_pd(xik_reg, yk4_k5_k6_k7_reg, Zm[i][1]);

        }
    }

    for (size_t i = 0; i < BLOCK_X; ++i)
    {
        __m256d *Zi0 = (__m256d*)&Z(i,0);
        *Zi0 = _mm256_add_pd(Zm[i][0], *Zi0);

        __m256d *Zi1 = (__m256d*)&Z(i,4);
        *Zi1 = _mm256_add_pd(Zm[i][1], *Zi1);
    }
}