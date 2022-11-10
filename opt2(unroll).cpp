#include <iostream>

//Macro for accessing (i,j)th element of a matrix
#define X(i,j) X[(i)*N+(j)]
#define Y(i,j) Y[(i)*N+(j)]
#define Z(i,j) Z[(i)*N+(j)]

//Function for Z[1*1] += X[1*N] * Y[N*1]
static void finddot(double *X, double *Y, double *Z, size_t N);

//Function for computing Z[N*N] = Z[N*N] + X[N*N] x Y[N*N]
void unrolling(double *X, double *Y, double *Z, size_t N)
{
    //Loop over the rows of Z
    for (size_t i = 0; i < N; ++i)
    {
        //Loop over the cols of Z
        for (size_t j = 0; j < N; j += 4)
        {
            //Find four elements in ith row at once 
            finddot(&X(i,0), &Y(0,j), &Z(i,j), N);
            finddot(&X(i,0), &Y(0,j+1), &Z(i,j+1), N);
            finddot(&X(i,0), &Y(0,j+2), &Z(i,j+2), N);
            finddot(&X(i,0), &Y(0,j+3), &Z(i,j+3), N);
        }
    }
}

static void finddot(double *X, double *Y, double *Z, size_t N)
{
    for (size_t k = 0; k < N; ++k)
    {
        Z(0,0) += X(0,k) * Y(k,0); 
    }
}
//The four consecutive function calls can be inlined
