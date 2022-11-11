#include <iostream>

//Macro for accessing (i,j)th element of a matrix
#define X(i,j) X[(i)*N+(j)]
#define Y(i,j) Y[(i)*N+(j)]
#define Z(i,j) Z[(i)*N+(j)]

/*Function to
    Z[1*4] += X[1*N] x Y[N*4]
            */
static void finddot4(double *X, double *Y, double *Z, size_t N);

//Function for computing Z[N*N] = Z[N*N] + X[N*N] x Y[N*N]
void inlining(double *X, double *Y, double *Z, size_t N)
{
    //Loop over the rows of Z
    for (size_t i = 0; i < N; ++i)
    {
        //Loop over the cols of Z
        for (size_t j = 0; j < N; j += 4)
        {
            //Find four elements in ith row starting from the jth one (at once) 
            finddot4(&X(i,0), &Y(0,j), &Z(i,j), N);
        }
    }
}

static void finddot4(double *X, double *Y, double *Z, size_t N)
{
    for (size_t k = 0; k < N; ++k)
    {
        //Accessing elements as if the four elements were part of a mini matrix
        Z(0,0) += X(0,k) * Y(k,0); 
        Z(0,1) += X(0,k) * Y(k,1); 
        Z(0,2) += X(0,k) * Y(k,2); 
        Z(0,3) += X(0,k) * Y(k,3); 
    }
}
//Since the inner loops x, y, z elements are accessed frequently, they better be in registers

