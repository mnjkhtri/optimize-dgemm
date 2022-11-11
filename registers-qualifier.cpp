#include <iostream>

//Registers need O3 too

//Macro for accessing (i,j)th element of a matrix
#define X(i,j) X[(i)*N+(j)]
#define Y(i,j) Y[(i)*N+(j)]
#define Z(i,j) Z[(i)*N+(j)]

/*Function to
    Z[1*4] += X[1*N] x Y[N*4]
            */
static void finddot4(double *X, double *Y, double *Z, size_t N);

//Function for computing Z[N*N] = Z[N*N] + X[N*N] x Y[N*N]
void registers(double *X, double *Y, double *Z, size_t N)
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
    //Declaring registers for z's elements 
    register double z00_reg, z01_reg, z02_reg, z03_reg;

    z00_reg = 0.0;
    z01_reg = 0.0;
    z02_reg = 0.0;
    z03_reg = 0.0; 

    //Likewise for x's and y's
    register double x0k_reg;
    register double yk0_reg, yk1_reg, yk2_reg, yk3_reg;

    for (size_t k = 0; k < N; ++k)
    {
        //Initialize the registers with x's and y's elements to be used in this iteration
        x0k_reg = X(0,k);

        yk0_reg = Y(k,0);
        yk1_reg = Y(k,1);
        yk2_reg = Y(k,2);
        yk3_reg = Y(k,3);

        //Accessing elements as if the four elements were part of a mini matrix

        z00_reg += x0k_reg * yk0_reg; 
        z01_reg += x0k_reg * yk1_reg; 
        z02_reg += x0k_reg * yk2_reg; 
        z03_reg += x0k_reg * yk3_reg;  
    }
    
    Z(0,0) += z00_reg;
    Z(0,1) += z01_reg;
    Z(0,2) += z02_reg;
    Z(0,3) += z03_reg;
}
//Modern compilers will probably just ignore the register request