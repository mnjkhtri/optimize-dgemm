#include <iostream>

//Macro for accessing (i,j)th element of a matrix
#define X(i,j) X[(i)*N+(j)]
#define Y(i,j) Y[(i)*N+(j)]
#define Z(i,j) Z[(i)*N+(j)]

//Macro of unroll count
#define UNROLL 4

/*Function to
    Z[4*4] += X[4*N] x Y[N*4]
            */
static void finddot4by4(double *X, double *Y, double *Z, size_t N);

//Function for computing Z[N*N] = Z[N*N] + X[N*N] x Y[N*N]
void more(double *X, double *Y, double *Z, size_t N)
{
    //Loop over the rows of Z
    for (size_t i = 0; i < N; i += 4)
    {
        //Loop over the cols of Z
        for (size_t j = 0; j < N; j += 4)
        {
            //Find the mini matrix here
            finddot4by4(&X(i,0), &Y(0,j), &Z(i,j), N);
        }
    }
}

static void finddot4by4(double *X, double *Y, double *Z, size_t N)
{
    //In one interation of the loop, we have to update four elements
    //We need to get 4*4 elements from Z matrix in registers to continuously update them and finally put into memory
    //Need 4 registers for X since at each interation one element from each row is needed

    //Registers for Z matrix
    register double z00_reg, z01_reg, z02_reg, z03_reg,
                    z10_reg, z11_reg, z12_reg, z13_reg,
                    z20_reg, z21_reg, z22_reg, z23_reg,
                    z30_reg, z31_reg, z32_reg, z33_reg;

    z00_reg = 0.0; z01_reg = 0.0; z02_reg = 0.0; z03_reg = 0.0;
    z10_reg = 0.0; z11_reg = 0.0; z12_reg = 0.0; z13_reg = 0.0;
    z20_reg = 0.0; z21_reg = 0.0; z22_reg = 0.0; z23_reg = 0.0;
    z30_reg = 0.0; z31_reg = 0.0; z32_reg = 0.0; z33_reg = 0.0;

    //Registers for X matrix
    register double x0k_reg, x1k_reg, x2k_reg, x3k_reg;

    //Since we will be using the accessed y's frequently inside the loop we put them in registers
    register double yk0_reg, yk1_reg, yk2_reg, yk3_reg;

    for (size_t k = 0; k < N; ++k)
    {
        //Get the X's elements here (each from one row)
        x0k_reg = X(0,k);
        x1k_reg = X(1,k);
        x2k_reg = X(2,k);
        x3k_reg = X(3,k);

        //Get the Y's elements (each from one column)
        yk0_reg = Y(k,0);
        yk1_reg = Y(k,1);
        yk2_reg = Y(k,2);
        yk3_reg = Y(k,3);

        //First row
        z00_reg += x0k_reg*yk0_reg;
        z01_reg += x0k_reg*yk1_reg;
        z02_reg += x0k_reg*yk2_reg;
        z03_reg += x0k_reg*yk3_reg;

        //Second row
        z10_reg += x1k_reg*yk0_reg;
        z11_reg += x1k_reg*yk1_reg;
        z12_reg += x1k_reg*yk2_reg;
        z13_reg += x1k_reg*yk3_reg;

        //Third row
        z20_reg += x2k_reg*yk0_reg;
        z21_reg += x2k_reg*yk1_reg;
        z22_reg += x2k_reg*yk2_reg;
        z23_reg += x2k_reg*yk3_reg;

        //Fourth row
        z30_reg += x3k_reg*yk0_reg;
        z31_reg += x3k_reg*yk1_reg;
        z32_reg += x3k_reg*yk2_reg;
        z33_reg += x3k_reg*yk3_reg;

    }
    Z(0,0) += z00_reg; Z(0,1) += z01_reg; Z(0,2) += z02_reg; Z(0,3) += z03_reg;
    Z(1,0) += z10_reg; Z(1,1) += z11_reg; Z(1,2) += z12_reg; Z(1,3) += z13_reg;
    Z(2,0) += z20_reg; Z(2,1) += z21_reg; Z(2,2) += z22_reg; Z(2,3) += z23_reg;
    Z(3,0) += z30_reg; Z(3,1) += z31_reg; Z(3,2) += z32_reg; Z(3,3) += z33_reg;

}

//TIME FOR VECTORS (change registers of 4by4 only)