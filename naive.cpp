#include <iostream>

#define X(i,j) X[(i)*N+(j)]
#define Y(i,j) Y[(j)*N+(i)]
#define Z(i,j) Z[(i)*N+(j)]

void naive(double *X, double *Y, double *Z, int N)
{
    for (int k = 0; k < N; ++k)
    {
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                Z(i,j) += X(i,k)*Y(k,j);
            }
        }
    }
}

