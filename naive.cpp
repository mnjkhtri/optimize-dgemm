#include <iostream>

#define X(i,j) X[(i)*N+(j)]
#define Y(i,j) Y[(i)*N+(j)]
#define Z(i,j) Z[(i)*N+(j)]

void naive(double *X, double *Y, double *Z, int N)
{
    for (int p = 0; p < N; ++p)
    {
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                Z(i,j) += X(i,p)*Y(p,j);
            }
        }
    }
}

