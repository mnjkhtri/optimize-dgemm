#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>

__global__
void matmul(double *X, double *Y, double* Z, int N)
{
    int i = 32*blockIdx.y + threadIdx.y; 
    int j = 32*blockIdx.x + threadIdx.x;
    if (i < N && j < N)
    {
        for (int k = 0; k < N; ++k)
        {
            Z[i*N+j] += X[i*N+k] * Y[j*N+k];
        }
    }
}

void cuda(double *X, double *Y, double *Z, int N)
{
    //Allocate GPU memory for given matrices X, Y, Z:
    double *Xc, *Yc, *Zc;

    assert(cudaMalloc(&Xc, N*N*sizeof(double)) == cudaSuccess);
    assert(cudaMemcpy(Xc, X, N*N*sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);

    assert(cudaMalloc(&Yc, N*N*sizeof(double)) == cudaSuccess);
    assert(cudaMemcpy(Yc, Y, N*N*sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);

    assert(cudaMalloc(&Zc, N*N*sizeof(double)) == cudaSuccess);
    assert(cudaMemcpy(Zc, Z, N*N*sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);

    //Organize CUDA threads as 32*32 which is the limit anyways:
    dim3 dimBlock(32, 32);
    //Rest goes into the blocks:
    dim3 dimGrid(N/32, N/32);

    matmul <<<dimGrid,dimBlock>>> (Xc, Yc, Zc, N);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) 
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    //Get back the Z matrix from GPU memory to CPU's:
    assert(cudaMemcpy(Z, Zc, N*N*sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess);

    cudaFree(Xc); cudaFree(Yc); cudaFree(Zc);
    //Freeing the memory is a good practise in itself:
}
