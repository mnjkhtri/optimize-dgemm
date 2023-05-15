//g++ -O3 -march=native *.c && ./a.out

#include <iostream>
#include <assert.h>
#include <fstream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstring>

const size_t N = 512;

//Alignment required for vector accesses- 64 bytes for cache blocks]
alignas(64) double X[N*N];
alignas(64) double Y[N*N];
alignas(64) double Z[N*N];
alignas(64) double Z_o[N*N];

//Swiggling for vector instructions
alignas(64) double Yc[N*N];

extern void naive(double *X, double *Y, double *Z, int N);
extern void simdavx256(double *X, double *Y, double *Z, int N);
extern void multithreading(double *X, double *Y, double *Z, int N);

static void benchmark();

int main()
{
        std::ifstream inFile("buffer.dat", std::ios::in|std::ios::binary);
        if (!inFile)
        {
                std::cout << "Run python first\n";
        }
        assert(inFile);
        for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < N; ++j)
                        inFile.read((char*)&X[i*N+j], sizeof(double));
        for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < N; ++j)
                        inFile.read((char*)&Y[i*N+j], sizeof(double));
         for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < N; ++j)
                        inFile.read((char*)&Z_o[i*N+j], sizeof(double));
        inFile.close();

        //Swigling the column major Y matrix so as to access 4 at a time through AVX's
        for (size_t i = 0; i < N; i += 4)
                for (size_t j = 0; j < N; ++j)
                        for (size_t iy = 0; iy < 4; ++iy)
                                Yc[i*N + j*4 + iy] = Y[i*N + j + iy*N];

        printf("\nC Benchmark:\n");

        for (unsigned int i = 0; i < 10; ++i)
        {
                benchmark();
        }
        return 0;
}

static void benchmark()
{
        std::memset(Z, 0, sizeof(Z));

        long long no_of_fmuls = N*N*N;
        long long no_of_fadds = N*N*(N-1);
        long long no_of_totals = no_of_fmuls + no_of_fadds;
        auto start_time = std::chrono::high_resolution_clock::now();

        // naive((double*)X, (double*)Y, (double*)Z, N);
        // simdavx256((double*)X, (double*)Yc, (double*)Z, N);
        // multithreading((double*)X, (double*)Yc, (double*)Z, N);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time-start_time);
        std::cout << std::setw(3) << int(no_of_totals/execution_time.count()) << " GFLOPS\n";

        for (size_t i = 0; i < N; ++i)
        {
                for (size_t j = 0; j < N; ++j)
                {
                        if (std::abs(Z[i*N+j] - Z_o[i*N+j]) > 1e-3)
                        {
                                std::cout << "Mismatch " << "(" << i << "," << j << "): " << Z[i*N+j] << " != " << Z_o[i*N+j] << std::endl;
                                exit(1);
                        }
                }
        }
}