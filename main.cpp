//"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64"

//g++ -O3 -Wall -march=native main.cpp "opt8(packing).cpp"

#include <iostream>
#include <assert.h>
#include <fstream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstring>

#define CHECK

//To multiply X[N*N] x Y[N*N]
const size_t N = 1024;

// double X[N*N];
// double Y[N*N];
// double Z[N*N];
// double Z_o[N*N];

//Alignment required for vector accesses (opt6) [probably 64 bytes alignment for cache blocks]
alignas(64) double X[N*N];
alignas(64) double Y[N*N];
alignas(64) double Z[N*N];
alignas(64) double Z_o[N*N];

extern void naive(double *X, double *Y, double *Z, size_t N);
extern void unrolling(double *X, double *Y, double *Z, size_t N);
extern void inlining(double *X, double *Y, double *Z, size_t N);
extern void registers(double *X, double *Y, double *Z, size_t N);
extern void more(double *X, double *Y, double *Z, size_t N);
extern void vectors(double *X, double *Y, double *Z, size_t N);
extern void blocking(double *X, double *Y, double *Z, size_t N);
extern void packing(double *X, double *Y, double *Z, size_t N);
extern void finalboss(double *X, double *Y, double *Z, size_t N);

//Function to bench the MFLOPS
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


        for (unsigned int i = 0; i < 1; ++i)
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

        //Call your function here:
        //naive((double*)X, (double*)Y, (double*)Z, N);
        //unrolling((double*)X, (double*)Y, (double*)Z, N);
        //inlining((double*)X, (double*)Y, (double*)Z, N);
        //registers((double*)X, (double*)Y, (double*)Z, N);
        //more((double*)X, (double*)Y, (double*)Z, N);
        //vectors((double*)X, (double*)Y, (double*)Z, N);
        //blocking((double*)X, (double*)Y, (double*)Z, N);
        //packing((double*)X, (double*)Y, (double*)Z, N);
        finalboss((double*)X, (double*)Y, (double*)Z, N);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time-start_time);
        std::cout << std::setw(8) << no_of_totals*1e3/execution_time.count() << "    MFLOPS: ";

        #ifdef CHECK

        for (size_t i = 0; i < N; ++i)
        {
                for (size_t j = 0; j < N; ++j)
                {
                        if (std::abs(Z[i*N+j] - Z_o[i*N+j]) > 1e-3)
                        {
                                std::cout << "Mismatch " << "(" << i << "," << j << ")" << Z[i*N+j] << " != " << Z_o[i*N+j] << std::endl;
                                return;
                        }
                }
        }
        std::cout << "Match\n";

        #endif
}