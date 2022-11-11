import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

import numpy
import time

#To use N = 8
N = 512
#max_mem_used = (N*N*3*8)/(1<<20)
#print(f"Memory used in MBs for doubles {max_mem_used}")
#assert(N <= 1024)

#Multiply 4*4 matrices averaging over TIMES times for BENCHMARK
BENCHMARK = 10
if __name__ == "__main__":
    x = numpy.array([[numpy.random.uniform(0, 1e4) for i in range(N)] for j in range(N)])
    #x = numpy.identity(N, dtype='d')
    y = numpy.array([[numpy.random.uniform(0, 1e4) for i in range(N)] for j in range(N)])
    #y = numpy.identity(N, dtype='d')

    #To calculate an element in N by N matrix, a dot product (N multiplies and N-1 additions are required)
    no_of_totals = N*N*N + N*N*(N-1)

    for xx in range(BENCHMARK):
        start_time = time.monotonic()
        z = x@(y.T)
        end_time = time.monotonic()
        execution_time = (end_time - start_time)
        total_flops = (no_of_totals / execution_time)*1e-6
        print(f"{total_flops:.1f} MFLOPS")
    oFile = open("buffer.dat", "wb")
    oFile.write(x)
    oFile.write(y)
    oFile.write(z)
    oFile.close()