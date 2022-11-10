import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

import numpy
import time

#To use N = 1024
N = 1024 
#max_mem_used = (N*N*3*8)/(1<<20)
#print(f"Memory used in MBs for doubles {max_mem_used}")
#assert(N <= 1024)

#Multiply 4*4 matrices averaging over TIMES times for BENCHMARK
BENCHMARK = 1000

x = numpy.array([[numpy.random.uniform(0, 1e4) for i in range(N)] for j in range(N)])
#x = numpy.identity(N, dtype='d')
y = numpy.array([[numpy.random.uniform(0, 1e4) for i in range(N)] for j in range(N)])
#y = numpy.identity(N, dtype='d')

#To calculate an element in N by N matrix, a dot product (N multiplies and N-1 additions are required)
no_of_fmuls = N*N*N
no_of_fadds = N*N*(N-1)
no_of_totals = no_of_fmuls + no_of_fadds

for xx in range(BENCHMARK):
    start_time = time.monotonic_ns()
    z = x@y
    end_time = time.monotonic_ns()
    oFile = open("buffer.dat", "wb")
    oFile.write(x)
    oFile.write(y)
    oFile.write(z)
    oFile.close()
    execution_time = (end_time - start_time)
    assert(execution_time != 0)
    total_flops = no_of_totals*1e3 / execution_time
    print(f"{total_flops:.2f} MFLOPS")