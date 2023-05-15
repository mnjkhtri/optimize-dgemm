import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1" 
# os.environ["NUMEXPR_NUM_THREADS"] = "1" 

import numpy
import time

N = 512

#Multiply 4*4 matrices averaging over TIMES times for BENCHMARK
BENCHMARK = 10
if __name__ == "__main__":
    x = numpy.array([[numpy.random.uniform(0, 1e4) for i in range(N)] for j in range(N)])
    y = numpy.array([[numpy.random.uniform(0, 1e4) for i in range(N)] for j in range(N)])
    
    no_of_totals = N*N*N + N*N*(N-1)

    print("Python Benchmark:")
    for xx in range(BENCHMARK):
        start_time = time.monotonic()
        z = x@(y.T)
        end_time = time.monotonic()
        execution_time = (end_time - start_time)
        total_flops = (no_of_totals / execution_time)*1e-9
        print(f"{total_flops:3.0f} GFLOPS")
    oFile = open("buffer.dat", "wb")
    oFile.write(x)
    oFile.write(y)
    oFile.write(z)
    oFile.close()

    #The C program must read these matrices before multiplying for itself for verfying purpose