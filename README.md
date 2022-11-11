Observations so far:

Following quite unnecessary for modern compilers
1. Loop unrolling 
2. Inlining functions
3. Register qualifiers

Must's:
1. Second matrix in column contiguous order
2. The order swiggled into blocks of 4 to accomodate AVX2 instructions
3. Intrins ymm's with O3 or masm
4. Calculate elements in blocks that fit in L1 cache (32 KB, in my case)
5. Another layer of blocking when matrices dont fit in LL caches (L2: 256 KB, L3: 8MB)
6. For extra blocking, packing is necessary for optimal performance
7. Packing also paves path for multithreading
8. Valgrind's result is worth looking into (D1 misses especially)
