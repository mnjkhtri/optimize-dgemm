Obtaininig near optimal performance of double general matrix multiplication.

Following optimizations quite unnecessary for modern compilers:
1. Loop unrolling 
2. Inlining functions
3. Register qualifiers

Considerations:
1. Three level of caches L1 (32KiB each), L2 (256KiB each), L3 (8MB shared). While tiling need to pack each blocks on the increasing layer of caches. The trade-off is between movement of cache block across hierarchy and bandwidth of the cache they are placed in. Permutate the combination of block sizes to find the optimal one

2. TLB misses are more critical than I thought so reorder the blocks in contiguous memory wherever possible. Unlike cache misses, the CPU stalls (for certain) at TLB misses. Another thread (same core: hart) can continue while other is stalling due to cache miss. My processor is dynamically issued fetches 4 instructions per cycle, 2 in each hart

3. SIMD instructions especially FMAs triple the performance. Need to swiggle the matrix so they are fetched in contiguous order

4. Outermost layer is multithreaded so there is no sharing of memory between cores. Safe from cache coherency and locking issues. The matrix is calculated in vertical splits by each thread. My CPU is 4 core, 8 threads (intel's hyperthreading)

5. Valgrind's result is worth looking into (D1 misses especially): valgrind --tool=cachegrind ./a.out

Currently achieves a performance of 70ish to 90ish GFLOPS in head-on battle between numpy. Single core performance is somehow slower.

6. Up next: Doing CUDA

References:
https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf

https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf


