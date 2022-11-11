__m256 *Am = (__m256*)A;
__m256 *Bm = (__m256*)B;
__m256 *Cm = (__m256*)C;

float Bf[N*N] __attribute__ ((aligned (64)));
__m256 *Bfm = (__m256*)Bf;

#define BLOCK 8
#define BLOCK_Y 4
#define BLOCK_X 2

void matmul(0, N) {

  divides the matrix into 4 by 16 size sub matrix and loop on them

      //this will accumulate the data to go into the 4 by 16 submatrix
      acc is 4 by 2 matrix of 256 data type

      for (int k = 0; k < N; k++) {

          ta = element (0, k) element of submatrix we are in broadcasted 8 times

          acc[0][0] += ta * Bfm[((x+0*BLOCK)*N + k*8)/8];
          acc[0][1] += ta * Bfm[((x+1*BLOCK)*N + k*8)/8];

          ta = element (1, k) element of submatrix we are in broadcasted 8 times

          acc[1][0] += ta * Bfm[((x+0)*N + k*8)/8]; 
          acc[1][1] += ta * Bfm[((x+8)*N + k*8)/8];

          ta = element (2, k) element of submatrix we are in broadcasted 8 times

          acc[2][0] += ta * Bfm[((x+0)*N + k*8)/8];
          acc[2][1] += ta * Bfm[((x+8)*N + k*8)/8];

          ta = element (3, k) element of submatrix we are in broadcasted 8 times

          acc[3][0] += ta * Bfm[((x+0)*N + k*8)/8];
          acc[3][1] += ta * Bfm[((x+1)*N + k*8)/8];
          }
      
      write the accumulator data  to the submatrix

  // preswizzle
  for (int y = 0; y < N; y+=8) {
    for (int x = 0; x < N; x++) {
      for (int iy = 0; iy < 8; iy++) {
        Bf[y*N + x*8 + iy] = B[(y+iy)*N + x];
      }
    }
  }
}
