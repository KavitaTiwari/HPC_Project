void atbt_par(const double *__restrict__ A, const double *__restrict__ B, double 
*__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k, i_tile, j_tile, k_tile;
  int R, UF = 4;
  const int tile_size = 128;

#pragma omp parallel private(i, j, k, i_tile, j_tile, k_tile)
{
  for (int i_tile = 0; i_tile < Ni; i_tile += tile_size) {
    int i_tile_end = i_tile + tile_size;
    i_tile_end = (i_tile_end > Ni) ? Ni : i_tile_end;
    for (int j_tile = 0; j_tile < Nj; j_tile += tile_size) {
      int j_tile_end = j_tile + tile_size;
      j_tile_end = (j_tile_end > Nj) ? Nj : j_tile_end;
      #pragma omp for
      for (int i = i_tile; i < i_tile_end; ++i) {
        for (int k = 0; k < Nk; ++k) {
          for (int j = j_tile; j < j_tile_end; ++j) {
            C[i*Nj+j] += A[k*Ni+i]*B[j*Nk+k];
          }
        }
      }
    }
  }
}
}
