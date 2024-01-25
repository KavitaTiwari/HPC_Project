void atb_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

#pragma omp parallel private(i,j,k)
 {
  #pragma omp for 
  for (i = 0; i < Ni; i++){
	  int rem = Nk%4;
    for (k = 0; k < rem; k++){
      for (j = 0; j < Nj; j++){
	     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
      }
    }
    for (k = rem; k < Nk; k+=4){
      for (j = 0; j < Nj; j++){
	     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
	     C[i*Nj+j]=C[i*Nj+j]+A[(k+1)*Ni+i]*B[(k+1)*Nj+j];
	     C[i*Nj+j]=C[i*Nj+j]+A[(k+2)*Ni+i]*B[(k+2)*Nj+j];
	     C[i*Nj+j]=C[i*Nj+j]+A[(k+3)*Ni+i]*B[(k+3)*Nj+j];
      }
    }
  }
 }
}

