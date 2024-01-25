// __global__ void atb(const double *A, const double *B, double *C, int Ni, int Nj, int Nk) {

//     // Get the row and column indices of the matrix C element being processed
//     int row = blockIdx.y*blockDim.y + threadIdx.y;
//     int col = blockIdx.x*blockDim.x + threadIdx.x;

//     // Check if the indices are within the bounds of the matrix C
//     if (row < Ni && col < Nj)
//     {
//         double value = 0;
//         for (int k = 0; k < Nk; k++)
//         {
//             //value += A[row*Nk + k] * B[k*Nj + col];
//             value += A[k*Ni + row] * B[k*Nj + col];
//         }
//         C[row*Nj + col] = value;
//         //C[col*Ni+row] = value;
//     }
// }


__global__ void atb(const double *A, const double *B, double *C, int Ni, int Nj, int Nk)
{   int BLOCK_SIZE = 8;
    __shared__ double mat_1_tile[8][8];
    __shared__ double mat_2_tile[8][8];

    double acc_sum{0};
    int temp = ceil((double)Nk / BLOCK_SIZE);
    for (int tile_idx = 0; tile_idx < temp; ++tile_idx)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = tile_idx * blockDim.x + threadIdx.x;
        if ((i < Ni) && (j < Nk))
        {
            mat_1_tile[threadIdx.y][threadIdx.x] = A[j * Ni + i];
            //mat_1_tile[threadIdx.x][threadIdx.y] = A[i * Nk + j];
        }
        else
        {
            mat_1_tile[threadIdx.y][threadIdx.x] = 0;
        }
        i = tile_idx * blockDim.y + threadIdx.y;
        j = blockIdx.x * blockDim.x + threadIdx.x;
        if ((i < Nk) && (j < Nj))
        {
            mat_2_tile[threadIdx.y][threadIdx.x] = B[i * Nj + j];
        }
        else
        {
            mat_2_tile[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            acc_sum += mat_1_tile[threadIdx.y][k] * mat_2_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    // 2D block and 2D thread
    // Each thread computes one cell in C.
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i < Ni) && (j < Nj))
    {
        C[i * Nj + j] = acc_sum;
    }
}
