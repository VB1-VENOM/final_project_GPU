#include <iostream>
#include <vector>
#include <mpi.h>
#include <hip/hip_runtime.h>

#define M_GLOBAL 512
#define K_GLOBAL 512
#define N_GLOBAL 512
#define BLOCK_SIZE 16

// ---------------- GPU ERROR CHECK ----------------
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line)
{
    if (code != hipSuccess)
    {
        std::cerr << "HIP error: " << hipGetErrorString(code)
                  << " at " << file << ":" << line << std::endl;
        exit(code);
    }
}

// ---------------- SIMPLE HIP KERNEL ----------------
__global__ void matmulKernel(float* C, const float* A, const float* B,
                             int M, int K, int N)
{
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int e = 0; e < K; e++)
            sum += A[row * K + e] * B[e * N + col];

        C[row * N + col] = sum;
    }
}

// ---------------- GPU WRAPPER ----------------
void computeMM(const float* d_A, const float* d_B, float* d_C,
               int M, int K, int N)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmulKernel<<<grid, threads>>>(d_C, d_A, d_B, M, K, N);
    gpuErrchk( hipPeekAtLastError() );
    gpuErrchk( hipDeviceSynchronize() );
}

// ---------------- MAIN PROGRAM ----------------
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---------------- HOST MATRICES ----------------
    int rows_per_rank = M_GLOBAL / size;

    std::vector<float> A_local(rows_per_rank * K_GLOBAL);
    std::vector<float> B(K_GLOBAL * N_GLOBAL);
    std::vector<float> C_local(rows_per_rank * N_GLOBAL);
    std::vector<float> C_global;

    if (rank == 0)
    {
        std::vector<float> A_global(M_GLOBAL * K_GLOBAL);
        C_global.resize(M_GLOBAL * N_GLOBAL);

        // Init A
        for (int i = 0; i < M_GLOBAL * K_GLOBAL; i++)
            A_global[i] = static_cast<float>(i % 100);

        // Init B
        for (int i = 0; i < K_GLOBAL * N_GLOBAL; i++)
            B[i] = 1.0f;

        // Scatter A
        MPI_Scatter(A_global.data(),
                    rows_per_rank * K_GLOBAL,
                    MPI_FLOAT,
                    A_local.data(),
                    rows_per_rank * K_GLOBAL,
                    MPI_FLOAT,
                    0,
                    MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatter(nullptr,
                    rows_per_rank * K_GLOBAL,
                    MPI_FLOAT,
                    A_local.data(),
                    rows_per_rank * K_GLOBAL,
                    MPI_FLOAT,
                    0,
                    MPI_COMM_WORLD);
    }

    // Broadcast B
    MPI_Bcast(B.data(),
              K_GLOBAL * N_GLOBAL,
              MPI_FLOAT,
              0,
              MPI_COMM_WORLD);

    // ---------------- DEVICE MEMORY ----------------
    float *d_A, *d_B, *d_C;
    gpuErrchk( hipMalloc(&d_A, rows_per_rank * K_GLOBAL * sizeof(float)) );
    gpuErrchk( hipMalloc(&d_B, K_GLOBAL * N_GLOBAL * sizeof(float)) );
    gpuErrchk( hipMalloc(&d_C, rows_per_rank * N_GLOBAL * sizeof(float)) );

    gpuErrchk( hipMemcpy(d_A, A_local.data(),
                         rows_per_rank * K_GLOBAL * sizeof(float),
                         hipMemcpyHostToDevice) );

    gpuErrchk( hipMemcpy(d_B, B.data(),
                         K_GLOBAL * N_GLOBAL * sizeof(float),
                         hipMemcpyHostToDevice) );

    // ---------------- GPU COMPUTE ----------------
    computeMM(d_A, d_B, d_C,
              rows_per_rank, K_GLOBAL, N_GLOBAL);

    // ---------------- COPY BACK ----------------
    gpuErrchk( hipMemcpy(C_local.data(), d_C,
                         rows_per_rank * N_GLOBAL * sizeof(float),
                         hipMemcpyDeviceToHost) );

    // ---------------- GATHER RESULT ----------------
    MPI_Gather(C_local.data(),
               rows_per_rank * N_GLOBAL,
               MPI_FLOAT,
               C_global.data(),
               rows_per_rank * N_GLOBAL,
               MPI_FLOAT,
               0,
               MPI_COMM_WORLD);

    // ---------------- PRINT SMALL RESULT ----------------
    if (rank == 0)
    {
        std::cout << "Matrix Multiplication Completed.\n";
        std::cout << "C[0][0] = " << C_global[0] << std::endl;
    }

    // ---------------- CLEANUP ----------------
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    MPI_Finalize();
    return 0;
}
