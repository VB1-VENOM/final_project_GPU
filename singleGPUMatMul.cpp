#include <iostream>
#include <vector>
#include <cstdlib>          // for atoi
#include <hip/hip_runtime.h>

#define BLOCK 16

#define gpuErrchk(ans) { if(ans!=hipSuccess){ std::cout<<"HIP Error\n"; exit(0);} }

__global__ void matmulKernel(float* C,
                             const float* A,
                             const float* B,
                             int M,
                             int K,
                             int N)
{
    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int e = 0; e < K; e++)
            sum += A[row * K + e] * B[e * N + col];

        C[row * N + col] = sum;
    }
}

int main(int argc, char* argv[])
{
    // --------- RUNTIME SIZE ----------
    int V = 15000;          // default
    if (argc > 1)
        V = std::atoi(argv[1]);

    int M = V;
    int K = V;
    int N = V;

    // --------- HOST ALLOCATION ----------
    std::vector<float> A(M * K), B(K * N), C(M * N);

    for (int i = 0; i < M * K; i++) A[i] = i % 100;
    for (int i = 0; i < K * N; i++) B[i] = 1.0f;

    // --------- DEVICE ALLOCATION ----------
    float *dA, *dB, *dC;
    gpuErrchk( hipMalloc(&dA, M * K * sizeof(float)) );
    gpuErrchk( hipMalloc(&dB, K * N * sizeof(float)) );
    gpuErrchk( hipMalloc(&dC, M * N * sizeof(float)) );

    gpuErrchk( hipMemcpy(dA, A.data(),
                         M * K * sizeof(float),
                         hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(dB, B.data(),
                         K * N * sizeof(float),
                         hipMemcpyHostToDevice) );

    // --------- KERNEL LAUNCH ----------
    dim3 threads(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK,
              (M + BLOCK - 1) / BLOCK);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    matmulKernel<<<grid, threads>>>(dC, dA, dB, M, K, N);
    gpuErrchk( hipPeekAtLastError() );
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms;
    hipEventElapsedTime(&ms, start, stop);

    gpuErrchk( hipMemcpy(C.data(), dC,
                         M * N * sizeof(float),
                         hipMemcpyDeviceToHost) );

    double time = ms / 1000.0;
    double flops = 2.0 * M * K * N;
    double gflops = flops / time / 1e9;

    std::cout << "GPU Time: " << time << " s\n";
    std::cout << "GPU GFLOPs: " << gflops << "\n";

    hipFree(dA);
    hipFree(dB);
    hipFree(dC);

    return 0;
}
