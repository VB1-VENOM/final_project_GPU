#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

void cpuMatMul(const std::vector<float>& A,
               const std::vector<float>& B,
               std::vector<float>& C,
               int M, int K, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i*K + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

int main(int argc, char* argv[])
{
    // --------- RUNTIME SIZE ----------
    int V = 500;               // default
    if (argc > 1)
        V = std::atoi(argv[1]);

    int M = V;
    int K = V;
    int N = V;

    // --------- HOST MEMORY ----------
    std::vector<float> A(M * K), B(K * N), C(M * N);

    for (int i = 0; i < M * K; i++) A[i] = i % 100;
    for (int i = 0; i < K * N; i++) B[i] = 1.0f;

    // --------- TIMING ----------
    auto t1 = std::chrono::high_resolution_clock::now();
    cpuMatMul(A, B, C, M, K, N);
    auto t2 = std::chrono::high_resolution_clock::now();

    double time =
        std::chrono::duration<double>(t2 - t1).count();

    double flops  = 2.0 * M * K * N;
    double gflops = flops / time / 1e9;

    std::cout << "CPU Time: " << time << " s\n";
    std::cout << "CPU GFLOPs: " << gflops << "\n";

    return 0;
}
