// #include <iostream>
// #include <vector>
// #include <mpi.h>
// #include <hip/hip_runtime.h>
// #define BLOCK 16

// int V = 15000;           // default size
// if (argc > 1)
//     V = atoi(argv[1]);  // override from command line

// #define M V
// #define K V
// #define N V


// __global__ void matmulKernel(float* C,const float* A,const float* B,int rows)
// {
//     int row = blockIdx.y*BLOCK + threadIdx.y;
//     int col = blockIdx.x*BLOCK + threadIdx.x;

//     if(row<rows && col<N)
//     {
//         float sum = 0;
//         for(int k=0;k<K;k++)
//             sum += A[row*K+k]*B[k*N+col];
//         C[row*N+col] = sum;
//     }
// }

// int main(int argc,char** argv)
// {
//     MPI_Init(&argc,&argv);

//     int rank,size;
//     MPI_Comm_rank(MPI_COMM_WORLD,&rank);
//     MPI_Comm_size(MPI_COMM_WORLD,&size);

//     int rows = M/size;

//     std::vector<float> A_local(rows*K);
//     std::vector<float> B(K*N);
//     std::vector<float> C_local(rows*N);
//     std::vector<float> A, C;

//     if(rank==0)
//     {
//         A.resize(M*K); C.resize(M*N);
//         for(int i=0;i<M*K;i++) A[i]=i%100;
//         for(int i=0;i<K*N;i++) B[i]=1.0f;
//     }

//     MPI_Scatter(A.data(),rows*K,MPI_FLOAT,
//                 A_local.data(),rows*K,MPI_FLOAT,0,MPI_COMM_WORLD);

//     MPI_Bcast(B.data(),K*N,MPI_FLOAT,0,MPI_COMM_WORLD);

//     float *dA,*dB,*dC;
//     hipMalloc(&dA,rows*K*sizeof(float));
//     hipMalloc(&dB,K*N*sizeof(float));
//     hipMalloc(&dC,rows*N*sizeof(float));

//     hipMemcpy(dA,A_local.data(),rows*K*sizeof(float),
//               hipMemcpyHostToDevice);
//     hipMemcpy(dB,B.data(),K*N*sizeof(float),
//               hipMemcpyHostToDevice);

//     dim3 threads(BLOCK,BLOCK);
//     dim3 grid((N+BLOCK-1)/BLOCK,(rows+BLOCK-1)/BLOCK);

//     MPI_Barrier(MPI_COMM_WORLD);
//     double t1 = MPI_Wtime();

//     matmulKernel<<<grid,threads>>>(dC,dA,dB,rows);
//     hipDeviceSynchronize();

//     MPI_Barrier(MPI_COMM_WORLD);
//     double t2 = MPI_Wtime();

//     double local = t2-t1;
//     double total;

//     MPI_Reduce(&local,&total,1,MPI_DOUBLE,
//                MPI_MAX,0,MPI_COMM_WORLD);

//     hipMemcpy(C_local.data(),dC,rows*N*sizeof(float),
//               hipMemcpyDeviceToHost);

//     MPI_Gather(C_local.data(),rows*N,MPI_FLOAT,
//                C.data(),rows*N,MPI_FLOAT,
//                0,MPI_COMM_WORLD);

//     if(rank==0)
//     {
//         double flops = 2.0*M*K*N;
//         double gflops = flops/total/1e9;
//         std::cout<<"MPI+GPU Time: "<<total<<" s\n";
//         std::cout<<"MPI+GPU GFLOPs: "<<gflops<<"\n";
//     }

//     hipFree(dA); hipFree(dB); hipFree(dC);
//     MPI_Finalize();
// }
#include <iostream>
#include <vector>
#include <cstdlib>
#include <mpi.h>
#include <hip/hip_runtime.h>

#define BLOCK 16

#define gpuErrchk(ans) { if(ans!=hipSuccess){ \
    std::cerr << "HIP Error: " << hipGetErrorString(ans) << std::endl; exit(1);} }

__global__ void matmulKernel(float* C,
                             const float* A,
                             const float* B,
                             int rows,
                             int K,
                             int N)
{
    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;

    if (row < rows && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];

        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv)
{
    // ---------------- MPI INIT ----------------
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---------------- RUNTIME MATRIX SIZE ----------------
    int V = 15000;                // default
    if (argc > 1)
        V = std::atoi(argv[1]);   // override from script

    int M = V;
    int K = V;
    int N = V;

    int rows = M / size;

    // ---------------- HOST MEMORY ----------------
    std::vector<float> A_local(rows * K);
    std::vector<float> B(K * N);
    std::vector<float> C_local(rows * N);
    std::vector<float> A, C;

    if (rank == 0)
    {
        A.resize(M * K);
        C.resize(M * N);

        for (int i = 0; i < M * K; i++) A[i] = i % 100;
        for (int i = 0; i < K * N; i++) B[i] = 1.0f;
    }

    // ---------------- MPI DISTRIBUTION ----------------
    MPI_Scatter(A.data(), rows * K, MPI_FLOAT,
                A_local.data(), rows * K, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    MPI_Bcast(B.data(), K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // ---------------- DEVICE MEMORY ----------------
    float *dA, *dB, *dC;

    gpuErrchk( hipMalloc(&dA, rows * K * sizeof(float)) );
    gpuErrchk( hipMalloc(&dB, K * N * sizeof(float)) );
    gpuErrchk( hipMalloc(&dC, rows * N * sizeof(float)) );

    gpuErrchk( hipMemcpy(dA, A_local.data(),
                         rows * K * sizeof(float),
                         hipMemcpyHostToDevice) );

    gpuErrchk( hipMemcpy(dB, B.data(),
                         K * N * sizeof(float),
                         hipMemcpyHostToDevice) );

    // ---------------- GPU KERNEL ----------------
    dim3 threads(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK,
              (rows + BLOCK - 1) / BLOCK);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    matmulKernel<<<grid, threads>>>(dC, dA, dB, rows, K, N);
    gpuErrchk( hipDeviceSynchronize() );

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    double local = t2 - t1;
    double total;

    MPI_Reduce(&local, &total, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    // ---------------- COPY BACK ----------------
    gpuErrchk( hipMemcpy(C_local.data(), dC,
                         rows * N * sizeof(float),
                         hipMemcpyDeviceToHost) );

    MPI_Gather(C_local.data(), rows * N, MPI_FLOAT,
               C.data(), rows * N, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    // ---------------- OUTPUT ----------------
    if (rank == 0)
    {
        double flops  = 2.0 * M * K * N;
        double gflops = flops / total / 1e9;

        std::cout << "MPI+GPU Time: " << total << " s\n";
        std::cout << "MPI+GPU GFLOPs: " << gflops << "\n";
    }

    // ---------------- CLEANUP ----------------
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);

    MPI_Finalize();
    return 0;
}
