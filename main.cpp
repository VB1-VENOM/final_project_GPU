#include <iostream>
#include <mpi.h>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <hip/hip_runtime.h>
#define BLOCK 16

#define gpuErrchk(ans) { if(ans!=hipSuccess){ \
    fprintf(stderr,"HIP Error %s:%d: %s\n",__FILE__,__LINE__, hipGetErrorString(ans)); exit(1);} }


__global__ void matmulKernel(double* C,
                             const double* A,
                             const double* B,
                             int n_local,
                             int k_local,
                             int m_local)
{
    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;

    if(row < n_local && col < m_local)
    {
        double sum = 0.0;

        for(int k = 0; k < k_local; k++)
            sum += A[row * k_local + k] * B[k * m_local + col];

        C[row * m_local + col] += sum;
    }
}


std::vector<double> init_matrix(int rows, int cols, int seed=42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> d(1.0, 10.0);
    std::vector<double> M(rows*cols);
    for(size_t i=0;i<M.size();++i) M[i] = d(gen);
    return M;
}

void local_gemm(const double* A, const double* B, double* C,
                int nr, int kr, int mc) {
    for (int i = 0; i < nr; ++i) {
        for (int k = 0; k < kr; ++k) {
            double a = A[i*kr + k];
            for (int j = 0; j < mc; ++j) {
                C[i*mc + j] += a * B[k*mc + j];
            }
        }
    }
}

void gpu_gemm(const std::vector<double>& A_panel,
              const std::vector<double>& B_panel,
              std::vector<double>& C_local,
              int n_local,
              int k_local,
              int m_local)
{
    double *dA, *dB, *dC;

    size_t sizeA = n_local * k_local * sizeof(double);
    size_t sizeB = k_local * m_local * sizeof(double);
    size_t sizeC = n_local * m_local * sizeof(double);

    gpuErrchk( hipMalloc(&dA, sizeA) );
    gpuErrchk( hipMalloc(&dB, sizeB) );
    gpuErrchk( hipMalloc(&dC, sizeC) );

    gpuErrchk( hipMemcpy(dA, A_panel.data(), sizeA, hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(dB, B_panel.data(), sizeB, hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(dC, C_local.data(), sizeC, hipMemcpyHostToDevice) );

    dim3 threads(BLOCK, BLOCK);
    dim3 grid((m_local + BLOCK - 1)/BLOCK,
              (n_local + BLOCK - 1)/BLOCK);

    matmulKernel<<<grid, threads>>>(dC, dA, dB,
                                    n_local, k_local, m_local);

    gpuErrchk( hipDeviceSynchronize() );

    gpuErrchk( hipMemcpy(C_local.data(), dC, sizeC, hipMemcpyDeviceToHost) );

    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
}


double SUMMA(int K, int N, int M_){
    auto start = std::chrono::high_resolution_clock::now();
    double t0, t1;


    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int numDevices = 0;
    hipGetDeviceCount(&numDevices);
    hipSetDevice(world_rank % numDevices);

 

    int q = (int)std::round(std::sqrt((double)world_size));
    if (q*q != world_size) {
        if (world_rank == 0) std::cerr << "Error: run with P = q^2 processes (perfect square)\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (N % q != 0 || K % q != 0 || M_ % q != 0) {
        if (world_rank == 0) std::cerr << "Error: N, K, M must be divisible by q (for this simple impl)\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    //create the process grid:
    int dims[2] = {q, q};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    //gets THIS processes grid specifically:
    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);
    int my_row = coords[0], my_col = coords[1];

    int remain_row[2] = {0, 1}; 
    int remain_col[2] = {1, 0}; 
    MPI_Comm row_comm, col_comm;
    MPI_Cart_sub(cart_comm, remain_row, &row_comm);
    MPI_Cart_sub(cart_comm, remain_col, &col_comm);

    int n_local = N / q;
    int k_local = K / q;
    int m_local = M_ / q;

    // Buffers each process will hold: A_local (n_local x k_local),
    // B_local (k_local x m_local), C_local (n_local x m_local)
    std::vector<double> A_local(n_local * k_local, 0.0);
    std::vector<double> B_local(k_local * m_local, 0.0);
    std::vector<double> C_local(n_local * m_local, 0.0);

    // Rank 0 initializes full matrices and scatters blocks
    std::vector<double> A_full, B_full, C_full;
    if (world_rank == 0) {
        A_full = init_matrix(N, K, 123);
        B_full = init_matrix(K, M_, 456);
        C_full.assign(N * M_, 0.0);
    }

    auto pack_block = [&](const std::vector<double>& full, int full_rows, int full_cols,
                          int bi, int bj, int brow, int bcol, std::vector<double>& out) {
        out.resize(brow * bcol);
        for (int r = 0; r < brow; ++r) {
            int src_row = bi * brow + r;
            for (int c = 0; c < bcol; ++c) {
                int src_col = bj * bcol + c;
                out[r*bcol + c] = full[src_row * full_cols + src_col];
            }
        }
    };

    if (world_rank == 0) {
        std::vector<double> tmp;
        for (int i = 0; i < q; ++i) {
            for (int j = 0; j < q; ++j) {
                int dest_coords[2] = {i, j};
                int dest_rank;
                MPI_Cart_rank(cart_comm, dest_coords, &dest_rank);
                // A block A_{i,j} of size n_local x k_local
                pack_block(A_full, N, K, i, j, n_local, k_local, tmp);
                if (dest_rank == 0) {
                    A_local = tmp;
                } else {
                    MPI_Send(tmp.data(), (int)tmp.size(), MPI_DOUBLE, dest_rank, 100 + 0, MPI_COMM_WORLD);
                }
                // B block B_{i,j} of size k_local x m_local
                pack_block(B_full, K, M_, i, j, k_local, m_local, tmp);
                if (dest_rank == 0) {
                    B_local = tmp;
                } else {
                    MPI_Send(tmp.data(), (int)tmp.size(), MPI_DOUBLE, dest_rank, 200 + 0, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        MPI_Status st;
        MPI_Recv(A_local.data(), n_local*k_local, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
        MPI_Recv(B_local.data(), k_local*m_local, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
    }

    // SUMMA main loop
    std::vector<double> A_panel(n_local * k_local); // buffer for broadcasted A_{i,k}
    std::vector<double> B_panel(k_local * m_local); // buffer for broadcasted B_{k,j}
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    for (int k = 0; k < q; ++k) {
        
        int root_col = k;
        int root_rank_in_row = root_col;
        if (my_col == k) {
            A_panel = A_local; 
        }
        MPI_Bcast(A_panel.data(), n_local * k_local, MPI_DOUBLE, root_rank_in_row, row_comm);

        int root_row = k;
        int root_rank_in_col = root_row; 
        if (my_row == k) {
            B_panel = B_local; 
        }
        MPI_Bcast(B_panel.data(), k_local * m_local, MPI_DOUBLE, root_rank_in_col, col_comm);

        //Varun replace this using your GPU based matrix multiplication
        //you can first start with just one GPU and multiple threads
        //then try using multiple GPUs on the amd node, hipSetDevice(world_rank) should do the trick;

        //local_gemm(A_panel.data(), B_panel.data(), C_local.data(), n_local, k_local, m_local);
        gpu_gemm(A_panel, B_panel, C_local, n_local, k_local, m_local);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double global_time = 0.0;

    MPI_Reduce(
        &local_time,
        &global_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );



    //gather the relevant parts from around the ranks
    if (world_rank == 0) {
        std::vector<double> tmp;
        for (int i = 0; i < q; ++i) {
            for (int j = 0; j < q; ++j) {
                int src_coords[2] = {i, j};
                int src_rank;
                MPI_Cart_rank(cart_comm, src_coords, &src_rank);
                if (src_rank == 0) {
                    for (int r = 0; r < n_local; ++r) {
                        int dst_row = i*n_local + r;
                        for (int c = 0; c < m_local; ++c) {
                            int dst_col = j*m_local + c;
                            C_full[dst_row * M_ + dst_col] = C_local[r*m_local + c];
                        }
                    }
                } else {
                    tmp.resize(n_local * m_local);
                    MPI_Recv(tmp.data(), (int)tmp.size(), MPI_DOUBLE, src_rank, 300, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (int r = 0; r < n_local; ++r) {
                        int dst_row = i*n_local + r;
                        for (int c = 0; c < m_local; ++c) {
                            int dst_col = j*m_local + c;
                            C_full[dst_row * M_ + dst_col] = tmp[r*m_local + c];
                        }
                    }
                }
            }
        }
    } else {
        MPI_Send(C_local.data(), n_local*m_local, MPI_DOUBLE, 0, 300, MPI_COMM_WORLD);
    }


    double result = 0.0;
    if (world_rank == 0) {
        result = global_time;
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);

    return result;

}

void matmul_cpu(const std::vector<double>& A,
                const std::vector<double>& B,
                std::vector<double>& C,
                int N)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

void fill_random(std::vector<double>& M) {
    static std::mt19937 rng(123);
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& x : M) x = dist(rng);
}

double run_cpu_baseline(int N)
{
    std::vector<double> A(N*N), B(N*N), C(N*N);
    fill_random(A);
    fill_random(B);

    auto start = std::chrono::high_resolution_clock::now();
    matmul_cpu(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    return ms;
}

//run main using:
//mpiexec -n 16 .\Release\SUMMA.exe 10000
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 2) {
        if (world_rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
        }
        MPI_Finalize();
        return 1;
    }

    int N = std::atoi(argv[1]);

    double summa_time = SUMMA(N, N, N);  //Varun this is the main function that does the summa multiply

    if (world_rank == 0) {
        std::cout << "Matrix size N = " << N
                  << ", MPI SUMMA time = " << summa_time << " s\n";
    }

    MPI_Finalize();
    return 0;
}