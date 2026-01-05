# Running on AMD HPC Fund Hardware

## Build Instructions

Compile the program using `hipcc` with ROCm and OpenMPI libraries:

```bash
hipcc <FILE_NAME> \
  -I/opt/rocm-7.1.0/include \
  -L/opt/rocm-7.1.0/lib \
  -lamdhip64 \
  -I/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/include \
  -L/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/lib \
  -lmpi \
  -lmpi_cxx \
  -o <EXECUTABLE_NAME>

```

## Run Instructions

Use mpirun to launch the executable with the desired number of MPI processes:

```bash
mpirun -np <NUM_PROCESSES> ./<EXECUTABLE_NAME>
```

### Examples:
```bash
mpirun -np 2 ./HW2_mGPU
./cpu 1024
mpirun -np 4 ./main 50
./gpu 1024
mpirun -np 4 ./multipleGPUMatMul 50

```

## Running All Experiments and Collecting Performance Results

To run all configurations and generate performance results in results.csv

```bash
chmod +x run_all.sh
./run_all.sh
```

# Notes

The final hybrid MPI + GPU + SUMMA implementation is located in main.cpp.

Ensure ROCm 7.1.0 and OpenMPI 4.1.8 paths are correctly configured.
