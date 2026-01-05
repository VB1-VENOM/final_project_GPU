# To run on AMD HPC fund hardware:

## Build: 
hipcc <FILE_NAME>     -I/opt/rocm-7.1.0/include     -L/opt/rocm-7.1.0/lib     -lamdhip64     -I/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/include     -L/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/lib     -lmpi     -lmpi_cxx     -o <EXECUTABLE_NAME>

Ex: hipcc HW2_multipleGPU.cpp     -I/opt/rocm-7.1.0/include     -L/opt/rocm-7.1.0/lib     -lamdhip64     -I/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/include     -L/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/lib     -lmpi     -lmpi_cxx     -o HW2_mGPU
## Run:
mpirun -np 2 ./<EXECUTABLE_NAME>

Examples: 
mpirun -np 2 ./HW2_mGPU

./cpu 1024

mpirun -np 4 ./main 50

./gpu 1024

mpirun -np 4 ./multipleGPUMatMul 50

# To run all and get the performance in result.csv:

chmod +x run_all.sh

./run_all.sh

