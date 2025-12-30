#!/bin/bash

SIZES=(50 100 200 1024 2048 4096 8192 12000 15000)

echo "N,CPU_Time,GPU_Time,MPI_Time,CPU_GFLOPs,GPU_GFLOPs,MPI_GFLOPs" > results.csv

for V in "${SIZES[@]}"
do
    echo "Running for N=$V"

    # ---------------- CPU (ONLY up to 4096) ----------------
    if [ "$V" -le 1024 ]; then
        CPU_OUT=$(./cpu $V)
        CPU_TIME=$(echo "$CPU_OUT" | grep "Time" | awk '{print $3}')
        CPU_GFLOPs=$(echo "$CPU_OUT" | grep "GFLOPs" | awk '{print $3}')
    else
        CPU_TIME="NA"
        CPU_GFLOPs="NA"
    fi
    #echo "CPU Done"
    # ---------------- GPU ----------------
    GPU_OUT=$(./gpu $V)
    GPU_TIME=$(echo "$GPU_OUT" | grep "Time" | awk '{print $3}')
    GPU_GFLOPs=$(echo "$GPU_OUT" | grep "GFLOPs" | awk '{print $3}')

    # ---------------- MPI + GPU ----------------
    MPI_OUT=$(mpirun -np 4 ./multipleGPUMatMul $V)
    MPI_TIME=$(echo "$MPI_OUT" | grep "Time" | awk '{print $3}')
    MPI_GFLOPs=$(echo "$MPI_OUT" | grep "GFLOPs" | awk '{print $3}')

    # ---------------- MPI + GPU + SUMMA ----------------
    MPI_OUT=$(mpirun -np 4 ./main $V)
    MPI_TIME=$(echo "$MPI_OUT" | grep "Time" | awk '{print $3}')
    MPI_GFLOPs=$(echo "$MPI_OUT" | grep "GFLOPs" | awk '{print $3}')



    # ---------------- SAVE TO CSV ----------------
    echo "$V,$CPU_TIME,$GPU_TIME,$MPI_TIME,$CPU_GFLOPs,$GPU_GFLOPs,$MPI_GFLOPs" >> results.csv
done


echo "All runs completed. Results saved in results.csv"

# To run:
# chmod +x run_all.sh
# ./run_all.sh
