import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Load Results
# ------------------------
df = pd.read_csv("results.csv")

N = df["N"]

cpu_gflops = df["CPU_GFLOPs"]
gpu_gflops = df["GPU_GFLOPs"]
mpi_gflops = df["MPI_GFLOPs"]

cpu_time = df["CPU_Time"]
gpu_time = df["GPU_Time"]
mpi_time = df["MPI_Time"]

# ------------------------
# Plot 1: Performance (GFLOPs) vs N
# ------------------------
plt.figure()
plt.plot(N, cpu_gflops, label="CPU")
plt.plot(N, gpu_gflops, label="Single GPU")
plt.plot(N, mpi_gflops, label="MPI + GPU")

plt.xlabel("Matrix Size (N)")
plt.ylabel("Performance (GFLOPs)")
plt.title("Performance Scaling (GFLOPs vs N)")
plt.legend()
plt.grid(True)
plt.savefig("performance_scaling.png")
plt.close()

# ------------------------
# Plot 2: Execution Time vs N
# ------------------------
plt.figure()
plt.plot(N, cpu_time, label="CPU")
plt.plot(N, gpu_time, label="Single GPU")
plt.plot(N, mpi_time, label="MPI + GPU")

plt.xlabel("Matrix Size (N)")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time Scaling")
plt.legend()
plt.grid(True)
plt.savefig("time_scaling.png")
plt.close()

# ------------------------
# Compute Roofline Metrics
# ------------------------
flops = 2 * N**3
bytes_ = 4 * (N*N + N*N + N*N)   # 4*(MK+KN+MN) with M=K=N
intensity = flops / bytes_

# ------------------------
# Roofline Plot
# ------------------------
I = np.logspace(-2, 4, 300)

MEM_BW = 1600        # GB/s  (AMD MI210 HBM2e)
PEAK_GFLOPS = 22600  # GFLOPs (AMD MI210 FP32 peak)

roof = np.minimum(MEM_BW * I, PEAK_GFLOPS)

plt.figure()
plt.loglog(I, roof, label="Roofline")

plt.scatter(intensity, gpu_gflops, label="GPU", marker="o")
plt.scatter(intensity, mpi_gflops, label="MPI+GPU", marker="s")

plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
plt.ylabel("Performance (GFLOPs)")
plt.title("Roofline Model")
plt.legend()
plt.grid(True)
plt.savefig("roofline.png")
plt.close()

# ------------------------
# Done
# ------------------------
print("Plots generated:")
print(" - performance_scaling.png")
print(" - time_scaling.png")
print(" - roofline.png")
