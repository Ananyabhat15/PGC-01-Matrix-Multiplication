# Matrix Multiplication using Sequential, OpenMP, MPI and CUDA

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objective](#2-objective)
3. [Computing Models at a Glance](#3-computing-models-at-a-glance)
4. [Problem Definition](#4-problem-definition)
5. [Components Required](#5-components-required)
6. [Project Structure](#6-project-structure)
7. [Part A: Sequential Matrix Multiplication](#7-part-a--sequential-matrix-multiplication)
8. [Part B: OpenMP Matrix Multiplication](#8-part-b--openmp-matrix-multiplication)
9. [Part C: MPI Distributed Matrix Multiplication](#9-part-c--mpi-distributed-matrix-multiplication)
10. [Part D: CUDA Matrix Multiplication](#10-part-d--cuda-matrix-multiplication)
11. [Results and Performance Comparison](#11-results-and-performance-comparison)
12. [Recommended Result Screenshots](#12-recommended-result-screenshots)
13. [Conclusion](#13-conclusion)

---

## 1. Project Overview

This laboratory experiment implements the **same 4000 × 4000 matrix multiplication** using four different computing models and compares their execution time and speedup.

The sequential program is executed first and serves as the baseline. The same computation is then:

- parallelized with **OpenMP** on a shared-memory CPU (WSL2 Ubuntu),
- distributed with **MPI** across four Ubuntu virtual machines, and
- accelerated with **CUDA** on an NVIDIA GPU.

For the sequential and OpenMP parts, Windows PowerShell is used to access the installed WSL2 Ubuntu environment; the C programs are compiled and executed inside Ubuntu.

### Execution Flow

```
Windows PowerShell
      ↓
WSL2 Ubuntu
      ↓
Sequential CPU Baseline
      ↓
OpenMP Shared Memory
      ↓
MPI Distributed Memory
      ↓
CUDA GPU Parallelism
      ↓
Results and Speedup Comparison
```

## 2. Objective

- Implement a 4000 × 4000 matrix multiplication using sequential, OpenMP, MPI and CUDA programs.
- Establish the sequential execution time as the performance baseline.
- Demonstrate shared-memory (OpenMP), distributed-memory (MPI) and GPU (CUDA) parallelism on the same workload.
- Verify that all four implementations produce the same result (`C[0][0] = 4000.00`).
- Measure execution time and calculate speedup for each implementation.
- Compare the practical performance differences between the four computing models.

## 3. Computing Models at a Glance

**Sequential.** A single CPU execution flow computes every element of the result matrix one after another using three nested loops. It is the simplest implementation and the baseline for all speedup calculations.

**OpenMP.** The algorithm is unchanged, but the outer loop is divided among multiple CPU threads running on the same machine. All threads share the matrices A, B and C in common memory, so no explicit data transfer is needed.

**MPI.** The work is split across multiple independent processes running on separate VMs (one Master and three Workers). Each process has its own memory, so rows of A are scattered, B is broadcast to every process, and partial results are gathered back on rank 0. Communication over the virtual network adds overhead.

**CUDA.** The multiplication is offloaded to an NVIDIA GPU. The CPU copies the matrices to GPU memory and launches a kernel over a grid of thread blocks, where each logical thread computes one element of C. The result is then copied back to the CPU.

| Model | Memory model | Execution resource |
|---|---|---|
| Sequential | Single address space | 1 CPU core |
| OpenMP | Shared memory | Multiple CPU threads |
| MPI | Distributed memory | Multiple processes on 4 VMs |
| CUDA | Host and device memory | GPU threads |

## 4. Problem Definition

```
A = 4000 × 4000 matrix, all elements = 1.0
B = 4000 × 4000 matrix, all elements = 1.0
C = A × B

C[0][0] = 1×1 + 1×1 + ... + 1×1   (4000 terms)
C[0][0] = 4000.00
```

Because A and B are filled with 1.0, every element of C is expected to be 4000.00. This is used as the verification value in every implementation.

## 5. Components Required

**Hardware**

- Windows 10/11 host system
- Sufficient CPU cores and RAM for WSL and virtual machines
- Four Ubuntu virtual machines (MPI experiment)
- NVIDIA GPU (CUDA experiment)

**Software**

- Windows PowerShell
- WSL2 with Ubuntu (sequential and OpenMP)
- GCC / `build-essential` (OpenMP support is provided through GCC)
- Open MPI and OpenSSH (distributed experiment)
- CUDA Toolkit and `nvcc` (GPU experiment)
- VMware Workstation or equivalent (MPI nodes)

## 7. Part A: Sequential Matrix Multiplication

**Environment:** WSL2 Ubuntu, accessed from Windows PowerShell.

**Prerequisites**

- WSL2 installed with an Ubuntu distribution
- Internet access for package installation
- `sudo` permission in Ubuntu
- GCC (`build-essential`) installed

**Approach:** three nested loops compute `C[i][j] += A[i][k] * B[k][j]` on a single CPU core. The program is compiled with `gcc -O2` and timed using `clock()`.

**Compile and run**

```bash
gcc -O2 matrix_sequential.c -o matrix_sequential
./matrix_sequential
```

**Result**

```
Initializing 4000 x 4000 matrices...

Sequential Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Execution Time = <<seconds>> seconds
Verification C[0][0] = <<value>>
```

| Metric | Value |
|---|---|
| Recorded execution time | `<<seconds>>` s |
| Verification C[0][0] | `<<4000.00 expected>>` |

This time is the **baseline for all speedup calculations**.

---

## 8. Part B: OpenMP Matrix Multiplication

**Environment:** WSL2 Ubuntu (same as Part A).

**Prerequisites**

- Part A environment is working and GCC is installed
- WSL exposes multiple logical CPUs
- Sequential baseline is already recorded

**Approach:** the multiplication logic is unchanged, but `#pragma omp parallel for` splits the outer loop across threads. The thread count is set with `OMP_NUM_THREADS`, the program is compiled with `-fopenmp`, and timing uses `omp_get_wtime()`. CPU usage can be observed with `htop` during the run.

**Compile and run**

```bash
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
./matrix_openmp
```

**Result**

```
OpenMP Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of Threads Used = <<threads>>
Execution Time = <<seconds>> seconds
Verification C[0][0] = <<value>>
```

| Metric | Value |
|---|---|
| Logical CPUs (`nproc`) | `<<number>>` |
| Threads used | `<<number>>` |
| Recorded execution time | `<<seconds>>` s |
| Speedup over sequential | `<<sequential time>> / <<OpenMP time>> = <<speedup>>×` |

**Why it is faster:** threads work on different outer-loop iterations while sharing matrices A, B and C.

---

## 9. Part C: MPI Distributed Matrix Multiplication

**Environment:** four Ubuntu VMs on the same virtual network, one Master and three Workers.

**Prerequisites**

- VMware Workstation or equivalent
- OpenSSH Server and Open MPI installed on all nodes
- Passwordless SSH from Master to Workers
- Executable copied to every Worker

**Cluster configuration**

| Node | Hostname | IP Address | MPI Role |
|---|---|---|---|
| Master | master | `<<master IP>>` | Rank 0 |
| Worker1 | worker1 | `<<worker1 IP>>` | Rank 1 |
| Worker2 | worker2 | `<<worker2 IP>>` | Rank 2 |
| Worker3 | worker3 | `<<worker3 IP>>` | Rank 3 |

**Matrix distribution:** each of the 4 ranks computes 1000 of the 4000 rows (`<<confirm rows per rank>>`).

**Hostfile (`hosts`):** lists the four nodes (`master`, `worker1`, `worker2`, `worker3`) with `slots=1` each.

**MPI data flow**

```
Matrix A (4000 rows)
        │
        +---- MPI_Scatter ----+
        │                     │
Rank 0 → 1000 rows      Rank 1 → 1000 rows
Rank 2 → 1000 rows      Rank 3 → 1000 rows

Matrix B ---- MPI_Bcast ----> all ranks

Each rank computes local_C
        │
        +---- MPI_Gather ----+
                              │
                        Complete C on Rank 0
```

**Compile and run** (on the Master VM)

```bash
mpicc -O2 matrix_mpi.c -o matrix_mpi
mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'
```

**Result**

```
Rank 0 on <<hostname>> computing <<rows>> rows
Rank 1 on <<hostname>> computing <<rows>> rows
Rank 2 on <<hostname>> computing <<rows>> rows
Rank 3 on <<hostname>> computing <<rows>> rows

MPI Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of MPI Processes = <<processes>>
Execution Time = <<seconds>> seconds
Verification C[0][0] = <<value>>
```

| Metric | Value |
|---|---|
| MPI processes / VMs | `<<number>>` |
| Recorded execution time | `<<seconds>>` s |
| Speedup over sequential | `<<sequential time>> / <<MPI time>> = <<speedup>>×` |

**Why communication matters:** MPI must distribute input data and gather partial results across separate address spaces and network-connected VMs.

---

## 10. Part D: CUDA Matrix Multiplication

**Environment:** a CUDA-capable system with an NVIDIA GPU.

**Prerequisites**

- NVIDIA GPU with driver installed and recognized (`nvidia-smi`)
- CUDA Toolkit and `nvcc` installed
- Supported C/C++ host compiler
- Sufficient GPU memory for the matrix data

| Item | Value |
|---|---|
| GPU model | `<<GPU name>>` |
| Driver version | `<<version>>` |
| CUDA version (`nvcc`) | `<<version>>` |

**Approach:** the CPU allocates and initializes the matrices, copies A and B to GPU memory, launches the kernel, copies C back, and verifies the output. CUDA events record both kernel-only time and total CUDA phase time.

**Execution configuration**

| Parameter | Configuration |
|---|---|
| Matrix size | 4000 × 4000 |
| Block size | 16 × 16 = 256 threads/block |
| Grid size | 250 × 250 blocks |
| Total blocks | 62,500 |
| Logical CUDA threads | 16,000,000 |

Because 4000 / 16 = 250, the grid is 250 × 250 blocks. Each block has 256 threads, so 62,500 × 256 = 16,000,000 logical threads, corresponding to the 4000 × 4000 output elements.

**Compile and run**

```bash
nvcc -O2 matrix_cuda.cu -o matrix_cuda
./matrix_cuda
```

**Result**

```
CUDA Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Grid Size = <<x>> x <<y>> blocks
Block Size = <<x>> x <<y>> threads
Kernel Execution Time = <<seconds>> seconds
Total CUDA Phase Time = <<seconds>> seconds
Verification C[0][0] = <<value>>
```

| Metric | Value |
|---|---|
| Kernel-only time | `<<seconds>>` s |
| Total CUDA phase time | `<<seconds>>` s |
| Speedup over sequential (total phase) | `<<sequential time>> / <<CUDA total time>> = <<speedup>>×` |

**Important:** total CUDA phase time includes host-to-device transfer, kernel execution and device-to-host transfer. The CUDA program uses single-precision `float`, while the CPU programs use `double`.

---

## 11. Results and Performance Comparison

All four implementations should produce the same verification value, `C[0][0] = 4000.00`.

### Execution time summary

| Implementation | Model | Resources | Time | Verification |
|---|---|---|---|---|
| Sequential | Single CPU execution | 1 CPU core | `<<seconds>>` s | `<<value>>` |
| OpenMP | Shared memory | `<<threads>>` CPU threads | `<<seconds>>` s | `<<value>>` |
| MPI | Distributed memory | `<<processes>>` processes / `<<VMs>>` VMs | `<<seconds>>` s | `<<value>>` |
| CUDA | GPU parallelism | `<<GPU model>>` | `<<seconds>>` s | `<<value>>` |

### Speedup

```
Speedup = Sequential Execution Time / Parallel Execution Time
```

| Implementation | Execution Time | Speedup |
|---|---|---|
| Sequential | `<<seconds>>` s | 1.00× |
| OpenMP | `<<seconds>>` s | `<<speedup>>`× |
| MPI | `<<seconds>>` s | `<<speedup>>`× |
| CUDA | `<<seconds>>` s | `<<speedup>>`× |

### Optional charts

- `<<Execution time bar chart: image path>>`
- `<<Speedup bar chart: image path>>`

### Observations

- **Baseline:** `<<observation on sequential time>>`
- **OpenMP:** `<<observation on thread-level speedup>>`
- **MPI:** `<<observation on communication and network overhead>>`
- **CUDA:** `<<observation on GPU performance>>`
- **Correctness:** `<<confirm all implementations gave the same verification value>>`

---

## 12. Recommended Result Screenshots

Store screenshots in a `screenshots/` folder and link them here.

| Part | Screenshots to capture | File |
|---|---|---|
| Sequential | PowerShell WSL verification, `gcc --version`, source code, compilation, final output | `<<screenshots/sequential_*.png>>` |
| OpenMP | `nproc`, `OMP_NUM_THREADS`, source code, `htop`, final output | `<<screenshots/openmp_*.png>>` |
| MPI | Four-VM network connectivity, SSH test, `mpirun` output showing ranks and verification | `<<screenshots/mpi_*.png>>` |
| CUDA | `nvidia-smi`, `nvcc --version`, compilation, final CUDA output | `<<screenshots/cuda_*.png>>` |

## 13. Conclusion

`<<Write the conclusion after collecting results.>>`
## 6. Project Structure

```
parallel_lab/
├── sequential/
│   ├── matrix_sequential.c
│   └── matrix_sequential
├── openmp/
│   ├── matrix_openmp.c
│   └── matrix_openmp
├── mpi/
│   ├── hosts
│   ├── matrix_mpi.c
│   └── matrix_mpi
└── cuda/
    ├── matrix_cuda.cu
    └── matrix_cuda
```

---


