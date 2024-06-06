100.411296Lab5实验报告

## PB21000164 来泽远

## CPU

### 型号

11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz   2.30 GHz

### 算法实现

#### Baseline

```c++
void gemm_baseline(float *A, float *B, float *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    return ;
}
```

按照矩阵乘法定义实现即可。

#### AVX

```c++
void gemm_avx(float *A, float *B, float *C) {
    auto *B_rev = new float[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B_rev[i * N + j] = B[j * N + i];
        }
    }
    __m256 sum, a, b;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum = _mm256_setzero_ps();
            for (int k = 0; k < N; k += 8) {
                a = _mm256_loadu_ps(&A[i * N + k]);
                b = _mm256_loadu_ps(&B_rev[j * N + k]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }
            C[i * N + j] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
        }
    }
    free(B_rev);
}
```

由于AVX指令集下存在256位向量寄存器，故可以优化矩阵乘法。具体来说，就是原始矩阵乘法每8次可以合成为一次向量的乘加。

#### AVX_Block

```
void mul_block(const float*A, const float*B, float *C, int size){
    __m256 c;
    for(int j = 0; j < size; j += 8){
        for(int i = 0; i < size; i ++){
            c = _mm256_loadu_ps(&C[i * N + j]);
            for(int k = 0; k < size; k++){
                __m256 a = _mm256_set1_ps(A[i * N + k]);
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                c = _mm256_fmadd_ps(a, b, c);
            }
            _mm256_storeu_ps(&C[i * N + j], c);
        }
    }
}

void gemm_avx_block(float *A, const float *B, float *C){
    int block_size = BLOCK_SIZE;
    __m256 sum, a, b;
    for(int i = 0; i < N; i += block_size){
        for(int j = 0; j < N; j += block_size){
            for(int k = 0; k < N; k += block_size){
                mul_block(A + i * N + k, B + k * N + j, C + i * N + j, block_size);
            }
        }
    }
}
```

AVX下的矩阵分块乘法。具体来说，在之前AVX的基础上，引入分块块大小也即遍历步长。在每一个小矩阵内使用AVX算法计算结果并求和。为了契合AVX算法特性，块大小应该为8的倍数。

### 实验数据

| 矩阵大小         | Baseline/ms | AVX/ms   | AVX_Block/ms |
| ---------------- | ----------- | -------- | ------------ |
| 8$\times$8       | 0.000597    | 0.000528 | 0.000126     |
| 16$\times$16     | 0.00385     | 0.003456 | 0.000719     |
| 32$\times$32     | 0.031956    | 0.010535 | 0.00423      |
| 64$\times$64     | 0.205935    | 0.031527 | 0.050611     |
| 128$\times$128   | 1.5976      | 0.284279 | 0.234541     |
| 256$\times$256   | 16.3401     | 4.15073  | 1.89044      |
| 512$\times$512   | 156.738     | 15.169   | 18.5154      |
| 1024$\times$1024 | 2374.52     | 134.319  | 215.169      |
| 2048$\times$2048 | 31996.1     | 1765.92  | 2475.1       |
| 4096$\times$4096 | 515327      | 15125.8  | 24584.6      |

同时，我们计算了每个元素的平均耗时。

| 矩阵大小         | Baseline/ms | AVX/ms   | AVX_Block/ms |
| ---------------- | ----------- | -------- | ------------ |
| 8$\times$8       | 0.000009    | 0.000008 | 0.000002     |
| 16$\times$16       | 0.000015    | 0.000013 | 0.000003     |
| 32$\times$32       | 0.000031    | 0.000010 | 0.000004     |
| 64$\times$64       | 0.000050    | 0.000008 | 0.000012     |
| 128$\times$128       | 0.000098    | 0.000017 | 0.000014     |
| 256$\times$256       | 0.000249    | 0.000063 | 0.000029     |
| 512$\times$512       | 0.000598    | 0.000058 | 0.000071     |
| 1024$\times$1024       | 0.002265    | 0.000128 | 0.000205     |
| 2048$\times$2048       | 0.007628    | 0.000421 | 0.000590     |
| 4096$\times$4096       | 0.030716    | 0.000902 | 0.001465     |

可以发现基本也保持了Baseline>AVX>AVX_Block的规律，可见各级的优化都有明显作用。为制图，我们将平均耗时取log并且加以常数，制图如下：

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20240605230841953.png" alt="image-20240605230841953" style="zoom:150%;" />

AVX的优化效果为在矩阵小时比较差，而在矩阵规模增加后甚至会优于分块AVX的效果。这可能是因为在矩阵小的时候，向量级的LoadStore以及算术等可能需要更多的检测、启动时间（常数级别），所以优化效果不明显。而矩阵变大之后元素平摊了这部分时间，使优化效果很明显。到最后分块AVX的效果劣于AVX可能是因为分块导致的额外时间消耗（如函数调用，跳转等）会随着规模变大而变大，而朴素AVX则没有这些消耗，当规模足够时分块会略微逊色于AVX。

同时我们也研究了关于分块大小的影响，数据如下。

| BLOCK_SIZE\N | 1024/ms | 2048/ms | 4096/ms |
| ------------ | ------- | ------- | ------- |
| 8            | 186.568 | 1835.4  | 26385.3 |
| 16           | 166.905 | 1492.35 | 16417.8 |
| 32           | 174.79  | 1539.44 | 14887.2 |
| 64           | 173.024 | 1542.36 | 13230.4 |
| 128          | 189.367 | 1545.09 | 22074.9 |
| 256          | 199.649 | 3006.82 | 27221.2 |
| 512          | 385.956 | 3274.37 | 25423.8 |

可以发现只要分块大小不与N相当都有不错的优化效果。不过BLOCK_SIZE太小也不合适，可以认为当BLOCK_SIZE远小于N的时候，分块不会造成什么影响。这也可以解释为什么上文（BLOCK_SIZE = 8）当N很大时，AVX_Block的效果会反而比AVX差，主要是因为分块太小。

### 其他矩阵乘法优化策略

针对矩阵乘法的库函数：BLAS（Basic Linear Algebra Subprograms）、Eigen。主要是通过算法、编译层面的优化。对于算法，可以使用Strassen、Winograd算法等减少复杂度，可以使用循环展开寄存器重用等技巧。对于编译，可以针对不同硬件和指令集开发不同的优化策略。

除此之外，我们也可以开发多线程并行程序，这与GPU计算的思路是一致的。比如使用OpenMP等多线程编程模型，可以方便地将矩阵乘法的不同部分分配到不同的线程上并行计算，从而充分利用多核CPU的计算能力。

## GPU

### 配置

NVIDIA GeForce RTX 3060 Laptop GPU	 CUDA Version: 12.0 

### 算法实现

#### Baseline

```c++
__global__ void gemm_baseline(float* A, float* B, float* C, int N){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int idx = threadId + blockDim.x * blockDim.y * blockId;
    C[idx] = 0;
    for (size_t i = 0; i < N; i++) {
        C[idx] += A[(idx / N) * N + i] * B[i * N + (idx % N)];
    }
}
```

对每一个元素的运算配置一个线程。

需要确定每一个线程实际操作元素的地址。按照grid、block、thread的定义，给出矩阵C中元素的地址，实现N次的求和。

#### Block

```c++
__global__ void gemm_block(float* A, float* B, float* C, int N){
    int begin_a = blockIdx.y * blockDim.y * N;
    int end_a = begin_a + N - 1;
    int step_a = blockDim.x;

    int begin_b = blockIdx.x * blockDim.x;
    int step_b = blockDim.y * N;

    float sum = 0;
    for (int i_a = begin_a, i_b = begin_b; i_a < end_a; i_a += step_a, i_b += step_b)  {
        __shared__ float At[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bt[BLOCK_SIZE][BLOCK_SIZE];

        At[threadIdx.y][threadIdx.x] = A[i_a + threadIdx.y * N + threadIdx.x];
        Bt[threadIdx.y][threadIdx.x] = B[i_b + threadIdx.y * N + threadIdx.x];

        __syncthreads();
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += At[threadIdx.y][i] * Bt[i][threadIdx.x];
        }
        __syncthreads();
    }

    int pos = blockIdx.y * blockDim.y * N + begin_b;
    C[pos + threadIdx.y * N + threadIdx.x] = sum;
}
```

需要注意对矩阵分块后，C矩阵中的元素就不再是仅一个线程进行访问了。并且我们引入shared memory来配合分块矩阵以达到更好的访存性能。其他分块的细节也如CPU一致，给出一个BLOCK_SIZE进行计算即可。

### 实验数据

| 矩阵大小         | Baseline/ms | BLOCK/ms |
| ---------------- | ----------- | -------- |
| 8$\times$8       | 0.006144    | 0.004096 |
| 16$\times$16     | 0.006336    | 0.004096 |
| 32$\times$32     | 0.01024     | 0.004928 |
| 64$\times$64     | 0.01536     | 0.006144 |
| 128$\times$128   | 0.029632    | 0.012224 |
| 256$\times$256   | 0.186112    | 0.057344 |
| 512$\times$512   | 2.54259     | 0.412672 |
| 1024$\times$1024 | 127.052     | 3.1191  |
| 2048$\times$2048 | 225.829     | 28.5676 |
| 4096$\times$4096 | 865.473     | 214.46  |

可以看出分块的优化效果十分显著。这主要来自于访存性能的提高，即访问shared memory比global memory有很大的性能提升。并且在矩阵大时优化效果显著，矩阵小时不显著，这主要是因为分块需要进行数据搬运，元素少时分摊的时间比较多。

### 探究gridsize、blocksize、BLOCK

首先需要说明的是，N、gridsize、blocksize三者之间是相关的，计算方法如下：
$$
gridsize = (N+blocksize-1)/blocksize
$$
即除后向上取整。所以在同样的N下，无法同时控制gridsize、blocksize的变化。

注明：GPU部分的计时与真实时间不一致，可能是线程异步等原因导致的。

#### Baseline

N = 512

| BLOCK_SIZE，GRID_SIZE | T/ms    |
| --------------------- | ------- |
| (8,64)                | 5.31962 |
| (16,32)               | 1.0711 |
| (32,16)               | 1.03526 |
| (64,8)                | 0.002048 |
| (128,4)               | 0.002048 |
| (256,2)               | 0.002048 |
| (512,1)               | 0.002048 |

N = 1024

| BLOCK_SIZE，GRID_SIZE | T/ms    |
| --------------------- | ------- |
| (8,128)               | 12.4896 |
| (16,64)               | 9.43616 |
| (32,32)               | 5.82792 |
| (64,16)               | 0.00512 |
| (128,8)               | 0.00512 |
| (256,4)               | 0.004096 |
| (512,2)               | 0.004096 |

N = 2048

| BLOCK_SIZE，GRID_SIZE | T/ms    |
| --------------------- | ------- |
| (8,256)               | 362.069 |
| (16,128)              | 267.314 |
| (32,64)               | 192.326 |
| (64,32)               | 0.003072 |
| (128,16)              | 0.002048 |
| (256,8)               | 0.003072 |
| (512,4)               | 0.003072 |

可以发现块大小增加，性能提高。这主要是因为有更多的线程处在同一个块中，通信成本减小。并且当BLOCK_SIZE超过64时性能大大增加，因为矩阵结果可以放在共享内存中了，使得进程通信的成本大大减小。并且再增大存在边际效应，这个时候的时间消耗主要是进程交互和数据拷贝了，超多线程使得计算速度极快但是进程交互和数据拷贝有上限，故性能难以增加。

#### 分块

分块的粒度BLOCK与线程块BLOCK_SIZE是相等的，因为需要保证每一个元素分配到一个线程。

N = 512

| BLOCK，GRID_SIZE | T/ms     |
| ---------------- | -------- |
| (8,64)           | 0.411296 |
| (16,32)          | 0.321472 |
| (32,16)          | 0.269835 |
| (64,8)           | 0.215953 |

N = 1024

| BLOCK，GRID_SIZE | T/ms    |
| ---------------- | ------- |
| (8,128)          | 3.16314 |
| (16,64)          | 2.43405 |
| (32,32)          | 2.72384 |
| (64,16)          | 1.98564 |

N = 2048

| BLOCK，GRID_SIZE | T/ms    |
| ---------------- | ------- |
| (8,256)          | 34.1413 |
| (16,128)         | 28.7478 |
| (32,64)          | 23.9333 |
| (64,32)          | 20.1549 |

基本上可以满足分块越大性能越高的规律。只测试到64是因为设备的共享内存的大小限制。性能提高是因为在同样都使用共享内存的前提下，分块越大使得更多的线程在这一快内存中进行计算而无需同步，而块间是需要同步的。块越大则块数越少，同步成本越低。

在GPU计算中，无论是否分块，计算成本相对固定，各个参数主要影响的是访存成本与通信成本。
