#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <cstdint>
#include <device_launch_parameters.h>

using namespace std;

#define BLOCK_SIZE 64
int N = (1 << 10);

__global__ void gemm_baseline(float* A, float* B, float* C, int N){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int idx = threadId + blockDim.x * blockDim.y * blockId;
    C[idx] = 0;
    for (size_t i = 0; i < N; i++) {
        C[idx] += A[(idx / N) * N + i] * B[i * N + (idx % N)];
    }
}

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

void gemm_baseline_cpu(float *A, float *B, float *C) {
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

void gemm_verify(float *A, float *B) {
    for (int i = 0; i < N * N; i++) {
        if (abs(A[i] - B[i]) > 1e-2) {
            std::cout << "error\n";
            return ;
        }
    }
    std::cout << "correct\n";
    return ;
}

int main(){
    // for(int i = 3; i<=9; i++){
    // printf("N = %d\n", (1 << i));
    // N = (1 << i);
    N = 2048;
    int DIM_SIZE = BLOCK_SIZE;
    printf("DIM_SIZE = %d\n", DIM_SIZE);
    printf("grid size = %d\n", (N + DIM_SIZE - 1) / DIM_SIZE);
    int size = N * N * sizeof(float);
    auto* c_A = new float[N * N];
    auto* c_B = new float[N * N];
    auto* c_C = new float[N * N];
    auto* c_D = new float[N * N];
    
    uniform_real_distribution<double> u(0, 10);
    default_random_engine e(time(nullptr));

    for (int i = 0; i < N * N; i++) {
        c_A[i] = (float)u(e);
        c_B[i] = (float)u(e);
        c_C[i] = 0;
        c_D[i] = 0;
    }

    gemm_baseline_cpu(c_A, c_B, c_C); //c_C 正确

    float* g_A, * g_B, * g_C;
    cudaMalloc((void**)&g_A, size);
    cudaMalloc((void**)&g_B, size);
    cudaMalloc((void**)&g_C, size);

    cudaMemcpy(g_A, c_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_B, c_B, size, cudaMemcpyHostToDevice);

    dim3 block_dim(DIM_SIZE, DIM_SIZE);
    dim3 grig_dim((N + DIM_SIZE - 1) / DIM_SIZE, (N + DIM_SIZE - 1) / DIM_SIZE);

    // dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 grig_dim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    gemm_block <<<grig_dim, block_dim >>> (g_A, g_B, g_C, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaMemcpy(c_D, g_C, size, cudaMemcpyDeviceToHost);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cout << "elapsed time: " << time << "micros\n";
    // gemm_verify(c_C, c_D);

    // cudaEventRecord(start, 0);
    // gemm_block <<<grig_dim, block_dim >>> (g_A, g_B, g_C, N);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaMemcpy(c_D, g_C, size, cudaMemcpyDeviceToHost);
    // cudaEventElapsedTime(&time, start, stop);
    // cout << "elapsed time: " << time << "micros\n";
    // gemm_verify(c_C, c_D);

    cudaFree(g_A);
    cudaFree(g_B);
    cudaFree(g_C);
    free(c_A);
    free(c_B);
    free(c_C);
    free(c_D);
    // }

    return 0;
}