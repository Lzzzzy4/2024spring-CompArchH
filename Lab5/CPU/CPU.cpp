#include <bits/stdc++.h>
#include <immintrin.h>

int N = (1 << 4);
int BLOCK_SIZE = 8;

void gemm_baseline(float *A, float *B, float *C); // you can use inline function
void gemm_verify(float *A, float *B); // you can use inline function
void gemm_avx(float *A, float *B, float *C); // you can use inline function
void gemm_avx_block(float *A, const float *B, float *C); // you can use inline function

int main(void) {
    // malloc A, B, C
    for(int i = 3; i<=9; i++){
    N = 4096;
    BLOCK_SIZE = 1 << i;
    std::cout << "BLOCK_SIZE = " << BLOCK_SIZE << std::endl;
    auto A = new float[N * N];
    auto B = new float[N * N];
    auto C = new float[N * N];
    auto D = new float[N * N];
    auto E = new float[N * N];
    // random initialize A, B
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }
    // print A, B
    // for (int i = 0; i < N * N; i++) {
    //     printf("%d %f %f\n",i,  A[i], B[i]);
    // }
    // measure time
    auto start = std::chrono::high_resolution_clock::now();
    gemm_baseline(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "baseline time: " << elapsed.count() * 1000<< "ms\n";

    start = std::chrono::high_resolution_clock::now();
    gemm_avx(A, B, D);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "avx time: " << elapsed.count() * 1000<< "ms\n";

    start = std::chrono::high_resolution_clock::now();
    gemm_avx_block(A, B, E);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "avx block time: " << elapsed.count() * 1000<< "ms\n";

    gemm_verify(C, D);
    gemm_verify(C, E);
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    }
    return 0;

}
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

void gemm_verify(float *A, float *B) {
    for (int i = 0; i < N * N; i++) {
        // printf("%d %f %f\n",i,  A[i], B[i]);
        if (abs(A[i] - B[i]) > 1e-3) {
            std::cout << "error\n";
            return ;
        }
    }
    std::cout << "correct\n";
    return ;
}

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
