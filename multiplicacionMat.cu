// multiplication of two matrices 4x4
#include <stdio.h>
#include <cuda_runtime.h>

#define N 4

__global__ void matrix_mul(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int sum = 0;
    if (i < n && j < n) {
        for (int k = 0; k < n; k++)
            sum += a[i * n + k] * b[k * n + j];
        c[i * n + j] = sum;
    }
}

int main() {
    int n = N;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = n * n * sizeof(int);

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            a[i * n + j] = i + j;
            b[i * n + j] = i * j;
        }

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 blockSize(N, N);
    dim3 gridSize((n + N - 1) / N, (n + N - 1) / N);

    clock_t start_d=clock();
    matrix_mul<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    clock_t end_d = clock();
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%d ", c[i * n + j]);
        printf("\n");
    }
    printf("N = %d \t GPU time = %f \t", N, time_d);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}