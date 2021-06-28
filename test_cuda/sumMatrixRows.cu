#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DEV 0

#define DGS_Check_Call( Statement , MsgString )                   \
                                                                  \
    {                                                             \
        printf ( "Checking CUDA Call... \n") ;                    \
      const cudaError_t error = Statement;                        \
      if (error != cudaSuccess)                                   \
      {                                                           \
        printf( "Error checking CUDA Call... \n")      ;            \
        printf( "Error: %s:%d, ", __FILE__, __LINE__)  ;          \
        printf( "code: %d, reason: %s\n", error, cudaGetErrorString(error));   \
        printf( "Call Checked with error, stopping...\n");       \
        exit(1);                                                  \
      }                                                           \
        printf ( "Call Checked OK, region... \n");                \
    }

// Kernel declaration
__global__ void sumMatrixRowsKernel(int N, double * M_d, double * sum_d);
__host__ void sumMatrixRows(int M, int N, double * M_h, double * sum_h);

// Auxiliary functions declarations
void random_vector(double *a, size_t n);
void array_zeros(double *r, size_t n);

int main(int argc, char * argv[]) {

    if (argc < 2) {
        printf("El programa recibe M la dimensión de la matriz cuadrada aleatoria a sumar por filas");
    }

    // Recieve matrix dimension as input and convert it to int
    int M = atoi(argv[1]);

    // Declare M_h the matrix to sum, and sum_h the array of sums.
    double * M_h = (double *) aligned_alloc(64, M*M*sizeof(double));
    double * sum_h = (double *) aligned_alloc(64, M*sizeof(double));

    // Assign random values to M_h and zeros to sum_h
    random_vector(M_h, M*M);
    array_zeros(sum_h, M);

    printf("sum_h before: ");
    for (int i = 0; i < M; ++i) {
        printf("%f",sum_h[i]);
    }
    printf("\n");

    // Invoke the function to sum the rows of M_h matrix
    sumMatrixRows(M, M, M_h, sum_h);

    printf("sum_h after: ");
    for (int i = 0; i < M; ++i) {
        printf("%f ",sum_h[i]);
    }
    printf("\n");
}

void sumMatrixRows(int M, int N, double * M_h, double * sum_h) {

    // Set up device
    // cudaSetDevice(DEV);
    DGS_Check_Call(cudaSetDevice(DEV), "cudaSetDevice");      // dev - device identifier

    // Declare the size of the matrix and the size of the sum array
    int size_matrix = M * N * sizeof(double), size_sum = M * sizeof(double);

    // Declare the Matrix and the sum array of the device
    double * M_d, * sum_d;

    // Allocate memory on device
    cudaMalloc((void**)&M_d, size_matrix); // allocate Matrix space on device global memory
    cudaMalloc((void**)&sum_d, size_sum); // allocate sum array space on device global memory

    // Initialize matrices on device
    cudaMemcpy(M_d, M_h, size_matrix, cudaMemcpyHostToDevice); // Copy matrix from host to device.
    cudaMemset(sum_d, 0, size_sum); // Initialize sum_d as an array of 0 and size: size_sum in device (dont need to copy just initialize it on device because it is a simpe array of zeros)

    // Set the execution
    dim3 gridSize(1, 1); // Grid dimension (Just 1 block in both dims for now)
    dim3 blockSize(M, 1, 1); // Block dimension (Just M threads -as many as rows- in x dim for now)

    // Invoke kernel
    sumMatrixRowsKernel <<< gridSize, blockSize >>> (N, M_d, sum_d);


    // Bring result to host
    cudaMemcpy(sum_h, sum_d, size_sum, cudaMemcpyDeviceToHost); // Copy the sum_d array from device global memory to host DRAM.

    // Free memory on device
    cudaFree(M_d); // Free matrix space in memory on device
    cudaFree(sum_d); // Free sum array space in memory on device

}

// Kernel definition
__global__ void sumMatrixRowsKernel(int N, double * M_d, double * sum_d) {
    
    double partial_sum = 0;
    int aux = threadIdx.x * N; // There are as many threads as rows in M_d matrix. aux represents the pointer pointing to the first column of this row. Each thread will represent each row.

    // Sum all elements of this row (N is the number of columns)
    for (int k = 0; k < N; ++k) {
        partial_sum += M_d[aux+k];
    }

    // Assign the partial_sum to the sum_d array of sums
    sum_d[threadIdx.x] = partial_sum;

}

void random_vector(double *a, size_t n) {
    for (unsigned int i = 0; i < n; i++) {
        a[i] = (double)rand() / (double)RAND_MAX;
    }
}

void array_zeros(double *r, size_t n) {
    for (unsigned int i = 0; i < n; i++) {
        r[i] = 0;
    }
}