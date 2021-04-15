void sumMatrixRow(int M, int N, float * M_h, float * sum_h) {

    // Declare the size of the matrix and the size of the sum array
    int size_matrix = M * N * sizeof(float), size_sum = M * sizeof(float);

    // Declare the Matrix and the sum array of the device
    float * M_d, * sum_d;

    // Allocate memory on device
    cudaMalloc((void**)&M_d, size_matrix); // allocate Matrix space on device global memory
    cudaMalloc((void**)&sum_d, size_sum); // allocate sum array space on device global memory

    // Initialize matrices on device
    cudaMemcpy(M_d, M_h, size_matrix, cudaMemCpyHostToDevice); // Copy matrix from host to device.
    cudaMemSet(sum_d, 0, size_sum); // Initialize sum_d as an array of 0 and size: size_sum in device (dont need to copy just initialize it on device because it is a simpe array of zeros)

    // Set the execution
    dim3 gridSize(1, 1)l // Grid dimension (Just 1 block in both dims for now)
    dim3 blockSize(M, 1, 1) // Block dimension (Just M threads -as many as rows- in x dim for now)

    // Invoke kernel
    sumMatrixRowsKernel <<< gridSize, blockSize >>> (N, M_d, sum_d);


    // Bring result to host
    cudaMemcpy(sum_h, sum_d, size_sum, cudaMemCpyDeviceToHost); // Copy the sum_d array from device global memory to host DRAM.

    // Free memory on device
    cudaFree(M_d); // Free matrix space in memory on device
    cudaFree(sum_d); // Free sum array space in memory on device

}

__global__ void sumMatrixRowsKernel(int N, float * M_d, float * sum_d) {
    
    float partial_sum = 0;
    int aux = threadIdx.x * N; // There are as many threads as rows in M_d matrix. aux represents the pointer pointing to the first column of this row. Each thread will represent each row.

    // Sum all elements of this row (N is the number of columns)
    for (int k = 0; k < N; ++k) {
        partial_sum += M_d[aux+k];
    }

    // Assign the partial_sum to the sum_d array of sums
    sum_d[threadIdx.x] = partial_sum;

}