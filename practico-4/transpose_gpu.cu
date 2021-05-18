#include "util.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace std;

__global__ void transpose_kernel(float* d_img_in, float* d_img_out, int width, int height) {
    
    int threadId, blockId, blockId_trans, threadId_trans;

    blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    threadId = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    blockId_trans = (gridDim.y * blockIdx.x) + blockIdx.y;
    threadId_trans = (blockId_trans * (blockDim.x * blockDim.y)) + (threadIdx.x * blockDim.y) + threadIdx.y;
    
    // blockId_trans = blockIdx.x * gridDim.x + blockIdx.y;
    // threadId_trans =  * (threadIdx.x * blockDim.x) + threadIdx.y;

    if (threadId <= width * height)
        d_img_out[threadId_trans] = d_img_in[threadId];
}

void transpose_gpu(float * img_in, int width, int height, float * img_out, int threadPerBlockx, int threadPerBlocky) {

    float *d_img_in, *d_img_out;
    int nbx;
    int nby;
    unsigned int size_img = width * height * sizeof(float);

    width % threadPerBlockx == 0 ? nbx = width / threadPerBlockx : nbx = width / threadPerBlockx + 1;
    height % threadPerBlocky == 0 ? nby = height / threadPerBlocky : nby = height / threadPerBlocky + 1;

    // Inicializo variables para medir tiempos
    CLK_CUEVTS_INIT;
    
    // Reservar memoria en la GPU
    CLK_CUEVTS_START;
    CUDA_CHK(cudaMalloc((void**)&d_img_in, size_img));
    CUDA_CHK(cudaMalloc((void**)&d_img_out, size_img));
    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;
    float t_elap_cuda_malloc = t_elap_cuda;

    // copiar imagen a la GPU
    CLK_CUEVTS_START;
    CUDA_CHK(cudaMemcpy(d_img_in, img_in, size_img, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_img_out, img_out, size_img, cudaMemcpyHostToDevice));
    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;
    float t_elap_cuda_cpyHtoD = t_elap_cuda;

    // configurar grilla y lanzar kernel
    dim3 grid(nbx,nby);
    dim3 block(threadPerBlockx,threadPerBlocky);

    CLK_CUEVTS_START;
    transpose_kernel <<< grid, block >>> (d_img_in, d_img_out, width, height);
    CLK_CUEVTS_STOP;

    // Obtengo los posibles errores en la llamada al kernel
	CUDA_CHK(cudaGetLastError());

	// Obligo al Kernel a llegar al final de su ejecucion y hacer obtener los posibles errores
	CUDA_CHK(cudaDeviceSynchronize());

    CLK_CUEVTS_ELAPSED;
    float t_elap_cuda_kernel = t_elap_cuda;

    // transferir resultado a la memoria principal
    CLK_CUEVTS_START;
    CUDA_CHK(cudaMemcpy(img_out, d_img_out, size_img, cudaMemcpyDeviceToHost));
    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;
    float t_elap_cuda_cpyDtoH = t_elap_cuda;

    // liberar la memoria
    CLK_CUEVTS_START;
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;
    float t_elap_cuda_free = t_elap_cuda;

    printf("Bright adjustment timing:\n");
    printf("type:     | cudaEvents\n");
    printf("malloc:   | %06.3f ms\n", t_elap_cuda_malloc);
    printf("cpyHtoD:  | %06.3f ms\n", t_elap_cuda_cpyHtoD);
    printf("kernel:   | %06.3f ms\n", t_elap_cuda_kernel);
    printf("cpyDtoH:  | %06.3f ms\n", t_elap_cuda_cpyDtoH);
    printf("free:     | %06.3f ms\n", t_elap_cuda_free);
    printf("TOTAL:    | %06.3f ms\n", t_elap_cuda_malloc + t_elap_cuda_cpyHtoD + t_elap_cuda_kernel + t_elap_cuda_cpyDtoH + t_elap_cuda_free + t_elap_cuda_malloc);
    printf("\n");
}