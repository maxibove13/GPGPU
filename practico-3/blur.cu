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

__global__ void blur_kernel(float* d_input, float* d_output, float* d_msk, int width, int height){

}

__global__ void ajustar_brillo_coalesced_kernel(float* d_img, int width, int height, float coef) {

    int threadId, blockId;

    blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    threadId = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (blockIdx.x == 0 && blockIdx.y == 0) {
        // printf("%d | (%d, %d) \n", threadId, threadIdx.x, threadIdx.y);
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("Grid & Block dimensions from GPU:\n");
            printf("grid:  (%d, %d, %d) \n", gridDim.x, gridDim.y, gridDim.z);
            printf("block: (%d, %d, %d) \n", blockDim.x, blockDim.y, blockDim.z);
            printf("\n");
        }
    }
        // printf("(%d, %d) | (%d, %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

    if (threadId <= width * height)
        d_img[threadId] = min(255.0f,max(0.0f,d_img[threadId]+coef));

}

__global__ void ajustar_brillo_no_coalesced_kernel(float* d_img, int width, int height, float coef) {

    int threadId, blockId;

    blockId = (gridDim.y * blockIdx.x) + blockIdx.y;
    threadId = (blockId * (blockDim.y * blockDim.x)) + (threadIdx.x * blockDim.y) + threadIdx.y;

    if (blockIdx.x == 0 && blockIdx.y == 0) {
        // printf("%d | (%d, %d) \n", threadId, threadIdx.x, threadIdx.y);
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("Grid & Block dimensions from GPU:\n");
            printf("grid:  (%d, %d, %d) \n", gridDim.x, gridDim.y, gridDim.z);
            printf("block: (%d, %d, %d) \n", blockDim.x, blockDim.y, blockDim.z);
            printf("\n");
        }
    }
        // printf("(%d, %d) | (%d, %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

    if (threadId <= width * height)
        d_img[threadId] = min(255.0f,max(0.0f,d_img[threadId]+coef));

}

void ajustar_brillo_gpu(float * img_in, int width, int height, float * img_out, float coef, int coalesced) {

    float *d_img;
    int nx = 8;
    int ny = 8;
    int nbx = width / nx + 1;
    int nby = height / ny + 1;
    unsigned int size_img = width * height * sizeof(float);

    printf("Image dimensions:\n");
    printf("width: %d px\n", width);
    printf("height: %d px\n", height);
    printf("\n");

    CLK_CUEVTS_INIT;
    CLK_POSIX_INIT;
    
    
    // Reservar memoria en la GPU
 
    CLK_POSIX_START;
    CLK_CUEVTS_START;
    CUDA_CHK(cudaMalloc((void**)&d_img, size_img));
    CLK_CUEVTS_STOP;
    CLK_POSIX_STOP;
    CLK_CUEVTS_ELAPSED;
    CLK_POSIX_ELAPSED;
    float t_elap_cuda_malloc = t_elap_cuda;
    float t_elap_get_malloc = t_elap_get;

    // copiar imagen a la GPU
    CLK_POSIX_START;
    CLK_CUEVTS_START;
    CUDA_CHK(cudaMemcpy(d_img, img_in, size_img, cudaMemcpyHostToDevice));
    CLK_CUEVTS_STOP;
    CLK_POSIX_STOP;
    CLK_CUEVTS_ELAPSED;
    CLK_POSIX_ELAPSED;
    float t_elap_cuda_cpyHtoD = t_elap_cuda;
    float t_elap_get_cpyHtoD = t_elap_get;

    // configurar grilla y lanzar kernel
    dim3 grid(nbx,nby);
    dim3 block(nx,ny);


    CLK_POSIX_START;
    CLK_CUEVTS_START;
    if (coalesced == 1) {
        ajustar_brillo_coalesced_kernel <<< grid, block >>> (d_img, width, height, coef);
    } else {
        ajustar_brillo_no_coalesced_kernel <<< grid, block >>> (d_img, width, height, coef);
    }


    // Obtengo los posibles errores en la llamada al kernel
	CUDA_CHK(cudaGetLastError());

	// Obligo al Kernel a llegar al final de su ejecucion y hacer obtener los posibles errores
	CUDA_CHK(cudaDeviceSynchronize());

    CLK_CUEVTS_STOP;
    CLK_POSIX_STOP;
    CLK_CUEVTS_ELAPSED;
    CLK_POSIX_ELAPSED;
    float t_elap_cuda_kernel = t_elap_cuda;
    float t_elap_get_kernel = t_elap_get;

    // transferir resultado a la memoria principal
    CLK_POSIX_START;
    CLK_CUEVTS_START;
    CUDA_CHK(cudaMemcpy(img_out, d_img, size_img, cudaMemcpyDeviceToHost));
    CLK_CUEVTS_STOP;
    CLK_POSIX_STOP;
    CLK_CUEVTS_ELAPSED;
    CLK_POSIX_ELAPSED;
    float t_elap_cuda_cpyDtoH = t_elap_cuda;
    float t_elap_get_cpyDtoH = t_elap_get;

    // liberar la memoria
    CLK_POSIX_START;
    CLK_CUEVTS_START;
    cudaFree(d_img);
    CLK_CUEVTS_STOP;
    CLK_POSIX_STOP;
    CLK_CUEVTS_ELAPSED;
    CLK_POSIX_ELAPSED;
    float t_elap_cuda_free = t_elap_cuda;
    float t_elap_get_free = t_elap_get;

    printf("time:     | cudaEvents        | gettimeofday\n");
    printf("malloc:   | %f ms       | %f ms\n", t_elap_cuda_malloc, t_elap_get_malloc);
    printf("cpyHtoD:  | %f ms       | %f ms\n", t_elap_cuda_cpyHtoD, t_elap_get_cpyHtoD);
    printf("kernel:   | %f ms       | %f ms\n", t_elap_cuda_kernel, t_elap_get_kernel);
    printf("cpyDtoH:  | %f ms       | %f ms\n", t_elap_cuda_cpyDtoH, t_elap_get_cpyDtoH);
    printf("free:     | %f ms       | %f ms\n", t_elap_cuda_free, t_elap_get_free);
}


void blur_gpu(float * img_in, int width, int height, float * img_out, float msk[], int m_size){
    
    // Reservar memoria en la GPU

    // copiar imagen y máscara a la GPU
   
    // configurar grilla y lanzar kernel
   
    // transferir resultado a la memoria principal

	// liberar la memoria
}

void ajustar_brillo_cpu(float * img_in, int width, int height, float * img_out, float coef){

    for(int imgx=0; imgx < width ; imgx++){
        for(int imgy=0; imgy < height; imgy++){
            img_out[imgy*width+imgx] = min(255.0f,max(0.0f,img_in[imgy*width+imgx]+coef));
        }
    }
}

void blur_cpu(float * img_in, int width, int height, float * img_out, float msk[], int m_size){

    float val_pixel=0;
    
    //para cada pixel aplicamos el filtro
    for(int imgx=0; imgx < width ; imgx++){
        for(int imgy=0; imgy < height; imgy++){

            val_pixel = 0;

            // aca aplicamos la mascara
            for (int i = 0; i < m_size ; i++){
                for (int j = 0; j < m_size ; j++){
                    
                    int ix =imgx + i - m_size/2;
                    int iy =imgy + j - m_size/2;
                    
                    if(ix >= 0 && ix < width && iy>= 0 && iy < height )
                        val_pixel = val_pixel +  img_in[iy * width +ix] * msk[i*m_size+j];
                }
            }
            
            // guardo valor resultado
            img_out[imgy*width+imgx]= val_pixel;
        }
    }
}