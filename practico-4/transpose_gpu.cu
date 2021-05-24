#include "util.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define THREAD_PER_BLOCK 32
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

__global__ void transpose_kernel_gobalMem(float* d_img_in, float* d_img_out, int width, int height) {
    
    int pixel_x, pixel_y,threadId_original,threadId_trans; //Declaro variables
    pixel_x = blockIdx.x * blockDim.x + threadIdx.x; //Indices imgx análogo a el CPU transpose
    pixel_y = blockIdx.y * blockDim.y + threadIdx.y; //Indices imgy análogo a el CPU transpose

    threadId_original = pixel_y*width+pixel_x; //Indice de acceso a la imagen original

    threadId_trans = (pixel_x*height+pixel_y);//Indice de acceso a la transpuesta
    
    if (threadId_original <= width * height)
        d_img_out[threadId_trans] = d_img_in[threadId_original];
}

__global__ void transpose_kernel_sharedMem(float* d_img_in, float* d_img_out, int width, int height) {

    __shared__ float tile[THREAD_PER_BLOCK*THREAD_PER_BLOCK]; //Defino el arrray tile en shared memory  
    
    //PASO 1: Leo variables en la imagen original por filas y copio al tile de forma coalseced por filas
    int original_pixel_x, original_pixel_y,threadId_original,threadId_tile_row;
    
    original_pixel_x = blockIdx.x  * blockDim.x + threadIdx.x;
    original_pixel_y = blockIdx.y  * blockDim.y + threadIdx.y;
    
    threadId_original = original_pixel_y * width + original_pixel_x ;//Indice de acceso a la imagen original
    threadId_tile_row = threadIdx.y * blockDim.x + threadIdx.x      ;//El block dim.x es el ancho del tile
    
    tile[threadId_tile_row]= d_img_in[threadId_original];
    __syncthreads(); // Me aseguro que se hayan copiado todos los datos al tile sino algunos threades impertientens se pueden encontrar con datos nulos
     //    Garantizado los datos en memoria compartida

    //PASO 2: Accedo por columnas al tile y calculo ese índice. 
    int threadId_tile_col;
    threadId_tile_col = threadIdx.x * blockDim.y + threadIdx.y;//El block dim.y es el height del tile

    // PASO 3: Pego en las filas de la imagen de salida de forma coalesced
    int transpose_pixel_x,transpose_pixel_y,threadId_trans;
    transpose_pixel_x = blockIdx.y * blockDim.y + threadIdx.x ;//Se accede por columnas
    transpose_pixel_y = blockIdx.x * blockDim.x + threadIdx.y ;
    threadId_trans    = transpose_pixel_x + transpose_pixel_y * height ;
    
    if (threadId_trans <= width * height)
        d_img_out[threadId_trans] = tile[threadId_tile_col];
}

// __global__ void transpose_kernel_sharedMem_fixedConflict(float* d_input, float* d_output, int width, int height){
//     __shared__ float tile[threadPerBlock]; //Defino el arrray tile en shared memory  
//     //PASO 1: Leo variables en la imagen original y copio al tile de forma coalseced
//     int original_pixel_x, original_pixel_y,threadId_original,threadId_tile_row;
    
//     original_pixel_x = blockIdx.x  * blockDim.x + threadIdx.x;
//     original_pixel_y = blockIdx.y  * blockDim.y + threadIdx.y;
//     int posicion = threadIdx.x + threadIdx.y;
//     int bl_pixel_y = 0;
//     int bl_pixel_x = 0;
//     if ( posicion < Bl_size ){
//       bl_pixel_x = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.y;
//     }else{
//       bl_pixel_x = threadIdx.x + (threadIdx.y-1)*blockDim.x + threadIdx.y;
//     }
//     tile[bl_pixel_x]= *(d_input + in_pixel_x + in_pixel_y*width );
//     __syncthreads();
//     // Segundo paso
//     if( posicion < Bl_size){
//       bl_pixel_y = threadIdx.y + threadIdx.x*(blockDim.x +1);
//     }else{
//       bl_pixel_y = threadIdx.y + (threadIdx.x-1)*(blockDim.x +1) +1;
//     }
//      // Tercer paso
//      int out_pixel_x = threadIdx.x + blockIdx.y*blockDim.y;
//      int out_pixel_y = threadIdx.y + blockIdx.x*blockDim.x;
//     *(d_output + out_pixel_x + out_pixel_y*height ) = tile[bl_pixel_y];
  
//   }



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


    // Ejecuta Kernel con globalMem
    CLK_CUEVTS_START;
    transpose_kernel_gobalMem <<< grid, block >>> (d_img_in, d_img_out, width, height);
    CLK_CUEVTS_STOP;

    // Obtengo los posibles errores en la llamada al kernel
	CUDA_CHK(cudaGetLastError());

	// Obligo al Kernel a llegar al final de su ejecucion y hacer obtener los posibles errores
	CUDA_CHK(cudaDeviceSynchronize());

    CLK_CUEVTS_ELAPSED;
    float t_elap_cuda_kernel_globalMem = t_elap_cuda;

    CLK_CUEVTS_START;
    transpose_kernel_sharedMem <<< grid, block >>> (d_img_in, d_img_out, width, height);
    CLK_CUEVTS_STOP;

    // Obtengo los posibles errores en la llamada al kernel
	CUDA_CHK(cudaGetLastError());

	// Obligo al Kernel a llegar al final de su ejecucion y hacer obtener los posibles errores
	CUDA_CHK(cudaDeviceSynchronize());

    CLK_CUEVTS_ELAPSED;
    float t_elap_cuda_kernel_sharedMem = t_elap_cuda;

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

    printf("Transpose adjustment timing:\n");
    printf("type:               | cudaEvents\n");
    printf("malloc:             | %06.3f ms\n", t_elap_cuda_malloc);
    printf("cpyHtoD:            | %06.3f ms\n", t_elap_cuda_cpyHtoD);
    printf("kernel globalMem:   | %06.3f ms\n", t_elap_cuda_kernel_globalMem);
    printf("kernel sharedMem:   | %06.3f ms\n", t_elap_cuda_kernel_globalMem);
    printf("cpyDtoH:            | %06.3f ms\n", t_elap_cuda_cpyDtoH);
    printf("free:               | %06.3f ms\n", t_elap_cuda_free);
    printf("TOTAL globalMem:    | %06.3f ms\n", t_elap_cuda_malloc + t_elap_cuda_cpyHtoD + t_elap_cuda_kernel_globalMem + t_elap_cuda_cpyDtoH + t_elap_cuda_free + t_elap_cuda_malloc);
    printf("TOTAL sharedMem:    | %06.3f ms\n", t_elap_cuda_malloc + t_elap_cuda_cpyHtoD + t_elap_cuda_kernel_sharedMem + t_elap_cuda_cpyDtoH + t_elap_cuda_free + t_elap_cuda_malloc);
    printf("\n");
}