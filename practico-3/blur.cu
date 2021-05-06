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

__global__ void blur_kernel(float* d_input, int width, int height, float* d_output, float * d_msk,   int m_size){

    int threadIdPixel, blockId;
    int neighbourPixel;
    float val_pixel=0;

        if (blockIdx.x == 0 && blockIdx.y == 0) {
            // printf("%d | (%d, %d) \n", threadId, threadIdx.x, threadIdx.y);
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                printf("El mask de 0 desde la entrada del kernel GPU es :%f\n",d_msk[1]);
            }
        }

    blockId         = (gridDim.x * blockIdx.y) + blockIdx.x;
    threadIdPixel   = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    printf("El mask de 0 despues del threadId y blockId del kernel GPU es :%f\n",d_msk[1]);
    
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        // printf("%d | (%d, %d) \n", threadId, threadIdx.x, threadIdx.y);
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("El mask de 0 desde habiendo definido blockId y threadId GPU es :%f\n",d_msk[0]);
        }
    }
    //zona de print
    // printf("El mask de 0 desde GPU es :%f\n",d_msk[0]);
    // printf("BlockID: %d\n",blockId);
    // printf("ThreadID:%d\n",threadIdPixel);
    // printf("Mask size:%d\n",m_size);
    // if (blockIdx.x == 0 && blockIdx.y == 0) {
    //     // printf("%d | (%d, %d) \n", threadId, threadIdx.x, threadIdx.y);
    //     if (threadIdx.x == 0 && threadIdx.y == 0) {
    //         printf("val_pixel_AntesFor:%f\n",val_pixel);
    //         printf("\n");
    //     }
    // }
    for (int i = 0; i < m_size ; i++){
        for (int j = 0; j < m_size ; j++){
            neighbourPixel =threadIdPixel + (j- m_size/2) +(i-m_size/2)*width ;
            // printf("neighbourPixel:%d\n",neighbourPixel);                      
            if(neighbourPixel >= 0 && neighbourPixel < width * height ){
                val_pixel = val_pixel +  d_input[neighbourPixel] * d_msk[i*m_size+j];
            }
                if (blockIdx.x == 0 && blockIdx.y == 0) {
                    // printf("%d | (%d, %d) \n", threadId, threadIdx.x, threadIdx.y);
                    if (threadIdx.x == 0 && threadIdx.y == 0) {
                        printf("threadIdPixel_GPU:%d\n",threadIdPixel);
                        printf("neighbourPixel_GPU:%d\n",neighbourPixel);
                        printf("El mask de 0 desde kernel GPU es :%f\n",d_msk[0]);
                        printf("index_mask_pixel_GPU:%d\n",i*m_size+j);
                        printf("i desde GPU:%d\n",i);
                        printf("j desde_GPU:%d\n",j);
                        printf("m_size desde_GPU:%d\n",m_size);
                        printf("mask_pixel_GPU:%f\n",d_msk[i*m_size+j]);
                        printf("img_pixel_GPU:%f\n",d_input[neighbourPixel]);
                        printf("val_pixel_GPU:%f\n",val_pixel);
                        printf("\n");
                    }
                }
        }
    }
    d_output[threadIdPixel]= val_pixel;
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
    int nx = 1;
    int ny = 1;
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



void blur_gpu(float * image_in, int width, int height, float * image_out,  float mask[], int m_size){
    
    // Reservar memoria en la GPU
    float *d_img_in; float *d_img_out; float *d_mask;
    int nx = 16 ; //Número de threads por blque en x
    int ny = 16 ; //Número de threads por blque en y  
    int nbx = width / nx +1 ; //Número de blques x
    int nby = height / ny +1; //Número de blques Y
    unsigned int size_img = width * height * sizeof(float);
    unsigned int size_msk = m_size * m_size * sizeof(int);
    CUDA_CHK(cudaMalloc( (void**)&d_img_in   , size_img));//Reservo memoria en el device para la imagen original
    CUDA_CHK(cudaMalloc( (void**)&d_img_out  , size_img));//Reservo memoria en el device para la imagen de salida
    CUDA_CHK(cudaMalloc( (void**)&d_mask     , size_msk));//Reservo memoria para la mascada
    

    // copiar imagen y máscara a la GPU
    CUDA_CHK(cudaMemcpy(d_img_in  , image_in  , size_img, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_img_out , image_out , size_img, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_mask    , &mask[0]  , size_msk, cudaMemcpyHostToDevice));

    // configurar grilla y lanzar kernel
    dim3 grid(nbx,nby)  ;
    dim3 block(nx,ny) ;
    printf("El mask de 0 antes de invoacar al kernelito es :%f\n",mask[0]);
    blur_kernel <<< grid, block >>> (d_img_in, width, height, d_img_out, d_mask,  m_size); 
        
    // Obtengo los posibles errores en la llamada al kernel
	CUDA_CHK(cudaGetLastError());

	// Obligo al Kernel a llegar al final de su ejecucion y hacer obtener los posibles errores
	CUDA_CHK(cudaDeviceSynchronize());

    // transferir resultado a la memoria principal
    CUDA_CHK(cudaMemcpy(image_out  , d_img_out , size_img, cudaMemcpyDeviceToHost));
	
    // liberar la memoria
    cudaFree(d_img_in); cudaFree(d_img_out) ; cudaFree(d_mask);

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
                        if (imgx == 0 && imgy== 0) {
                            // printf("%d | (%d, %d) \n", threadId, threadIdx.x, threadIdx.y);
                            printf("threadIdPixel_CPU:%d\n",imgy * width +imgx) ;
                            printf("neighbourPixel_CPU:%d\n",iy * width +ix)    ;
                            printf("mask_pixel_CPU:%f\n",msk[i*m_size+j])       ;
                            printf("img_pixel_CPU:%f\n",img_in[iy * width +ix]) ;
                            printf("val_pixel_CPU:%f\n",val_pixel)              ;

                            printf("\n");
                            
                        }
                
                    }
            }
            
            // guardo valor resultado
            img_out[imgy*width+imgx]= val_pixel;
        }
    }
}
