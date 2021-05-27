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

// __global__ void blur_kerne_2aii(float* d_input, int width, int height, float* d_output, float * d_msk,   int m_size){

//     __shared__ float tile[THREAD_PER_BLOCK*THREAD_PER_BLOCK]; //Defino el arrray tile en shared memory  
    
//     //PASO 1: Leo variables en la imagen original por filas y copio al tile de forma coalseced por filas
//     int original_pixel_x, original_pixel_y,threadId_original,threadId_tile_row, threadIdPixel;
    
//     original_pixel_x = blockIdx.x  * blockDim.x + threadIdx.x;
//     original_pixel_y = blockIdx.y  * blockDim.y + threadIdx.y;
    
//     threadId_original = original_pixel_y * width + original_pixel_x ;//Indice de acceso a la imagen original
//     threadId_tile_row = threadIdx.y * blockDim.x + threadIdx.x      ;//El block dim.x es el ancho del tile
//     tile[threadId_tile_row]= d_input[threadId_original];
//     __syncthreads();
//         //Inicializo el valor ponderado del pixel
//         float val_pixel = 0;
//       // PASO 2: Aplicamos la mascara
//       for (int i = 0; i < m_size ; i++){//Los indices se mueven en el tile que es igual al tamaño de la mascara
//           for (int j = 0; j < m_size ; j++){
//             int mask_pixel_x = original_pixel_x + i - m_size/2;
//             int mask_pixel_y = original_pixel_y + j - m_size/2;
//             int threadId_mask= mask_pixel_x + mask_pixel_y*blockDim.x;
            

//             int tile_pixel_x = threadIdx.x + i - m_size/2;
//             int tile_pixel_y = threadIdx.y + j - m_size/2;
//             int threadId_tile= tile_pixel_x + tile_pixel_y*blockDim.x;

//             if(mask_pixel_x >= 0 && mask_pixel_x < width && mask_pixel_y>= 0 && mask_pixel_y < height ) { // Chequeo no excederme de los limites de la imagen original
//               if(tile_pixel_x >= 0 && tile_pixel_x < blockDim.x && tile_pixel_x>= 0 && tile_pixel_x < blockDim.y ){   // Estoy dentro del bloque compartido??
//                 val_pixel = val_pixel +  tile[threadId_tile] * d_msk[i*m_size+j];
//               }else{ // Voy a buscar a memoria a global los datos faltantes
//                 val_pixel = val_pixel +  d_input[threadId_mask] * d_msk[i*m_size+j];
//               }
//              }
//            }
//         }
//         // Escribo valor en la imagen de salida
//         if (threadIdPixel <= width * height )
//         d_output[threadIdPixel] = val_pixel;
//   }
__global__ void blur_kernel_2ai(float* d_input, int width, int height, float* d_output, float * d_msk,   int m_size){

    extern __shared__ float tile[]; //Defino el arrray tile en shared memory  

    // Defino indices de mapeo en el tile
    int tile_pixel_x = threadIdx.x ;
    int tile_pixel_y = threadIdx.y ;
    int width_tile   = blockDim.x + m_size - 1;


    //PASO 1: Leo variables en la imagen original por filas y copio al tile de forma coalseced por filas en el bloque 1 
    
    int original_pixel_x_1 = blockIdx.x  * blockDim.x + threadIdx.x -m_size/2;
    int original_pixel_y_1 = blockIdx.y  * blockDim.y + threadIdx.y -m_size/2;
    
    int threadIdTile_row  = tile_pixel_x + tile_pixel_y * width_tile    ;
    int threadId_original_1 = original_pixel_y_1 * width + original_pixel_x_1 ;

    if (original_pixel_x_1 > 0 && original_pixel_y_1 > 0 ){
       tile[threadIdTile_row] = d_input[threadId_original_1]   
    } 

    //PASO 2: Leo variables en la imagen original por filas y copio al tile de forma coalseced por filas en el bloque 2 
    
    int original_pixel_x_2 = blockIdx.x  * blockDim.x + threadIdx.x + m_size/2;
    int original_pixel_y_2 = blockIdx.y  * blockDim.y + threadIdx.y - m_size/2; //Es igual al 1
    
    int threadId_original_2 = original_pixel_y * width + original_pixel_x ;

    if (original_pixel_x_2 <= width_tile-blockDim.x && original_pixel_y_2 > 0 ){
       tile[threadIdTile_row] = d_input[threadId_original_2]   
    }
    //PASO 3: Leo variables en la imagen original por filas y copio al tile de forma coalseced por filas en el bloque 2 
    
    int original_pixel_x_3 = blockIdx.x  * blockDim.x + threadIdx.x - m_size/2;
    int original_pixel_y_3 = blockIdx.y  * blockDim.y + threadIdx.y + m_size/2; //Es igual al 1
    
    int threadId_original_3 = original_pixel_y * width + original_pixel_x ;

    if (original_pixel_x_3 > 0 && original_pixel_y_3 <= width_tile - blockDim.y ){
        tile[threadIdTile_row] = d_input[threadId_original_3]   
    }
    //PASO 3: Leo variables en la imagen original por filas y copio al tile de forma coalseced por filas en el bloque 2 
    
    int original_pixel_x_4 = blockIdx.x  * blockDim.x + threadIdx.x + m_size/2;
    int original_pixel_y_4 = blockIdx.y  * blockDim.y + threadIdx.y + m_size/2; //Es igual al 1
        
    int threadId_original_4 = original_pixel_y * width + original_pixel_x ;

    if (original_pixel_x_4 <= width_tile-blockDim.x  && original_pixel_y_3 <= width_tile - blockDim.y ){
        tile[threadIdTile_row] = d_input[threadId_original_4]   
    }    
    __syncthreads();
    //Inicializo el valor ponderado del pixel
    float val_pixel  = 0;
    int out_pixel_x  = threadIdx.x + blockIdx.x * blockDim.x ;
    int out_pixel_y  = threadIdx.y + blockIdx.y * blockDim.y ;
    int out_threadId = out_pixel_x + out_pixel_y * width ;

    for (int i = 0; i < m_size ; i++){
      for (int j = 0; j < m_size ; j++){
        int read_tile_x = threadIdx.x + i ;
        int read_tile_y = threadIdx.y + j ;
        int read_threadId = read_tile_x + width_tile * read_tile_y;

        if(read_threadId >= 0 && read_threadId < width_tile * width_tile ){
              val_pixel = val_pixel +  tile[read_threadId] * d_msk[i*m_size+j];
          }
      }
    }
    // Escribo valor en la imagen de salida
    if (out_threadId <= width * height )
        d_output[out_threadId] = val_pixel;
    }  
}
  void blur_gpu(float * image_in, int width, int height, float * image_out,  float mask[], int m_size, int threadPerBlockx, int threadPerBlocky){
    
    // Reservar memoria en la GPU
    float *d_img_in; float *d_img_out; float *d_mask;
    int nbx;//Número de blques x
    int nby;//Número de blques Y
    unsigned int size_img = width * height * sizeof(float);
    unsigned int size_msk = m_size * m_size * sizeof(int);
   
    
    width % threadPerBlockx == 0 ? nbx = width / threadPerBlockx : nbx = width / threadPerBlockx + 1;
    height % threadPerBlocky == 0 ? nby = height / threadPerBlocky : nby = height / threadPerBlocky + 1;

    unsigned int size_tile = (threadPerBlockx+m_size-1) * (threadPerBlocky+m_size-1) * sizeof(int);

    CUDA_CHK(cudaMalloc( (void**)&d_img_in   , size_img));//Reservo memoria en el device para la imagen original
    CUDA_CHK(cudaMalloc( (void**)&d_img_out  , size_img));//Reservo memoria en el device para la imagen de salida
    CUDA_CHK(cudaMalloc( (void**)&d_mask     , size_msk));//Reservo memoria para la mascada


    // copiar imagen y máscara a la GPU
    CUDA_CHK(cudaMemcpy(d_img_in  , image_in  , size_img, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_img_out , image_out , size_img, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_mask    , &mask[0]  , size_msk, cudaMemcpyHostToDevice));

    // configurar grilla y lanzar kernel
    dim3 grid(nbx,nby)  ;
    dim3 block(threadPerBlockx,threadPerBlocky) ;

    blur_kernel_2ai <<< grid, block,size_tile >>> (d_img_in, width, height, d_img_out, d_mask,  m_size); 
    // blur_kernel_2aii <<< grid, block >>> (d_img_in, width, height, d_img_out, d_mask,  m_size); 

    // Obtengo los posibles errores en la llamada al kernel
	  CUDA_CHK(cudaGetLastError());

	// Obligo al Kernel a llegar al final de su ejecucion y hacer obtener los posibles errores
	  CUDA_CHK(cudaDeviceSynchronize());

    // transferir resultado a la memoria principal
    CUDA_CHK(cudaMemcpy(image_out  , d_img_out , size_img, cudaMemcpyDeviceToHost));
	
    // liberar la memoria
    cudaFree(d_img_in); cudaFree(d_img_out) ; cudaFree(d_mask);

}

