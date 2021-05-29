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

  __global__ void blur_kernel_2ai(float* d_input, int width, int height, float* d_output, float * d_msk, int m_size){
  
      extern __shared__ float tile[]; //Defino el arrray tile en shared memory  
  
      // Defino indices de mapeo en el tile
      int tile_pixel_x = threadIdx.x;
      int tile_pixel_y = threadIdx.y;
      int width_tile   = blockDim.x + m_size - 1;

      int original_pixel_x = blockIdx.x  * blockDim.x + threadIdx.x - m_size/2;
      int original_pixel_y = blockIdx.y  * blockDim.y + threadIdx.y - m_size/2;
      int threadIdTile_row  = tile_pixel_x + tile_pixel_y * width_tile;
      int threadId_original = (original_pixel_y) * width + (original_pixel_x) ;
      
      
      //PASO 1:
      if (threadId_original >= 0 ) {
        tile[threadIdTile_row] = d_input[threadId_original];
      } 
      __syncthreads();
      
      
      
      // if (blockIdx.x == 1 && blockIdx.y == 0) {
      //   if (threadIdx.x == 0 && threadIdx.y == 0) {
      //     printf(" tile[32]: %f\n", tile[32]);
      //   }
      // }
      
      
      //PASO 2
      if (threadIdx.x < m_size - 1 && threadIdTile_row + blockDim.x > blockDim.x && (threadIdTile_row + blockDim.x) % width_tile >= blockDim.x && threadIdTile_row + blockDim.x < width_tile * width_tile && threadId_original + blockDim.x < width * height) {
        tile[threadIdTile_row + blockDim.x] = d_input[threadId_original + blockDim.x];
      }
      __syncthreads();

      // if (blockIdx.x == 1 && blockIdx.y == 0) {
      //   if (threadIdx.x == 0 && threadIdx.y == 0) {
      //     printf(" tile[32]: %f\n", tile[32]);
      //   }
      // }


      //PASO 3:
      if (threadIdx.y < m_size - 1 && threadIdTile_row + blockDim.y * width_tile > width_tile * (blockDim.y) && threadIdTile_row + blockDim.y < width_tile * width_tile && threadId_original + blockDim.y * width < width * height){
        tile[threadIdTile_row + blockDim.y * width_tile] = d_input[threadId_original + blockDim.y * width];
      }
      __syncthreads();

      // if (blockIdx.x == 0 && blockIdx.y == 0) {
      //     if (threadIdTile_row + blockDim.x + blockDim.y * width_tile == 104) {
      //       printf("tile[threadIdTile_row]: %f\n", tile[threadIdTile_row]);
      //       printf("threadIdx.x :%d\n", threadIdx.x);
      //       printf("threadIdx.y :%d\n", threadIdx.y);
      //       printf("original_pixel_x %d\n", original_pixel_x);
      //       printf("original_pixel_y %d\n", original_pixel_y);
      //       printf("threadId_original: %d\n", threadId_original);
      //       printf("threadIdTile_row: %d\n", threadIdTile_row);
      //     }
      //   }

      //PASO 4:
      if (threadIdx.x < m_size - 1 && threadIdx.y < m_size - 1 && threadIdTile_row + blockDim.x + blockDim.y * width_tile > width_tile * (blockDim.y) && (threadIdTile_row + blockDim.x + blockDim.y * width_tile) % width_tile >= blockDim.x && threadIdTile_row + blockDim.x + blockDim.y * width_tile >= blockDim.x + blockDim.y*width_tile && threadIdTile_row + blockDim.x + blockDim.y * width_tile < width_tile * width_tile && threadId_original + blockDim.x + blockDim.y * width < width * height)
        tile[threadIdTile_row + blockDim.x + blockDim.y * width_tile] = d_input[threadId_original + blockDim.x + blockDim.y * width];

      __syncthreads();
            if (blockIdx.x == 0 && blockIdx.y == 0) {
              if (threadIdx.x == 0 && threadIdx.y == 0) {
                printf("0,0 pixel friends GPU:\n");
              }
            }
      
      // Redefino el pixel en cuestión en función de sus vecinos y la máscara:
      float val_pixel  = 0;
      int out_pixel_x  = threadIdx.x + blockIdx.x * blockDim.x;
      int out_pixel_y  = threadIdx.y + blockIdx.y * blockDim.y;
      for (int i = 0; i < m_size ; i++) {
        for (int j = 0; j < m_size ; j++) {
          
          int read_tile_x = threadIdx.x + i - m_size/2;
          int read_tile_y = threadIdx.y + j - m_size/2;
          int read_threadId = read_tile_x + m_size/2 + width_tile * (read_tile_y + m_size/2);
  
          if(read_threadId >= 0 && out_pixel_x + i - m_size/2 >= 0 && out_pixel_y + j - m_size/2 >= 0) {
            
            val_pixel +=  tile[read_threadId] * d_msk[i*m_size + j];
            // if (blockIdx.x == 0 && blockIdx.y == 0) {
            //   if (threadIdx.x == 7 && threadIdx.y == 7) {
            //     printf("%04.1f | %2.0f | %04d | %d | %d | %f \n", tile[read_threadId], d_msk[i*m_size + j], read_threadId, i, j, val_pixel);
            //   }
            // }
          }


        }
      }

      

      int out_threadId = out_pixel_x + out_pixel_y * width ;
      // Escribo valor en la imagen de salida
      if (out_threadId < width * height) {
        d_output[out_threadId] = val_pixel;
      }

      // if (blockIdx.x == 0 && blockIdx.y == 0) {
      //   if (threadIdx.x == 7 && threadIdx.y == 7) {
      //     printf("d_output[8967]_gpu: %f", d_output[8967]);
      //   }
      // }

    }
  
__global__ void blur_kernel_2aii(float* d_input, int width, int height, float* d_output, float * d_msk,   int m_size){

    __shared__ float tile[THREAD_PER_BLOCK*THREAD_PER_BLOCK]; //Defino el arrray tile en shared memory  
    
    //PASO 1: Leo variables en la imagen original por filas y copio al tile de forma coalseced por filas

    int original_pixel_x = blockIdx.x  * blockDim.x + threadIdx.x;
    int original_pixel_y = blockIdx.y  * blockDim.y + threadIdx.y;
    
    int threadId_original = original_pixel_y * width + original_pixel_x ;//Indice de acceso a la imagen original
    int threadId_tile_row = threadIdx.y * blockDim.x + threadIdx.x      ;//El block dim.x es el ancho del tile
    
    tile[threadId_tile_row]= d_input[threadId_original];
    __syncthreads();
        //Inicializo el valor ponderado del pixel
        float val_pixel = 0;
      // PASO 2: Aplicamos la mascara
      for (int i = 0; i < m_size ; i++){//Los indices se mueven en el tile que es igual al tamaño de la mascara
          for (int j = 0; j < m_size ; j++){
            int mask_pixel_x = original_pixel_x + i - m_size/2;
            int mask_pixel_y = original_pixel_y + j - m_size/2;
            int threadId_mask= mask_pixel_x + mask_pixel_y * blockDim.x;
            

            int tile_pixel_x = threadIdx.x + i - m_size/2;
            int tile_pixel_y = threadIdx.y + j - m_size/2;
            int threadId_tile= tile_pixel_x + tile_pixel_y*blockDim.x;

            if(mask_pixel_x >= 0 && mask_pixel_x < width && mask_pixel_y>= 0 && mask_pixel_y < height ) { // Chequeo no excederme de los limites de la imagen original
              if(tile_pixel_x >= 0 && tile_pixel_x < blockDim.x && tile_pixel_y>= 0 && tile_pixel_y < blockDim.y ){   // Estoy dentro del bloque compartido??
                val_pixel = val_pixel +  tile[threadId_tile] * d_msk[i*m_size+j];
              }else{ // Voy a buscar a memoria a global los datos faltantes
                val_pixel = val_pixel +  d_input[threadId_mask] * d_msk[i*m_size + j];
              }
            }
          }
        }
        // Escribo valor en la imagen de salida
        if (threadId_original <= width * height )
        d_output[threadId_original] = val_pixel;
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
    dim3 grid(nbx,nby);
    dim3 block(threadPerBlockx,threadPerBlocky);

    blur_kernel_2ai <<< grid, block, size_tile >>> (d_img_in, width, height, d_img_out, d_mask, m_size); 
    // blur_kernel_2aii <<< grid, block >>> (d_img_in, width, height, d_img_out, d_mask, m_size); 

    // Obtengo los posibles errores en la llamada al kernel
    CUDA_CHK(cudaGetLastError());

	  // Obligo al Kernel a llegar al final de su ejecucion y hacer obtener los posibles errores
    CUDA_CHK(cudaDeviceSynchronize());

    // transferir resultado a la memoria principal
    CUDA_CHK(cudaMemcpy(image_out, d_img_out , size_img, cudaMemcpyDeviceToHost));
	
    // liberar la memoria
    cudaFree(d_img_in); 
    cudaFree(d_img_out); 
    cudaFree(d_mask);

}




