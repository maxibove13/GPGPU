#include "util.h"

void blur_cpu(float * img_in, int width, int height, float * img_out, float msk[], int m_size){

    float val_pixel=0;

    // Inicializo variables para medir tiempos
    CLK_POSIX_INIT;
    
    CLK_POSIX_START;
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
    CLK_POSIX_STOP;
    CLK_POSIX_ELAPSED;

    float t_elap = t_elap_get;

    printf("blur_cpu timing: %06.3f ms\n",t_elap);
}