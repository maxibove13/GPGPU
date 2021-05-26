#include "util.h"

void transpose_cpu(float * img_in, int width, int height, float * img_out) {

CLK_POSIX_INIT
CLK_POSIX_START
    for(int imgx=0; imgx < width ; imgx++){
        for(int imgy=0; imgy < height; imgy++){
            img_out[imgx*height+imgy] = img_in[imgy*width+imgx];
        }
    }
CLK_POSIX_STOP
CLK_POSIX_ELAPSED

printf("transpose_cpu: %f ms\n", t_elap_get);


}
