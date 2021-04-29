#include "util.h"
#include "CImg.h"

using namespace cimg_library;

void blur_gpu(float * image, int width, int height);
void blur_cpu(float * image_in, int width, int height, float * image_out, float * mask, int m_size);
void ajustar_brillo_cpu(float * img_in, int width, int height, float * img_out, float coef);
void ajustar_brillo_gpu(float * img_in, int width, int height, float * img_out, float coef, int coalesced=1);
    
int main(int argc, char** argv){


	const char * path;

	if (argc < 3) printf("Debe ingresar el nombre del archivo y 1 o 0 dependiendo si se quiere un acceso coalesced o no.\n");
	else
		path = argv[argc-2];

	int coalesced = atoi(argv[2]);


    //inicializamos la mascara
    float mascara[25]={1, 4, 6, 4, 1,
						4,16,24,16, 4,
						6,24,36,24, 6,
						4,16,24,16, 4,
						1, 4, 6, 4, 1};




	CImg<float> image(path);
	CImg<float> image_out(image.width(), image.height(),1,1,0);

	float *img_matrix = image.data();
    float *img_out_matrix = image_out.data();

	float elapsed = 0;

	// ajustar_brillo_cpu(img_matrix, image.width(), image.height(), img_out_matrix, 100);

	ajustar_brillo_gpu(img_matrix, image.width(), image.height(), img_out_matrix, 100, coalesced);

   	image_out.save("output_brillo.ppm");

	// blur_cpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, 5);

   	// image_out.save("output_blur.ppm");
   	
    return 0;
}

