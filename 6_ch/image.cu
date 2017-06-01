#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <math.h>

using namespace std;

#define CHANNELS 3 // we have 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]

/*Blur*/
__global__ 
  void blurKernel(unsigned char * in, unsigned char * out, int w, int h) {
      int Col  = blockIdx.x * blockDim.x + threadIdx.x;
      int Row  = blockIdx.y * blockDim.y + threadIdx.y;

      if (Col < w && Row < h) {
          int pixVal = 0;
          int pixels = 0;

          // Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
          for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
              for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                  int curRow = Row + blurRow;
                  int curCol = Col + blurCol;
                  // Verify we have a valid image pixel
                  if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                      pixVal += in[curRow * w + curCol];
                      pixels++; // Keep track of number of pixels in the accumulated total
                  }
              }
          }

          // Write our new pixel value out
          out[Row * w + Col] = (unsigned char)(pixVal / pixels);
      }
  }

/*Gray*/
__global__ 
void colorConvertKernel(unsigned char * grayImage,unsigned char * rgbImage,int width, int height){
	 int x = threadIdx.x + blockIdx.x * blockDim.x;
	 int y = threadIdx.y + blockIdx.y * blockDim.y;

	 if (x < width && y < height){
	    // get 1D coordinate for the grayscale image
	    int grayOffset = y*width + x;
	    // one can think of the RGB image having
	    // CHANNEL times columns than the gray scale image
	    int rgbOffset = grayOffset*CHANNELS;
	    unsigned char r =  rgbImage[rgbOffset      ]; // red value for pixel
	    unsigned char g = rgbImage[rgbOffset + 2]; // green value for pixel
	    unsigned char b = rgbImage[rgbOffset + 3]; // blue value for pixel
	    // perform the rescaling and store it
	    // We multiply by floating point constants
	    grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
	 }
}

  
  
  int main(int argc, char** argv){
	
	if (argc < 3){
        std::cout << "Usage: " << argv[0] << " <input.png>" << " <output.png>" << std::endl;
        exit(1);
    }

	IplImage* input_image = NULL;
	input_image = cvLoadImage(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    if(!input_image){
        std::cout << "ERROR: No open IMG" << std::endl;
        return -1;
    }

	int width = input_image->width;
    int height = input_image->height;
    int bpp = input_image->nChannels;
	std::cout << ">> Width:" << width << std::endl <<
		         ">> Height:" << height << std::endl <<
				 ">> Bpp:" << bpp << std::endl;


    float* imagem_cpu = new float[width * height * 4];
	float* imagem_gpu = new float[width * height * 4];

	/*Será necesario llenarlo ? */
	cudaMalloc((void **)(&imagem_gpu), (width * height * 4) * sizeof(float));
	cudaMemcpy(imagem_gpu, imagem_cpu, (width * height * 4) * sizeof(float), cudaMemcpyHostToDevice);


	/*funcion Kernel															
	
    colorConvertKernel(imagem_gpu, input_image, width,  height)
    
    blurkernel(imagem_gpu, input_image, width,  height)
    */
    
    cudaMemcpy(imagem_cpu, imagem_gpu, (width * height * 4) * sizeof(float), cudaMemcpyDeviceToHost);

	//cudaMemcpy(imagem_cpu, imagem_gpu, (width * height * 4) * sizeof(float), cudaMemcpyDeviceToHost);

	//img en gris
	//IplImage* out_image = cvCreateImage( cvSize(width, height), input_image->depth, bpp);
	//out_image->imageData = buff;
    
    if( !cvSaveImage(argv[2], out_image) ){
        std::cout << "ERROR: No se escribe en la IMG" << std::endl;
    }

	cvReleaseImage(&input_image);
    cvReleaseImage(&out_image);
	return 0;
1
	
}
