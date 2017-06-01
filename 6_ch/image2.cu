#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdlib.h>
#include "lodepng.h"

__global__
void PictureKernell(unsigned char* d_Pin, unsigned char* d_Pout, int n, int m){
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int new_pos;
  if((y < n) && (x < m)) {
    new_pos = (y*m+x)*4;
    unsigned char r = d_Pin[new_pos];
    unsigned char g = d_Pin[new_pos+1];
    unsigned char b = d_Pin[new_pos+2];
    d_Pout[new_pos] = 0.21f*r + 0.71f*g + 0.07f*b;
    d_Pout[new_pos+1] = d_Pout[new_pos];
    d_Pout[new_pos+2] = d_Pout[new_pos];
    d_Pout[new_pos+3] = d_Pin[new_pos+3];
  }
}

__global__
void PictureKernel1D(unsigned char* d_Pin, unsigned char* d_Pout, int n, int m){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  x = x*4;
  if(x < n*m*4) {
    unsigned char r = d_Pin[x];
    unsigned char g = d_Pin[x+1];
    unsigned char b = d_Pin[x+2];
    d_Pout[x] = 0.21f*r + 0.71f*g + 0.07f*b;
    d_Pout[x+1] = d_Pout[x];
    d_Pout[x+2] = d_Pout[x];
    d_Pout[x+3] = d_Pin[x+3];
  }
}

void Picture(unsigned char* Pin, unsigned char* Pout, int n, int m){
  unsigned char* d_Pout, *d_Pin;
  long int size = n*m*4;
  cudaMalloc((void **) &d_Pin,size);
  cudaMemcpy(d_Pin, Pin, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **) &d_Pout,size);

  dim3 gridDim((m-1)/8+1,(n-1)/16+1,1);
  dim3 blockDim(8,16,1);
  PictureKernell<<<gridDim,blockDim>>>(d_Pin,d_Pout,n,m);
  //PictureKernel1D<<<(size-1)/256+1,256>>>(d_Pin,d_Pout,n,m);
  cudaMemcpy(Pout, d_Pout, size, cudaMemcpyDeviceToHost);
  cudaFree(d_Pin); cudaFree(d_Pout);
}

int main(){
  unsigned char *image, *out_image;
  int i;
  char name_in[100], name_out[100];
  unsigned width, height;
  scanf("%s %s", name_in, name_out);
  i = lodepng_decode32_file(&image, &width, &height, name_in);
  if(i < 0) printf("NO\n");
  out_image = (unsigned char*) malloc(width*height*4);
  /*for(i = 0; i < (width * height)*4; i++){
    if(i%4==0) image[i] = 0;
    if(i%4==1) image[i] = 255;
    if(i%4==3) image[i] = 120;
  }*/
  Picture(image,out_image,height,width);
  lodepng_encode32_file(name_out,out_image,width,height);

  free(image);
  free(out_image);
  return 0;
}
