#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

__global__
void matMultKernel(float *d_M, float *d_N, float *d_P, int Width){
  int Row = blockIdx.y*blockDim.y + threadIdx.y;
  int Col = blockIdx.x*blockDim.x + threadIdx.x;
  int k = 0;
  if(Row < Width && Col < Width){
      float Pvalue = 0;
      for(k = 0; k < Width; ++k){
          Pvalue += d_M[Row*Width + k] * d_N[k*Width+Col];
      }
      d_P[Row*Width+Col] = Pvalue;
  }
}

void matMult(float* A, float* B, float* C, int n){
  int size = n*n*sizeof(float);
  float *d_A, *d_B, *d_C;

  cudaMalloc((void **) &d_A, size);
  cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
  cudaMalloc((void **) &d_B, size);
  cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);
  cudaMalloc((void **) &d_C, size);

  dim3 dimGrid(ceil(n/256.0),ceil(n/256.0),1);
  dim3 dimBlock(256,256,1);
  matMultKernel<<<dimGrid, dimBlock>>>(d_A,d_B,d_C,n);
  
  cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}


int main(int argc, char* argv[]){

  int n,i,j;
  n = int(strtol(argv[1], NULL, 10));
  float *h_A,*h_B,*h_C;
  //printf("n: ");
  //scanf("%d", &n);
  h_A = (float*) malloc(n*n*sizeof(float));
  h_B = (float*) malloc(n*n*sizeof(float));
  h_C = (float*) malloc(n*n*sizeof(float));
  
  /*---A---*/
  for(i = 0; i < n; i++){
    //scanf("%f", &h_A[i]);
    for(j = 0; j < n; j++)
      h_A[i*n+j] = i+j;
  }
  
  printf("A\n");
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++)
        printf("%f ", h_A[i*n+j]);
    printf("\n");	
  }
  printf("\n");
  
  /*---B---*/
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++)
      h_B[i*n+j] = i+j+10;
  }
  printf("B\n");
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++)
        printf("%f ", h_B[i*n+j]);
    printf("\n");	
  }
  printf("\n");	
  float t;
  t=clock();
  matMult(h_A,h_B,h_C,n);
  t=clock()-t;
  
  /*---C---*/  
  printf("A*B=C\n");
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
        printf("%f ", h_C[i*n+j]);
    }
    printf("\n");	
  }
  printf("\n");
 printf("tiempo: %f \n",t/CLOCKS_PER_SEC);

  return 0;
}
