#include <stdio.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <Windows.h>
#include <iostream>

using namespace std;

__global__
void matVecMultKernel(float* A, float* B, float* C, int n){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i<n){
		C[i] = 0;
		for (int j = 0; j<n; j++)
			C[i] += A[j*n + i] * B[j];
	}
}

void matVecMult(float* A, float* B, float* C, int n) {
	int size = n * n * sizeof(float);
	int sizevect = n * sizeof(float);
	float *d_A, *d_B, *d_C;
	///Redimensionar y copiar de Host a Device
	cudaMalloc((void **)&d_A, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_B, sizevect);
	cudaMemcpy(d_B, B, sizevect, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_C, sizevect);

	//Llamada Kernel
	matVecMultKernel<<< ceil((n*n) / 256.0), 256 >>> (d_A, d_B, d_C, n);
	//ceil se asegura de que tener suficientes hilos para cubrir los elementos

	//copiar de Device a Host
	cudaMemcpy(C, d_C, sizevect, cudaMemcpyDeviceToHost);

	//liberar memoria
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void showVecMult(float* matriz, float fila, float columna){
	
    for (int x = 0; x < fila; x++){
		for (int y = 0; y < columna; y++){
			int puesto = x*columna + y;
			printf("%3.0f ", matriz[puesto]);
		}
		printf("\n");
	}
}

int main() {

    int fila , columna ;
	cout<<"ingrese dimensiones"<< endl;
	cin>>fila;
	columna=fila;
	float* A = (float*)malloc(fila*columna*sizeof(float));
	float* B = (float*)malloc(fila*sizeof(float));
	float* C = (float*)malloc(fila*sizeof(float));
	for (int i = 0; i < fila*columna; i++)
        A[i] = i;
    for (int i = 0; i < columna; i++)
        B[i] = i;
	
    cout<<" vector "<<endl;
	showVecMult(B, 1, columna);
	cout<<" * matriz "<<endl;
	showVecMult(A, fila, columna);
	
	cout <<"Resultado"<<endl;
	matrizXvector(A, B, C, fila);
	showVecMult(C, 1, columna);

	system("PAUSE");
	exit(0);
}
