#include <stdio.h>
#include <ctime>
#include <stdlib.h>
#include <sys/time.h>
 
// Thread block size
#define BLOCK_SIZE 16
#define TILE_SIZE 32 

#define ROW 1024
#define COL 1024

// GPU Functions
void MM_Basic(float *a, float *b, float *c, int row, int col, int k);
__global__ void MM_Basic_kernel( float *devA, float *devB, float *devC, int row, int col, int k);

void MM_Improved(float *a, float *b, float *c, int row, int col, int k);
__global__ void MM_Improved_kernel( float *devA, float *devB, float *devC, int row, int col, int k);

/*
 * Main function
 */
int main(int argC, char** argV) {
        
        //
        // Setup
        //////////////////

	// Time Variables
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	
	// Matrices
	float *a, *b;
        float *c_cpu, *c_gpu_basic, *c_gpu_improved;
		
	//Setting matrix parameters.
	int row = ROW;
	int col = COL;
	int   k = COL;
	int sum = 0;
	
	// Process input arguments (if specified)
	switch (argC) {
		case 2: {
	                row = atoi(argV[1]);
        	        col = row;
                	  k = col;
			break;
		}
		case 3: {
			row = atoi(argV[1]);
			col = atoi(argV[2]);
			  k = col;
			break;
		}
		default: {
			//Nothing
		}
	}
	
	//Setting host memory space.
	a               = (float *) malloc(row*k*sizeof(float));
	b               = (float *) malloc(k*col*sizeof(float));
	c_cpu           = (float *) malloc(row*col*sizeof(float));
	c_gpu_basic     = (float *) malloc(row*col*sizeof(float));
	c_gpu_improved  = (float *) malloc(row*col*sizeof(float));
	
	//Initializing [A] and [B] with random values from 1 to 10, and C to 0
	printf ("Initializing Matricies, could take some time...\n");
	for(int i = 0 ; i < row ; i++ ){
		for(int j = 0 ; j < k ; j++ ){
			a[i*k+j] = rand()%10;
		}
	}
	for(int i = 0 ; i < k ; i++ ){
		for(int j = 0 ; j < col ; j++ ){
			b[i*col+j] = rand()%10;
		}
	}
	for(int i = 0 ; i < k ; i++ ){
		for(int j = 0 ; j < col ; j++ ){
			c_cpu           [i*col+j] = 0;
			c_gpu_basic     [i*col+j] = 0;
			c_gpu_improved  [i*col+j] = 0;
		}
	}

        //
        // CPU Calculation
        //////////////////
        
	printf("Running sequential job.\n");
	cudaEventRecord(start,0);
	for(int i = 0 ; i < row ; i++ ){
		for(int j = 0 ; j < col ; j++ ){
			sum = 0;
			for(int w = 0 ; w < k ; w++ ){
				sum += a[i*k+w] * b[w*col+j];
			}
			c_cpu[i*col+j] = sum;
		}
	}
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("\tSequential Job Time: %.2f ms\n", time);

	//
        // Basic GPU Calculation
        ////////////////////////
        
	printf("Running Basic parallel job.\n");
	
	cudaEventRecord(start,0);
	MM_Basic(a, b, c_gpu_basic, row, col, k);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	printf("\tBasic Parallel Job Time: %.2f ms\n", time);

	// Compares matrices to make sure answer is correct, initializes c for next kernel.
	bool error = false;
	for(int i = 0 ; i < k ; i++ ){
		for(int j = 0 ; j < col ; j++ ){
			if (c_cpu[i*col+j] != c_gpu_basic[i*col+j]) {
				printf("\tError: Starting at [%d][%d]\n", i, j);
				error = true;
			}
			if (error) break;
		}
		if (error) break;
	}
	if (!error) printf("\tNo errors found.\n");
	
        //
        // Improved GPU Calculation
        ////////////////////////
        
	printf("Running Improved parallel job.\n");
	
	cudaEventRecord(start,0);
	MM_Improved(a, b, c_gpu_improved, row, col, k);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	printf("\tImproved Parallel Job Time: %.2f ms\n", time);

	// Compares matrices to make sure answer is correct, initializes c for next kernel.
	error = false;
	for(int i = 0 ; i < k ; i++ ){
		for(int j = 0 ; j < col ; j++ ){
			if (c_cpu[i*col+j] != c_gpu_improved[i*col+j]) {
				printf("\tError: Starting at [%d][%d]\n", i, j);
				error = true;
			}
			if (error) break;
		}
		if (error) break;
	}
	if (!error) printf("\tNo errors found.\n");

        free (a);
        free (b);
        free (c_cpu);
        free (c_gpu_basic);
        free (c_gpu_improved);
}

void MM_Basic(float *a, float *b, float *c, int row, int col, int k) {
	
	cudaEvent_t kernelstart, kernelstop;
	float time;
	cudaEventCreate (&kernelstart);
	cudaEventCreate (&kernelstop);
	
	int sizeA = row*k*sizeof(float);
	int sizeB = k*col*sizeof(float);
	int sizeC = row*col*sizeof(float);
	float *devA, *devB, *devC;
	
	cudaMalloc((void**)&devA, sizeA);
	cudaMalloc((void**)&devB, sizeB);
	cudaMalloc((void**)&devC, sizeC);
	
	cudaMemcpy(devA, a, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, sizeB, cudaMemcpyHostToDevice);
	
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid((COL+dimBlock.x-1)/dimBlock.x, (ROW+dimBlock.y-1)/dimBlock.y, 1);
	
	cudaEventRecord(kernelstart,0);
	MM_Basic_kernel<<<dimGrid, dimBlock>>>(devA, devB, devC, row, col, k);
	cudaEventRecord(kernelstop,0);
	cudaEventSynchronize(kernelstop);

	cudaEventElapsedTime(&time, kernelstart, kernelstop);
	printf("\tKernel Job Time: %.2f ms\n", time);
	
	cudaMemcpy(c, devC, sizeC, cudaMemcpyDeviceToHost);
	
	//Freeing device matrices.
	cudaFree(devA); cudaFree(devB); cudaFree(devC);
}

__global__ void MM_Basic_kernel( float *devA, float *devB, float *devC, int row, int col, int k) {
	int txID = blockIdx.x * blockDim.x + threadIdx.x;
	int tyID = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((txID < col) && (tyID < row)) {
		float Pvalue = 0;
		for(int w = 0 ; w < k ; w++) {
			Pvalue += devA[tyID*k+w] * devB[w*k+txID];
		}
		devC[tyID*k+txID] = Pvalue;
	}
}

void MM_Improved(float *a, float *b, float *c, int row, int col, int k){

        // Write Code here
        cudaEvent_t kernelstart, kernelstop;
        float time;
        cudaEventCreate (&kernelstart);
        cudaEventCreate (&kernelstop);

        int sizeA = row*k*sizeof(float);
        int sizeB = k*col*sizeof(float);
        int sizeC = row*col*sizeof(float);
        float *devA, *devB, *devC;

        cudaMalloc((void**)&devA, sizeA);
        cudaMalloc((void**)&devB, sizeB);
        cudaMalloc((void**)&devC, sizeC);

        cudaMemcpy(devA, a, sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, b, sizeB, cudaMemcpyHostToDevice);

        dim3 dimBlock(32, 32, 1);
        dim3 dimGrid((COL+dimBlock.x-1)/dimBlock.x, (ROW+dimBlock.y-1)/dimBlock.y, 1);

        cudaEventRecord(kernelstart,0);
        MM_Basic_kernel<<<dimGrid, dimBlock>>>(devA, devB, devC, row, col, k);
        cudaEventRecord(kernelstop,0);
        cudaEventSynchronize(kernelstop);

        cudaEventElapsedTime(&time, kernelstart, kernelstop);
        printf("\tKernel Job Time: %.2f ms\n", time);

        cudaMemcpy(c, devC, sizeC, cudaMemcpyDeviceToHost);

        //Freeing device matrices.
        cudaFree(devA); cudaFree(devB); cudaFree(devC);

        
}
__global__ void MM_Improved_kernel( float *devA, float *devB, float *devC, int row, int col, int k){
        // Write Code here
	
	__shared__ int shareBlockA[TILE_SIZE][TILE_SIZE];
	__shared__ int shareBlockB[TILE_SIZE][TILE_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Row = by * TILE_SIZE + ty;
	int Col = bx * TILE_SIZE + tx;
	
	float Cvalue = 0;
        for (int m = 0; m < row/TILE_SIZE; ++m) {
          shareBlockA[ty][tx] = devA[Row * row + (m*TILE_SIZE + tx)];
	      shareBlockB[ty][tx] = devB[Col + (m * TILE_SIZE + ty) * row];
	      __syncthreads();
	      for (int k = 0; k < TILE_SIZE; ++k)
 		    Cvalue += shareBlockA[ty][k] * shareBlockB[k][tx];
	      __syncthreads();
      }
      devC[Row*row+Col] = Cvalue;
} 
