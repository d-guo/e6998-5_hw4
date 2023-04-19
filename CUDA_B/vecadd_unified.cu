// Includes
#include <stdio.h>
#include "timer.h"
#include "vecaddKernel.h"

// Variables for host and device vectors.
float* d_A; 
float* d_B; 
float* d_C; 

// Utility Functions
void Cleanup(bool);
void checkCUDAError(const char *msg);

// Host code performs setup and calls the kernel.
int main(int argc, char** argv)
{
    int N; //Vector size
    int mode;

	// Parse arguments.
    if(argc != 3){
     printf("Usage: %s N mode\n", argv[0]);
     printf("N is the vector size.\n");
     printf("mode is 1, 2, or 3.\n");
     exit(0);
    } else {
      sscanf(argv[1], "%d", &N);
      sscanf(argv[2], "%d", &mode);
    }      

    if(N % 256 != 0){
     printf("Error: N must be multiple of 256.\n");
     exit(0);
    }

    printf("Total vector size: %d\n", N); 
    // size_t is the total number of bytes for a vector.
    size_t size = N * sizeof(float);

    // Tell CUDA how big to make the grid and thread blocks.
    // Since this is a vector addition problem,
    // grid and thread block are both one-dimensional.
    int dgrid;
    int dblock;
    int ValuesPerThread;
    if(mode == 1){
        dgrid = 1;
        dblock = 1;
        ValuesPerThread = N;
    }
    else if (mode == 2){
        dgrid = 1;
        dblock = 256;
        ValuesPerThread = N / 256;
    }
    else if (mode == 3)
    {
        dgrid = N / 256;
        dblock = 256;
        ValuesPerThread = 1;
    }
    dim3 dimGrid(dgrid);
    dim3 dimBlock(dblock);
    
    // Allocate vectors in UVM memory.
    cudaError_t error;
    error = cudaMallocManaged((void**)&d_A, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMallocManaged((void**)&d_B, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMallocManaged((void**)&d_C, size);
    if (error != cudaSuccess) Cleanup(false);

    // Initialize vectors d_A and d_B
    int i;
    for(i=0; i<N; ++i){
     d_A[i] = (float)i;
     d_B[i] = (float)(N-i);   
    }

    // Warm up
    AddVectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, ValuesPerThread);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    // Initialize timer  
    initialize_timer();
    start_timer();

    // Invoke kernel
    AddVectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, ValuesPerThread);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    // Compute elapsed time 
    stop_timer();
    double time = elapsed_time();

    // Compute floating point operations per second.
    int nFlops = N;
    double nFlopsPerSec = nFlops/time;
    double nGFlopsPerSec = nFlopsPerSec*1e-9;

	// Compute transfer rates.
    int nBytes = 3*4*N; // 2N words in, 1N word out
    double nBytesPerSec = nBytes/time;
    double nGBytesPerSec = nBytesPerSec*1e-9;

	// Report timing data.
    printf( "Time: %lf (sec), GFlopsS: %lf, GBytesS: %lf\n", 
             time, nGFlopsPerSec, nGBytesPerSec);

    // Verify & report result
    for (i = 0; i < N; ++i) {
        float val = d_C[i];
        if (fabs(val - N) > 1e-5)
            break;
    }
    printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");

    // Clean up and exit.
    Cleanup(true);
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
        
    // Free device vectors
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);
        
    error = cudaDeviceReset();
    
    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");
    
    fflush( stdout);
    fflush( stderr);

    exit(0);
}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      exit(-1);
    }                         
}
