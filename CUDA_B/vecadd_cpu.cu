// Includes
#include <stdio.h>
#include "timer.h"
#include "vecaddKernel.h"

// Variables for host and device vectors.
float* h_A; 
float* h_B; 
float* h_C;

void addVectors(const float* A, const float* B, float* C, int N)
{
    int i;
    for( i=0; i<N; ++i ){
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv)
{
    int N; //Vector size

	// Parse arguments.
    if(argc != 2){
     printf("Usage: %s N\n", argv[0]);
     printf("N is the vector size.\n");
     exit(0);
    } else {
      sscanf(argv[1], "%d", &N);
    }      

    if(N % 256 != 0){
     printf("Error: N must be multiple of 256.\n");
     exit(0);
    }

    printf("Total vector size: %d\n", N); 
    // size_t is the total number of bytes for a vector.
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // Initialize host vectors h_A and h_B
    int i;
    for(i=0; i<N; ++i){
     h_A[i] = (float)i;
     h_B[i] = (float)(N-i);   
    }

    // Warm up
    addVectors(h_A, h_B, h_C, N);

    // Initialize timer  
    initialize_timer();
    start_timer();

    // Invoke kernel
    addVectors(h_A, h_B, h_C, N);

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
        float val = h_C[i];
        if (fabs(val - N) > 1e-5)
            break;
    }
    printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");

    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
}
