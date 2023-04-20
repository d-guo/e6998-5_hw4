// Includes
#include <stdio.h>
#include "timer.h"
#include <cudnn.h>

// Defines
#define epsilon (float)1e-4
#define verbose 0

#define C 3
#define H 1024
#define W 1024
#define K 64
#define FH 3
#define FW 3
#define P 1


// Create input matrix stored in host memory.
double* createIMatrix() {
  double* input_matrix;
  input_matrix = (double*) malloc(C * H * W * sizeof(double));
  for(int c = 0; c < C; c++) {
    for(int h = 0; h < H; h++) {
        for(int w = 0; w < W; w++) {
            input_matrix[c * H * W + h * W + w] = c * (h + w);
        }
    }
  }
  return input_matrix;
}

// Create all K filter matrices on host
double* createFilterMatrices() {
  double* filters;
  filters = (double*) malloc(K * C * FH * FW * sizeof(double));
  for(int k = 0; k < K; k++) {
    for(int c = 0; c < C; c++) {
        for(int h = 0; h < FH; h++) {
            for(int w = 0; w < FW; w++) {
                filters[k * C * FH * FW + c * FH * FW + h * FW + w] = (c + k) * (h + w);
            }
        }
    }
  }
  return filters;
}

// Compute and check the checksum of the result matrix
// expected to be 122756344698240
void checkResult(double* result) {

  double checksum = 122756344698240;
  double checksum_M = 0;
  
  for(int k = 0; k < K; k++) {
    for(int h = 0; h < H; h++) {
        for(int w = 0; w < W; w++) {
            checksum_M += result[k * H * W + h * W + w];
        }
    }
  }
  printf("checksum: %lf\n", checksum_M);
  if(fabs(checksum - checksum_M)> epsilon * checksum) {
    printf("\n\nTEST FAILED\n");
    printf("computed checksum: %lf\n", checksum_M);
    printf("actual checksum: %lf\n", checksum);
  }
}

//
// main
//
int main(int argc, char** argv) {
  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  int image_bytes = C * H * W * sizeof(double);
  double* input_matrix = createIMatrix();
  int filters_bytes = K * C * FH * FW * sizeof(double);
  double* filters = createFilterMatrices();
  int result_bytes = K * H * W * sizeof(double);
  double* result;
  result = (double*) malloc(result_bytes);

  // descriptors
  cudnnTensorDescriptor_t input_descriptor;
  cudnnFilterDescriptor_t kernel_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
  cudnnConvolutionDescriptor_t convolution_descriptor;

  cudnnCreateTensorDescriptor(&input_descriptor);
  cudnnCreateFilterDescriptor(&kernel_descriptor);
  cudnnCreateTensorDescriptor(&output_descriptor);
  cudnnCreateConvolutionDescriptor(&convolution_descriptor);

  cudnnSetTensor4dDescriptor(input_descriptor,
    /*format=*/CUDNN_TENSOR_NCHW,
    /*dataType=*/CUDNN_DATA_DOUBLE,
    /*batch_size=*/1,
    /*channels=*/C,
    /*image_height=*/H,
    /*image_width=*/W);

  cudnnSetFilter4dDescriptor(kernel_descriptor,
    /*dataType=*/CUDNN_DATA_DOUBLE,
    /*format=*/CUDNN_TENSOR_NCHW,
    /*out_channels=*/K,
    /*in_channels=*/C,
    /*kernel_height=*/FH,
    /*kernel_width=*/FW);

  cudnnSetConvolution2dDescriptor(convolution_descriptor,
    /*pad_height=*/P,
    /*pad_width=*/P,
    /*vertical_stride=*/1,
    /*horizontal_stride=*/1,
    /*dilation_height=*/1,
    /*dilation_width=*/1,
    /*mode=*/CUDNN_CONVOLUTION,
    /*computeType=*/CUDNN_DATA_DOUBLE);
  
  cudnnSetTensor4dDescriptor(output_descriptor,
    /*format=*/CUDNN_TENSOR_NCHW,
    /*dataType=*/CUDNN_DATA_DOUBLE,
    /*batch_size=*/1,
    /*channels=*/K,
    /*image_height=*/H,
    /*image_width=*/W);

  double* d_input;
  cudaMallocManaged(&d_input, image_bytes);
  cudaMemcpy(d_input, input_matrix, image_bytes, cudaMemcpyHostToDevice);
  
  double* d_kernel;
  cudaMallocManaged(&d_kernel, filters_bytes);
  cudaMemcpy(d_kernel, filters, filters_bytes, cudaMemcpyHostToDevice);

  double* d_output;
  cudaMallocManaged(&d_output, result_bytes);

  const int n_requestedAlgo = 10;
  int n_returnedAlgo;
  cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf[n_requestedAlgo];
  cudnnFindConvolutionForwardAlgorithm(cudnn, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, n_requestedAlgo, &n_returnedAlgo, fwd_algo_perf);

  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  convolution_algorithm = fwd_algo_perf[0].algo;

  size_t workspace_bytes = 0;
  cudnnGetConvolutionForwardWorkspaceSize(cudnn,
    input_descriptor,
    kernel_descriptor,
    convolution_descriptor,
    output_descriptor,
    convolution_algorithm,
    &workspace_bytes);

  double* d_workspace;
  cudaMalloc(&d_workspace, workspace_bytes);

  double alpha = 1, beta = 0;

    // warmup
  cudnnConvolutionForward(cudnn,
    &alpha,
    input_descriptor,
    d_input,
    kernel_descriptor,
    d_kernel,
    convolution_descriptor,
    convolution_algorithm,
    d_workspace,
    workspace_bytes,
    &beta,
    output_descriptor,
    d_output);

  cudaDeviceSynchronize();
  
    // Set up timer
  initialize_timer();
  start_timer();

  cudnnConvolutionForward(cudnn,
    &alpha,
    input_descriptor,
    d_input,
    kernel_descriptor,
    d_kernel,
    convolution_descriptor,
    convolution_algorithm,
    d_workspace,
    workspace_bytes,
    &beta,
    output_descriptor,
    d_output);

  cudaDeviceSynchronize();

  stop_timer();

  cudaMemcpy(result, d_output, result_bytes, cudaMemcpyDeviceToHost);

  double time = elapsed_time();
  printf( "Time: %lf (sec)\n", time);

  // Verify that the result is correct.
  checkResult(result);

  cudaFree(d_kernel);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);

  cudnnDestroy(cudnn);

  free(input_matrix);
  free(filters);
  free(result);
}
