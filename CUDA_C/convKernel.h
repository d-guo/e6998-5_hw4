/// matmultKernel.h
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-19 DVN
///
/// Kernels defined with this header must 
/// multiply two matrices using CUDA: A x B = C
///

#ifndef __MMKERNEL__
#define __MMKERNEL__

// Defines for size of thread block and data computed by a thread block
#define BLOCK_SIZE 16
#ifndef FOOTPRINT_SIZE
#define FOOTPRINT_SIZE BLOCK_SIZE
#endif

// The type Matrix is really a MATRIX DESCRIPTOR. 
// Matrices are stored in row major order:
//       M[row,col] = *(M.elements + row * M.stride + col)
//
// A sub matrix is not copied but allocated in the full matrix.
//
// This requires the stride of the full matrix to properly get to the
// next row of the sub matrix (a block).
//
// Stride is the width in bytes from one element of the larger matrix 
// to the element in the same column but one row down.


#define C 3
#define H 1024
#define W 1024
#define K 64
#define FH 3
#define FW 3
#define P 1
#define OUT_FOOTPRINT 16

typedef struct {
  int channels;
  int height;
  int width;
  int stride_channel;
  int stride_height;
  double* elements;
} Matrix;

// Forward declaration of the kernel function that performs the work.
__global__ void ConvKernel(const Matrix, const Matrix*, Matrix);

#endif

