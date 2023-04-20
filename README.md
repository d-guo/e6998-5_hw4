# Instructions for Running Code

The code for each part is separated into each directory with the appropriate name.

For all parts, running `make all` will build all executables

## PART A
`./vecadd00 ValuesPerThread` runs the provided code (naive, no colasced memory reads)
`./vecadd01 ValuesPerThread` runs new code (yes colasced memory reads)
`./matmult00 NumBlocks` runs the provided code (single value per thread)
`./matmult01 NumBlocks` runs new code (four values per thread)

## PART B
N is the array size
mode is 1, 2, or 3 denoting
1. 1 block, 1 thread
2. 1 block, 256 threads
3. multiple blocks, 256 threads

`./vecadd_cpu N mode` runs on cpu
`./vecadd N mode` runs on gpu
`./vecadd_unified N mode` runs on gpu with unified memory

## PART C
`./conv` runs the naive version
`./conv_tiled` runs the tiled version
`./conv_cudnn` runs the cudnn version
