#!/bin/bash

echo "running all commands for CUDA_B"

echo "running vecadd_cpu 1000192"
./vecadd_cpu 1000192
echo "running vecadd_cpu 5000192"
./vecadd_cpu 5000192
echo "running vecadd_cpu 10000128"
./vecadd_cpu 10000128
echo "running vecadd_cpu 50000128"
./vecadd_cpu 50000128
echo "running vecadd_cpu 100000000"
./vecadd_cpu 100000000


echo "running vecadd 1000192 1"
./vecadd 1000192 1
echo "running vecadd 1000192 2"
./vecadd 1000192 2
echo "running vecadd 1000192 3"
./vecadd 1000192 3

echo "running vecadd 5000192 1"
./vecadd 5000192 1
echo "running vecadd 5000192 2"
./vecadd 5000192 2
echo "running vecadd 5000192 3"
./vecadd 5000192 3

echo "running vecadd 10000128 1"
./vecadd 10000128 1
echo "running vecadd 10000128 2"
./vecadd 10000128 2
echo "running vecadd 10000128 3"
./vecadd 10000128 3

echo "running vecadd 50000128 1"
./vecadd 50000128 1
echo "running vecadd 50000128 2"
./vecadd 50000128 2
echo "running vecadd 50000128 3"
./vecadd 50000128 3

echo "running vecadd 100000000 1"
./vecadd 100000000 1
echo "running vecadd 100000000 2"
./vecadd 100000000 2
echo "running vecadd 100000000 3"
./vecadd 100000000 3


echo "running vecadd_unified 1000192 1"
./vecadd_unified 1000192 1
echo "running vecadd_unified 1000192 2"
./vecadd_unified 1000192 2
echo "running vecadd_unified 1000192 3"
./vecadd_unified 1000192 3

echo "running vecadd_unified 5000192 1"
./vecadd_unified 5000192 1
echo "running vecadd_unified 5000192 2"
./vecadd_unified 5000192 2
echo "running vecadd_unified 5000192 3"
./vecadd_unified 5000192 3

echo "running vecadd_unified 10000128 1"
./vecadd_unified 10000128 1
echo "running vecadd_unified 10000128 2"
./vecadd_unified 10000128 2
echo "running vecadd_unified 10000128 3"
./vecadd_unified 10000128 3

echo "running vecadd_unified 50000128 1"
./vecadd_unified 50000128 1
echo "running vecadd_unified 50000128 2"
./vecadd_unified 50000128 2
echo "running vecadd_unified 50000128 3"
./vecadd_unified 50000128 3

echo "running vecadd_unified 100000000 1"
./vecadd_unified 100000000 1
echo "running vecadd_unified 100000000 2"
./vecadd_unified 100000000 2
echo "running vecadd_unified 100000000 3"
./vecadd_unified 100000000 3
