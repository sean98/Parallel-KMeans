#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "Point.h"
#include "Cluster.h"
#include "KMean.h"

#define NUM_OF_THREADS 512
#define NUM_OF_BLOCKS 1
#define QUALITY_DEVICE 0
#define CPU 0
#define GPU 1

//makes a dummy call to the gpu to "wake" him
//this is used to start the gpu faster in future cases on some devices
void cudaInit();


//Move each point location from srcPoints array to dstPoints by delta*velocity.
//This method accelerated by Cuda.
void cudaMovePoints(point_t h_dstPoints[], point_t h_srcPoints[], int n, double dt);

//calculate clusters for a set of points in specific time
//This method accelerated by Cuda.
cluster_t* cudaKMeans(int n, int k, int limit, point_t h_points[], double* h_qaulity);