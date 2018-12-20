#pragma once

#include "Cluster.h"
#include <omp.h>

//calculate clusters for a set of points in specific time
//This method accelerated by OpenMp.
cluster_t* kMeans(int n, int k, int limit, point_t p[]);

//calculate quality of a set of clusters
//This method accelerated by OpenMp.
double quality(cluster_t clusters1[], int k);