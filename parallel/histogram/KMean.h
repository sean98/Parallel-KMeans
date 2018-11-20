#pragma once

#include "Cluster.h"

cluster_t* kMeans(int n, int k, int limit, point_t p[]);

double quality(cluster_t clusters1[], int k);