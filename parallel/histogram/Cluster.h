#pragma once

#include "ArrayList.h"
#include "Point.h"

typedef struct cluster {
	vector_t location;
	node_t* pointsList = NULL;
} cluster_t;

void freeCluster(cluster_t* c);