#pragma once

#include "Point.h"
#include <stdlib.h>

typedef struct cluster {
	vector_t location;
	point_t** pointArray = NULL;
	int numOfPoints;
	int sizeOfPointArray;
} cluster_t;

void addPoint(cluster_t *c, point_t *p);