#include "Cluster.h"

void addPoint(cluster_t * c, point_t * p)
{
	if (c->pointArray == NULL)
	{
		c->pointArray = (point_t**)calloc(++c->sizeOfPointArray, sizeof(point_t*));
		c->numOfPoints = 0;
	}
	else if (c->numOfPoints + 1 > c->sizeOfPointArray)
	{
		c->sizeOfPointArray = 2 * c->sizeOfPointArray + 1;
		c->pointArray = (point_t**)realloc(c->pointArray, c->sizeOfPointArray * sizeof(point_t*));
	}
	c->pointArray[c->numOfPoints++] = p;
}