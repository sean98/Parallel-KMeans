#include "Point.h"

void vectorInit(vector_t* vec, double x, double y, double z)
{
	vec->x = x;
	vec->y = y;
	vec->z = z;
}

double distance(vector_t v1, vector_t v2)
{
	return (v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z);
}

void addVector(vector_t* v1, vector_t v2)
{
	v1->x += v2.x;
	v1->y += v2.y;
	v1->z += v2.z;
}

void atomicAddVector(vector_t* v1, vector_t v2)
{
	#pragma omp atomic
	v1->x += v2.x;
	#pragma omp atomic
	v1->y += v2.y;
	#pragma omp atomic
	v1->z += v2.z;
}

int vectorIsEqual(vector_t v1, vector_t v2)
{
	return v1.x == v2.x&&v1.y == v2.y&&v1.z == v2.z;
}