#include "Point.h"

void vectorInit(vector_t* vec, double x, double y, double z)
{
	vec->x = x;
	vec->y = y;
	vec->z = z;
}

double distance(vector_t v1, vector_t v2)
{
	return pow(v1.x - v2.x, 2) + pow(v1.y - v2.y, 2) + pow(v1.z - v2.z, 2);
}

void addVector(vector_t* v1, vector_t v2)
{
	#pragma omp atomic
	v1->x += v2.x;
	#pragma omp atomic
	v1->y += v2.y;
	#pragma omp atomic
	v1->z += v2.z;
}

void printVector(vector_t v)
{
	printf_s("[%-6.2lf, %-6.2lf, %-6.2lf]", v.x, v.y, v.z);
}

int vectorIsEqual(vector_t v1, vector_t v2)
{
	return v1.x == v2.x&&v1.y == v2.y&&v1.z == v2.z;
}

//int pointHashCode(void* point)
//{
//	point_t* tmp = (point_t*)point;
//	return (int)(tmp->x*tmp->y*tmp->z*tmp->vx*tmp->vy*tmp->vz);
//}