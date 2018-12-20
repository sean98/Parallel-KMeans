#pragma once

#include <stdio.h>
#include <math.h>

//struct vector to easilty define points in space
typedef struct vector {
	double x, y, z;
} vector_t;

//struct point witch contains location and speed/velocity
typedef struct point {
	vector_t location, speed;
} point_t;

//initiate vector with the arguments (x,y,z)
void vectorInit(vector_t* vec, double x, double y, double z);

//calculate distace between two vectors
double distance(vector_t v1, vector_t v2);

//add vector v2 to vector v1
void addVector(vector_t* v1, vector_t v2);

//add vector v2 to vector v1
//each dim is added atomicly
void atomicAddVector(vector_t* v1, vector_t v2);


//return 1 if vectors are eqaual in all dimensions
int vectorIsEqual(vector_t v1, vector_t v2);