#pragma once

#include <stdio.h>
#include <math.h>

typedef struct vector {
	double x, y, z;
} vector_t;

typedef struct point {
	vector_t location, speed;
} point_t;

void vectorInit(vector_t* vec, double x, double y, double z);

double distance(vector_t v1, vector_t v2);

void addVector(vector_t* v1, vector_t v2);

void printVector(vector_t v);

int vectorIsEqual(vector_t v1, vector_t v2);