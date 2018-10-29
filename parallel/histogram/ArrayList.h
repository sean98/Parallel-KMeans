#pragma once

#include <stdlib.h>
#include "Point.h"

/*
typedef struct arraylist
{
	int physicalSize, logicalSize;
	void** data;
} arraylist_t;

void initArrayList(arraylist_t** arr, int size);

void resetArrayList(arraylist_t* arr);

void addElement(arraylist_t* arr, void* e);

void freeArrayList(arraylist_t* arr);
*/

struct node {
	void* data;
	struct node* next = NULL;
} typedef node_t;

void addElement(node_t** arr, void* e);

void freeArrayList(node_t** arr);
