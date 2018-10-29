#include "ArrayList.h"

/*
void initArrayList(arraylist_t** arr, int size)
{
	*arr = (arraylist_t*)malloc(sizeof(arraylist_t));
	(*arr)->logicalSize = 0;
	(*arr)->physicalSize = size;
	(*arr)->data = (void**)calloc(size, sizeof(void*));
}

void resetArrayList(arraylist_t* arr)
{
	arr->logicalSize = 0;
}

void addElement(arraylist_t* arr, void* e)
{
	if (arr->logicalSize < arr->physicalSize)
		arr->data[arr->logicalSize++] = e;
	else
	{
		arr->physicalSize = (2 * arr->physicalSize + 1);
		arr->data = (void**)realloc(arr->data, arr->physicalSize * sizeof(void*));
		addElement(arr, e);
	}
}

void freeArrayList(arraylist_t* arr)
{
	free(arr->data);
	free(arr);
}
*/

void addElement(node_t** arr, void* e)
{
	node_t* newHead = (node_t*)malloc(sizeof(node_t));
	newHead->data = e;
	newHead->next = *arr;
	*arr = newHead;
}

void freeArrayList(node_t** arr)
{
	node_t* tmp1 = *arr;
	node_t* tmp2;
	while (tmp1)
	{
		tmp2 = tmp1->next;
		free(tmp1);
		tmp1 = tmp2;
	}
	//*arr = NULL;
}