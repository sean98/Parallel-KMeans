#include "Cluster.h"

void freeCluster(cluster_t* c)
{
	freeArrayList(&c->pointsList);
	free(c);
}