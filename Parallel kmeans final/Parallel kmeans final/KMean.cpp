#include "KMean.h"

//Create new clusters where each cluster location corresponding to point in the same locaion
cluster_t* clustersInit(int k, point_t points[], int n)
{
	cluster_t* clusters = (cluster_t*)malloc(k * sizeof(cluster_t));

	#pragma omp parallel for
	for (int i=0; i < k; i++)
	{
		vectorInit(&clusters[i].location, points[i].location.x, points[i].location.y, points[i].location.z);
		//clusters[i].pointsList = NULL;
		clusters[i].pointArray = NULL;
		clusters[i].sizeOfPointArray = 2*n/k;
		clusters[i].numOfPoints = 0;
	}
	return clusters;
}

//for each point find the nearest cluster
int findNearestCluster(cluster_t c[], int k, point_t p)
{
	double minDist = distance(c[0].location, p.location);
	int minIndex = 0;
	for (int i = 1; i < k; i++)
	{
		double tempDist = distance(c[i].location,p.location);
		if (tempDist < minDist) {
			minDist = tempDist;
			minIndex = i;
		}
	}
	return minIndex;
}

//for each cluster calculate its new location depending the amount of points in it
void defineNewClusterLocation(cluster_t c[], vector_t l[], int pointsInCluster[], int k)
{
	#pragma omp parallel for
	for (int i = 0; i < k; i++) {
		if (pointsInCluster[i] > 0)
		{
			c[i].location = l[i];
			c[i].location.x /= pointsInCluster[i];
			c[i].location.y /= pointsInCluster[i];
			c[i].location.z /= pointsInCluster[i];
		}
	}
}

//one iteration inside kmean algorithm
int kMeansIteration(cluster_t clusters[], int k, point_t points[], int pointToCluster[], int n)
{
	//create a vector array that will represent cluster's new location
	vector_t* newLocation = (vector_t*)calloc(k, sizeof(vector_t));
	#pragma omp parallel for
	for (int i = 0; i < k; i++)
		vectorInit(&newLocation[i], 0, 0, 0);

	int* pointsInCluster = (int*)calloc(k, sizeof(int));
	int isSame = 1;
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		int minIndex = findNearestCluster(clusters, k, points[i]);
		if (pointToCluster[i] != minIndex) {
			isSame = 0;
			pointToCluster[i] = minIndex;
		}
		#pragma omp atomic
		pointsInCluster[pointToCluster[i]]++;
		atomicAddVector(&newLocation[pointToCluster[i]], points[i].location);
	}
	if (isSame)
		return isSame;

	defineNewClusterLocation(clusters, newLocation, pointsInCluster, k);
	free(pointsInCluster);
	free(newLocation);
	return isSame;
}

cluster_t* kMeans(int n, int k, int limit, point_t p[])
{
	int* pointToCluster = (int*)calloc(n, sizeof(int));

	cluster_t* clusters = clustersInit(k, p, n);

	int isSame = 0;
	for (int i = 0; i < limit && !isSame; i++)
		isSame = kMeansIteration(clusters, k, p, pointToCluster, n);

	for (int i = 0; i < n; i++)
		addPoint(&clusters[pointToCluster[i]], &p[i]);
		//addElement(&clusters[pointToCluster[i]].pointsList, &p[i]);

	free(pointToCluster);
	return clusters;
}

double quality(cluster_t clusters[], int k)
{
	double* radius = (double*)calloc(k, sizeof(double));
	double q = 0;
	#pragma omp parallel for reduction(+: q)
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < clusters[i].numOfPoints; j++)
		{
			for (int k = j + 1; k < clusters[i].numOfPoints; k++)
			{
				double dist = distance(clusters[i].pointArray[j]->location, clusters[i].pointArray[k]->location);
				if (dist > radius[i])
					radius[i] = dist;
			}
		}
		for (int j = 0; j < k; j++)
		{
			if (i == j)
				continue;
			q += sqrt(radius[i]) / sqrt(distance(clusters[i].location, clusters[j].location));
		}
	}
	free(radius);
	return q / (k*(k - 1));
}
