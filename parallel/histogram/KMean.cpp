#include "KMean.h"

cluster_t* clustersInit(int k, point_t points[], int n)
{
	cluster_t* clusters = (cluster_t*)malloc(k * sizeof(cluster_t));

	#pragma omp parallel for
	for (int i=0; i < k; i++)
	{
		vectorInit(&clusters[i].location, points[i].location.x, points[i].location.y, points[i].location.z);
		clusters[i].pointsList = NULL;
	}
	return clusters;
}

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
		addElement(&clusters[pointToCluster[i]].pointsList, &p[i]);

	free(pointToCluster);
	return clusters;
}

double quality(cluster_t clusters1[], int k)
{
	double* radius = (double*)calloc(k, sizeof(double));
	int* pointInCluster = (int*)calloc(k, sizeof(int));
	double q = 0;
	int counter = 0;
	#pragma omp parallel for reduction(+: q)
	for (int i = 0; i < k; i++)
	{
		node_t* head = clusters1[i].pointsList;
		while (head)
		{
			pointInCluster[i]++;
			node_t* run = head->next;
			while (run)
			{
				double dist = distance(((point_t*)head->data)->location,
					((point_t*)run->data)->location);
				radius[i] = fmax(radius[i], dist);
				run = run->next;
			}
			head = head->next;
		}
		for (int j = 0; j < k; j++)
		{
			if (i == j)
				continue;
			q += sqrt(radius[i]) / sqrt(distance(clusters1[i].location, clusters1[j].location));
		}
	}
	free(radius);
	return q / (k*(k - 1));
}
