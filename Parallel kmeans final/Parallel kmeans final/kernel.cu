#include "kernel.cuh"

//https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
__device__ double atomicAddDouble(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

void cudaInit()
{
	cudaFree(0);
}

__device__ int getId()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

__device__ double cudaDist(vector_t v1, vector_t v2)
{
	return (v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z);
}

__global__ void initPointInCluster(int globalPointInCluster[], int k)
{
	int i = getId();
	if (i<k)
		globalPointInCluster[i] = 0;
}

__global__ void resetClusters(cluster_t clusters[], int pointInCluster[], int k)
{
	int i = getId();
	if (i < k && pointInCluster[i]>0)
	{
		clusters[i].location.x = 0;
		clusters[i].location.y = 0;
		clusters[i].location.z = 0;
	}
}

__global__ void addPointsToCluster(cluster_t clusters[], point_t points[], int pointToCluster[], int n)
{
	int i = getId();
	if (i < n)
	{
		atomicAddDouble(&clusters[pointToCluster[i]].location.x, points[i].location.x);
		atomicAddDouble(&clusters[pointToCluster[i]].location.y, points[i].location.y);
		atomicAddDouble(&clusters[pointToCluster[i]].location.z, points[i].location.z);
	}
}

__global__ void avrageClusters(cluster_t clusters[], int pointsInCluster[], int k)
{
	int i = getId();
	if (i < k && pointsInCluster[i]>0)
	{
		clusters[i].location.x /= pointsInCluster[i];
		clusters[i].location.y /= pointsInCluster[i];
		clusters[i].location.z /= pointsInCluster[i];
		pointsInCluster[i] = 0;
	}
}

__global__ void kMeansIteration(cluster_t clusters[], int pointInCluster[], int k, point_t points[], int pointToCluster[], int n, int* isSame)
{
	int i = getId();
	if (i < n)
	{
		double minDist = cudaDist(clusters[0].location, points[i].location);
		int minIndex = 0;
		for (int j = 1; j < k; j++)
		{
			double tempDist = cudaDist(clusters[j].location, points[i].location);
			minDist = fmin(minDist, tempDist);
			minIndex = tempDist == minDist ? j : minIndex;
		}
		if (pointToCluster[i] != minIndex) {
			*isSame = 0;
			pointToCluster[i] = minIndex;
		}
		atomicAdd(&pointInCluster[minIndex], 1);
	}
}

__global__ void cudaInitClusters(cluster_t clusters[], int k, point_t points[], int n)
{
	int i = getId();
	clusters[i].location.x = points[i].location.x;
	clusters[i].location.y = points[i].location.y;
	clusters[i].location.z = points[i].location.z;
	//clusters[i].pointsList = NULL;
	clusters[i].pointArray = NULL;
	clusters[i].sizeOfPointArray = 2*n/k;
	clusters[i].numOfPoints = 0;
}

__global__ void cudaAddPointsParallel(point_t points[], int n, double dt)
{
	int i = getId();
	if (i < n)
	{
		points[i].location.x += dt * points[i].speed.x;
		points[i].location.y += dt * points[i].speed.y;
		points[i].location.z += dt * points[i].speed.z;
		//points[i].location.squared = pow(points[i].location.x, 2) + pow(points[i].location.y, 2) + pow(points[i].location.z, 2);
		//points[i].location.dist = sqrt(points[i].location.squared);
	}
}

__global__ void cudaQuality(point_t points[], int pointToCluster[], double maxDist[], int n)
{
	int id = getId();
	if (id < n)
	{
		maxDist[id] = 0;
		for (int i = id + 1; i < n; i++)
		{
			if (pointToCluster[id] == pointToCluster[i])
			{
				double dist = cudaDist(points[id].location, points[i].location);
				maxDist[id] = dist > maxDist[id] ? dist : maxDist[id];
			}
		}
	}
}

void cudaMovePoints(point_t h_dstPoints[], point_t h_srcPoints[], int n, double dt)
{
	point_t* d_points;
	cudaMalloc(&d_points, n * sizeof(point_t));
	cudaMemcpy(d_points, h_srcPoints, n * sizeof(point_t), cudaMemcpyHostToDevice);

	cudaAddPointsParallel<<<n/ NUM_OF_THREADS + 1, NUM_OF_THREADS >>>(d_points, n, dt);

	cudaMemcpy(h_dstPoints, d_points, n * sizeof(point_t), cudaMemcpyDeviceToHost);
	cudaFree(d_points);
}

double CudaQuality(point_t d_points[], int d_pointToCluster[], int n, cluster_t d_clusters[], int k)
{
	int* h_pointToCluster = (int*)calloc(n, sizeof(int));
	cluster_t* h_clusters = (cluster_t*)calloc(k, sizeof(cluster_t));
	//maxDist
	double* h_maxDist = (double*)calloc(n, sizeof(double));
	double* d_maxDist;
	cudaMalloc(&d_maxDist, n * sizeof(double));

	cudaQuality <<<n / NUM_OF_THREADS + 1, NUM_OF_THREADS >>> (d_points, d_pointToCluster, d_maxDist, n);
	cudaMemcpy(h_clusters, d_clusters, k * sizeof(cluster_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pointToCluster, d_pointToCluster, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(h_maxDist, d_maxDist, n * sizeof(double), cudaMemcpyDeviceToHost);

	double* radius = (double*)calloc(k, sizeof(double));
	for (int i = 0; i < n; i++)
	{
		//printf_s("%lf\n", sqrt(h_maxDist[i]));
		if (h_maxDist[i] > radius[h_pointToCluster[i]])
			radius[h_pointToCluster[i]] = h_maxDist[i];
	}
	double qaulity = 0;
	for (int i = 0; i < k; i++)
	{
		//printf_s("%lf\n", sqrt(radius[i]));
		for (int j = 0; j < k; j++)
		{
			if (i == j)
				continue;
			qaulity += sqrt(radius[i]) / sqrt(distance(h_clusters[i].location, h_clusters[j].location));
		}
	}
	qaulity /= k * (k - 1);
	free(radius);
	free(h_maxDist);

	//maxDist
	cudaFree(d_maxDist);
	return qaulity;
}

cluster_t* cudaKMeans(int n, int k, int limit, point_t h_points[], double* h_qaulity)
{
	//points
	point_t* d_points;
	cudaMalloc(&d_points, n * sizeof(point_t));
	cudaMemcpy(d_points, h_points, n * sizeof(point_t), cudaMemcpyHostToDevice);

	//clusters
	cluster_t* h_clusters = (cluster_t*)calloc(k, sizeof(cluster_t));
	cluster_t* d_clusters;
	cudaMalloc(&d_clusters, k * sizeof(cluster_t));
	cudaInitClusters<<<1,k>>>(d_clusters, k, d_points, n);

	//pointToCluster
	int* h_pointToCluster = (int*)calloc(n, sizeof(int));
	int* d_pointToCluster;
	cudaMalloc(&d_pointToCluster, n * sizeof(int));

	//pointsInCluster
	int* h_pointsInCluster = (int*)calloc(k, sizeof(int));
	int* d_pointsInCluster;
	cudaMalloc(&d_pointsInCluster, k * sizeof(int));

	//isSame
	const int one = 1;
	int *d_isSame, h_isSame = 0;
	cudaMalloc(&d_isSame, sizeof(int));
	cudaMemcpy(d_isSame, &one, sizeof(int), cudaMemcpyHostToDevice);

	initPointInCluster <<<1, k >> >(d_pointsInCluster, k);
	for (int i = 0; i < limit && !h_isSame; i++)
	{
		cudaMemcpy(d_isSame, &one, sizeof(int), cudaMemcpyHostToDevice);
		
		kMeansIteration <<<n/ NUM_OF_THREADS + 1, NUM_OF_THREADS >> > (d_clusters, d_pointsInCluster, k, d_points, d_pointToCluster, n, d_isSame);
		resetClusters <<<1, k >>> (d_clusters, d_pointsInCluster, k);
		addPointsToCluster<<<n / NUM_OF_THREADS + 1, NUM_OF_THREADS >>>(d_clusters, d_points, d_pointToCluster, n);
		avrageClusters<<<1, k>>>(d_clusters, d_pointsInCluster, k);
		
		cudaMemcpy(&h_isSame, d_isSame, sizeof(int), cudaMemcpyDeviceToHost);
	}
	cudaMemcpy(h_clusters, d_clusters, k * sizeof(cluster_t), cudaMemcpyDeviceToHost);
	if (QUALITY_DEVICE == GPU)
	{
		*h_qaulity = CudaQuality(d_points, d_pointToCluster, n, d_clusters, k);
	}
	else if (QUALITY_DEVICE == CPU)
	{
		cudaMemcpy(h_pointToCluster, d_pointToCluster, n * sizeof(int), cudaMemcpyDeviceToHost);
		for (int i = 0; i < n; i++)
			addPoint(&h_clusters[h_pointToCluster[i]], &h_points[i]);
			//addElement(&h_clusters[h_pointToCluster[i]].pointsList, &h_points[i]);
		*h_qaulity = quality(h_clusters, k);
	}
	//points
	cudaFree(d_points);
	//clusters
	cudaFree(d_clusters);
	//pointToCluster
	free(h_pointToCluster);
	cudaFree(d_pointToCluster);
	//pointInCluster
	free(h_pointsInCluster);
	cudaFree(d_pointsInCluster);
	//isSame
	cudaFree(d_isSame);
	return h_clusters;
}