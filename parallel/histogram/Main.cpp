#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <omp.h>
#include "mpi.h"

#include "Point.h"
#include "Cluster.h"

#define RESULT_TAG 0
#define FILE_PATH "C:/Users/sean9/Desktop/Points/5k.txt"

MPI_Datatype MPI_VECTOR;

extern cluster_t* CudaKMeans(int n, int k, int limit, point_t h_points[], double* h_qaulity);
extern void cudaInit();
extern void cudaAddPoints(point_t h_dstPoints[], point_t h_srcPoints[], int n, double dt);

int getDataFromFile(const char* path, int* n, int* k, int* limit, double* qm, double* t, double* dt, point_t** p)
{
	FILE* f;
	fopen_s(&f, path, "r");
	if (f == NULL)
	{
		printf_s("error reading from file");
		_flushall();
		return 0;
	}

	fscanf_s(f, "%d %d %lf %lf %d %lf", n, k, t, dt, limit, qm);//, limit, qm, t, dt);
	*p = (point_t*)calloc(*n, sizeof(point_t));

	double x, y, z;
	point_t* temp = *p;
	for (int i = 0; i < *n; i++) {
		fscanf_s(f, "\n%lf %lf %lf", &x, &y, &z);
		vectorInit(&temp[i].location, x, y, z);

		fscanf_s(f, " %lf %lf %lf", &x, &y, &z);
		vectorInit(&temp[i].speed, x, y, z);
	}
	fclose(f);
	return 1;
}

cluster_t* clustersInit(int k, point_t points[], int n)
{
	cluster_t* clusters = (cluster_t*)malloc(k * sizeof(cluster_t));

	//#pragma omp parallel for
	#pragma omp parallel for
	for (int i=0; i < k; i++)
	{
		vectorInit(&clusters[i].location, points[i].location.x, points[i].location.y, points[i].location.z);
		clusters[i].pointsList = NULL;
	}
	return clusters;
}

int kMeansIteration(cluster_t clusters[], int k, point_t points[], int pointToCluster[], int n)
{
	vector_t* newLocation = (vector_t*)calloc(k, sizeof(vector_t));
	#pragma omp parallel for
	for (int i = 0; i < k; i++)
		vectorInit(&newLocation[i], 0, 0, 0);

	int* pointsInCluster = (int*)calloc(k, sizeof(int));
	int isSame = 1;
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		//match each point to it's closest cluster
		double minDist = distance(clusters[0].location, points[i].location);
		int minIndex = 0;
		for (int j = 1; j < k; j++)
		{
			double tempDist = distance(clusters[j].location, points[i].location);
			if (tempDist < minDist) {
				minDist = tempDist;
				minIndex = j;
			}
		}
		if (pointToCluster[i] != minIndex) {
			isSame = 0;
			pointToCluster[i] = minIndex;
		}
		#pragma omp atomic
		pointsInCluster[pointToCluster[i]]++;
		addVector(&newLocation[pointToCluster[i]], points[i].location);
	}
	if (isSame)
		return isSame;

	#pragma omp parallel for
	for (int i = 0; i < k; i++) {
		if (pointsInCluster[i] > 0)
		{
			clusters[i].location = newLocation[i];
			clusters[i].location.x /= pointsInCluster[i];
			clusters[i].location.y /= pointsInCluster[i];
			clusters[i].location.z /= pointsInCluster[i];
		}
	}
	free(pointsInCluster);
	free(newLocation);
	return isSame;
}

double quality(cluster_t clusters1[], int k)
{
	//printTime(clusters1, k);
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
	return q / (k*(k-1));//((pow(k, 2) + k) / 2);
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

void movePoints(point_t p1[], point_t p2[], int n, double delta)
{
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		p1[i].location.x = p2[i].location.x + delta * p2[i].speed.x;
		p1[i].location.y = p2[i].location.y + delta * p2[i].speed.y;
		p1[i].location.z = p2[i].location.z + delta * p2[i].speed.z;
	}
}

vector_t* findFirstGoodCluster(int rank, int numprocs, int n, int k, int limit, double qm, double start, double t, double dt, point_t p[])
{
	//Recv non blocking message
	MPI_Request request;
	MPI_Status status;
	int flag;
	double timeRecv;
	MPI_Irecv(&timeRecv, 1, MPI_DOUBLE, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &request);

	//create variables for threads
	double result[3] = { INFINITY, INFINITY, INFINITY };
	point_t* points[2] = { NULL, NULL };
	cluster_t* clusters[2] = { NULL, NULL };


	double curTime = start;
	int succ = 0;

	omp_set_nested(1);
	omp_set_dynamic(1);
	#pragma omp parallel num_threads(2)
	{
		double myTime;
		int tid = omp_get_thread_num();
		points[tid] = (point_t*)calloc(n, sizeof(point_t));

		while (curTime <= t && curTime < result[2] && !succ)
		{
			if (clusters[tid] != NULL)
			{
				for (int i = 0; i < k; i++)
					freeArrayList(&clusters[tid][i].pointsList);
				free(clusters[tid]);
			}
			//each thread takes his time
			#pragma omp critical
			{
				myTime = curTime;
				curTime += dt;
			}

			double q;
			if (tid == 0)//cpu
			{
				movePoints(points[tid], p, n, myTime - start);
				clusters[tid] = kMeans(n, k, limit, points[tid]);
				q = quality(clusters[tid], k);
			}
			else
			{
				cudaAddPoints(points[tid], p, n, myTime - start);
				clusters[tid] = CudaKMeans(n, k, limit, points[tid], &q);
			}

			printf_s("rank = %d, thread = %d, time = %lf, quality = %lf\n", rank, tid, myTime, q);
			_flushall();

			if (q < qm)
			{
				result[tid] = myTime;
				succ = 1;
				break;
			}

			//check if message recived
			#pragma omp master
			{
				MPI_Test(&request, &flag, &status);
				if (flag)
				{
					printf_s("*****rank %d recived from %d) data = %lf\n", rank, status.MPI_SOURCE, timeRecv);
					result[2] = fmin(result[2], timeRecv);
					MPI_Irecv(&timeRecv, 1, MPI_DOUBLE, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &request);
				}
			}
		}
	}
	int minIndex;
	if (result[0] < result[1])
		minIndex = 0;
	else
		minIndex = 1;

	if (result[minIndex] < result[2])
	{
		for (int i = 0; i < numprocs; i++)
		{
			if (i != rank)
				MPI_Isend(&result[minIndex], 1, MPI_DOUBLE, i, RESULT_TAG, MPI_COMM_WORLD, &request);
		}
	}
	printf_s("p[%d] sends %lf\n", rank, result[minIndex]);
	double* bestResults = (double*)calloc(numprocs, sizeof(double));
	MPI_Gather(&result[minIndex], 1, MPI_DOUBLE,
		bestResults, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	vector_t* centers = (vector_t*)malloc(k * sizeof(vector_t));
	for (int i = 0; i < k; i++)
		centers[i] = clusters[minIndex][i].location;
	if (rank!=0)
		MPI_Isend(centers, k, MPI_VECTOR, 0, 0, MPI_COMM_WORLD, &request);

	if (rank == 0)
	{
		printf_s("p[0] recived:\n");
		for (int i = 0; i < numprocs; i++)
			printf_s("from p[%d] got %lf\n", i, bestResults[i]);
		minIndex = 0;
		for (int i = 1; i < numprocs; i++)
		{
			if (bestResults[i] < bestResults[minIndex])
				minIndex = i;
		}
		printf_s("time is %lf", bestResults[minIndex]);
		if (minIndex != 0)
			MPI_Recv(centers, k, MPI_VECTOR, minIndex, 0, MPI_COMM_WORLD, &status);
		return centers;
	}
}

void Bcast(int rank, int* n, int* k, int* limit, double* qm, double* t, double* dt, point_t** p)
{
	MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(k, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(limit, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Bcast(qm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	vector_t vector_tmp;
	MPI_Datatype MPI_Vector_type[3] = { MPI_DOUBLE, MPI_DOUBLE , MPI_DOUBLE };
	int vector_blocklen[3] = { 1,1,1 };
	MPI_Aint MPI_Vector_disp[3] = { (char*)&vector_tmp.x - (char*)&vector_tmp, 
		(char*)&vector_tmp.y - (char*)&vector_tmp, (char*)&vector_tmp.z - (char*)&vector_tmp };
	MPI_Type_create_struct(3, vector_blocklen, MPI_Vector_disp, MPI_Vector_type, &MPI_VECTOR);
	MPI_Type_commit(&MPI_VECTOR);

	point_t point_tmp;
	MPI_Datatype MPI_POINT;
	MPI_Datatype MPI_Point_type[2] = { MPI_VECTOR, MPI_VECTOR };
	int point_blocklen[2] = { 1,1 };
	MPI_Aint MPI_Point_disp[2] = { (char*)&point_tmp.location - (char*)&point_tmp,
		(char*)&point_tmp.speed - (char*)&point_tmp };
	MPI_Type_create_struct(2, point_blocklen, MPI_Point_disp, MPI_Point_type, &MPI_POINT);
	MPI_Type_commit(&MPI_POINT);

	if (rank != 0)
		*p = (point_t*)calloc(*n, sizeof(point_t));
	MPI_Bcast(*p, *n, MPI_POINT, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
	printf_s("%s", argv[1]);
	MPI_Init(&argc, &argv);
	cudaInit();

	int rank, numprocs;

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int n, k, limit; //number of points, number of clusters, iteration limit
	double qm, t, dt; //quality measure, end of time interval, time jump
	point_t* p;

	if (rank == 0)
		getDataFromFile(/*argv[1]*/FILE_PATH, &n, &k, &limit, &qm, &t, &dt, &p);
	
	Bcast(rank, &n, &k, &limit, &qm, &t, &dt, &p);
	printf_s("rank = %d, n = %d, k = %d, limit = %d, qm = %lf, t = %lf, dt = %lf\n", rank, n, k, limit, qm, t, dt);
	_flushall();

	movePoints(p, p, n, rank*dt);
	//cudaAddPoints(p, p, n, rank*dt);

	double t1 = MPI_Wtime();
	vector_t* centers = findFirstGoodCluster(rank, numprocs, n, k, limit, qm, rank*dt, t, numprocs*dt, p);
	if (rank == 0)
		for (int i = 0; i < k; i++)
			printf_s("\n%d) [%lf,%lf,%lf]", i, centers[i].x, centers[i].y, centers[i].z);
	double t2 = MPI_Wtime();
	printf_s("\ntime took to compute: %lf\n", t2 - t1);

	free(p);
	_flushall();
	MPI_Finalize();
	system("pause");
	return 0;
}