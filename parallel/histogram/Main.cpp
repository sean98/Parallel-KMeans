#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <omp.h>
#include "mpi.h"

#include "Point.h"
#include "Cluster.h"

#define FILE_PATH "C:\\Users\\sean9\\source\\repos\\Parallel Final\\testCreator\\newTest.txt"//C:/Users/sean9/Desktop/newTest (2) (1).txt"//"C:/Users/sean9/Desktop/newTest (2).txt"
void printTime(cluster_t clusters[], int k);

extern int cudaKmeanIteration(cluster_t clusters[], int k, point_t points[], int pointToCluster[], int n);
extern cluster_t* CudaKMeans(int n, int k, int limit, point_t h_points[], double* h_qaulity);
extern void cudaInit();
extern void cudaAddPoints(point_t h_dstPoints[], point_t h_srcPoints[], int n, double dt);

void getDataFromFile(const char* path, int* n, int* k, int* limit, double* qm, double* t, double* dt, point_t** p)
{
	FILE* f;
	fopen_s(&f, path, "r");
	if (f == NULL)
	{
		printf_s("error reading from file");
		_flushall();
		return;
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
		//initArrayList(&clusters[i].pointsList, n / k);
		//addElement(&clusters[i].pointsList, &points[i]);
	}
	return clusters;
}

int kMeansIteration(cluster_t clusters[], int k, point_t points[], int pointToCluster[], int n)
{
	vector_t* newLocation = (vector_t*)calloc(k, sizeof(vector_t));
	#pragma omp parallel for
	for (int i = 0; i < k; i++)
	{
		vectorInit(&newLocation[i], 0, 0, 0);
		//initArrayList(&clusters->pointsList, n / k);
	}

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
		//addElement(&clusters[pointToCluster[i]].pointsList, &points[i]);
		addVector(&newLocation[pointToCluster[i]], points[i].location);
	}
	if (isSame)
		return isSame;

	#pragma omp parallel for
	for (int i = 0; i < k; i++) {
		clusters[i].location = newLocation[i];
		clusters[i].location.x /= pointsInCluster[i];
		clusters[i].location.y /= pointsInCluster[i];
		clusters[i].location.z /= pointsInCluster[i];
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
		//printf_s("cpu %lf\n", sqrt(radius[i]));
		for (int j = 0; j < k; j++)
		{
			if (i == j)
				continue;
			q += sqrt(radius[i]) / sqrt(distance(clusters1[i].location, clusters1[j].location));
		}
	}
	//for (int i=0;i<k;i++)
	//	printf_s("%lf with %d points\n", radius[i], pointInCluster[i]);
	free(radius);
	return q / (k*(k-1));//((pow(k, 2) + k) / 2);
}

//void printTime(cluster_t clusters[], int k)
//{
//	for (int i = 0; i < k; i++) 
//	{
//		printf_s("\t");
//		printVector(clusters[i].location);
//		printf_s("\t\n\t{\n");
//
//		for (int j = 0; j < clusters->pointsList->logicalSize; j++)
//		{
//			printf_s("\t|\t");
//			printVector(((point_t*)clusters[i].pointsList->data[j])->location);
//			printf_s("\t\n");
//		}
//
//		/*
//		node_t* temp = clusters[i].pointsList;
//		while (temp)
//		{
//			printf_s("\t|\t");
//			printVector(((point_t*)temp->data)->location);
//			printf_s("\t\n");
//			temp = temp->next;
//		}
//		*/
//		printf_s("\t}\n\n");
//	}
//}

cluster_t* kMeans(int n, int k, int limit, point_t p[])
{
	int* pointToCluster = (int*)calloc(n, sizeof(int));

	cluster_t* clusters = clustersInit(k, p, n);

	int isSame = 0;
	for (int i = 0; i < limit && !isSame; i++)
	{
		isSame = kMeansIteration(clusters, k, p, pointToCluster, n);
		//isSame = cudaKmeanIteration(clusters, k, p, pointToCluster, n);
	}

	for (int i = 0; i < n; i++)
		addElement(&clusters[pointToCluster[i]].pointsList, &p[i]);

	free(pointToCluster);
	return clusters;
}

void findFirstGoodCluster(int rank, int numprocs, int n, int k, int limit, double qm, double start, double t, double dt, point_t p[])
{
	double result[3] = { INFINITY, INFINITY, INFINITY };
	double curTime = start;
	int succ = 0;
	double minTime = 0, minQuality = INFINITY;

	//double delta = 0;
	//cluster_t* myClusters;
	//for (int i = 0; i < n; i++)
	//{
	//	p[i].location.x += (102.764046) * p[i].speed.x;
	//	p[i].location.y += (102.764046) * p[i].speed.y;
	//	p[i].location.z += (102.764046) * p[i].speed.z;
	//}
	//for (int i = 0; i < 100; i++)
	//{
	//	double q1, q2;

	//	myClusters = kMeans(n, k, limit, p);
	//	q1 = quality(myClusters, k);

	//	myClusters = CudaKMeans(n, k, limit, p, &q2);
	//	delta = fmax(delta, fabs(q1 - q2));
	//	printf_s("%d) %lf\n", i, delta);
	//}

	omp_set_nested(1);
	omp_set_dynamic(1);

	cluster_t* c1;
	cluster_t* c2;
	#pragma omp parallel num_threads(2)
	{
		int tid = omp_get_thread_num();
		point_t *myPoints = (point_t*)calloc(n, sizeof(point_t));
		cluster_t* myClusters;
		double myTime;

		while (curTime <= t && !succ)
		{
			#pragma omp critical
			{
				myTime = curTime;
				curTime += dt;
			}
			double q;
			if (tid == 0)//cpu
			{
				for (int i = 0; i < n; i++)
				{
					myPoints[i].location.x = p[i].location.x + (myTime - start) * p[i].speed.x;
					myPoints[i].location.y = p[i].location.y + (myTime - start) * p[i].speed.y;
					myPoints[i].location.z = p[i].location.z + (myTime - start) * p[i].speed.z;
				}
				myClusters = kMeans(n, k, limit, myPoints);
				q = quality(myClusters, k);
			}
			else
			{
				cudaAddPoints(myPoints, p, n, myTime - start);
				myClusters = CudaKMeans(n, k, limit, myPoints, &q);
			}/*

			double q;
			#pragma omp critical
			{
				if (tid == 0)
					myClusters = kMeans(n, k, limit, myPoints);
				q = quality(myClusters, k);
			}*/

			printf_s("rank = %d, thread = %d, time = %lf, quality = %lf\n", rank, omp_get_thread_num(), myTime, q);

			if (q < minQuality)
			{
				minQuality = q;
				minTime = myTime;
			}

			if (q < qm)
			{
				//printf_s("thread %d found cluster at %lf with qaulity %lf\n", omp_get_thread_num(), myTime, q);
				result[tid] = myTime;
				succ = 1;
				break;
			}
			//printf_s("thread %d:  time %lf qaulity %lf\n\n", omp_get_thread_num(), myTime, q, curTime);
			for (int i = 0; i < k; i++)
				freeArrayList(&myClusters[i].pointsList);
			free(myClusters);
			_flushall();
		}
	}
	double bestResult = fmin(result[0], fmin(result[1], result[2]));

	printf_s("rank %d: %lf\n", rank, bestResult);
	double* bestResults = (double*)calloc(numprocs, sizeof(double));

	MPI_Gather(&bestResult, 1, MPI_DOUBLE,
		bestResults, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (rank == 0)
	{
		for (int i = 0; i < numprocs; i++)
		{
			bestResult = fmin(bestResult, bestResults[i]);
			printf_s("best result[%d]=%lf\n", i, bestResults[i]);
		}
	}
	if (rank==0)
		printf_s("time is %lf", bestResult);
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
	MPI_Datatype MPI_Vector;
	MPI_Datatype MPI_Vector_type[3] = { MPI_DOUBLE, MPI_DOUBLE , MPI_DOUBLE };
	int vector_blocklen[3] = { 1,1,1 };
	MPI_Aint MPI_Vector_disp[3] = { (char*)&vector_tmp.x - (char*)&vector_tmp, 
		(char*)&vector_tmp.y - (char*)&vector_tmp, (char*)&vector_tmp.z - (char*)&vector_tmp };
	MPI_Type_create_struct(3, vector_blocklen, MPI_Vector_disp, MPI_Vector_type, &MPI_Vector);
	MPI_Type_commit(&MPI_Vector);

	point_t point_tmp;
	MPI_Datatype MPI_Point;
	MPI_Datatype MPI_Point_type[2] = { MPI_Vector, MPI_Vector };
	int point_blocklen[2] = { 1,1 };
	MPI_Aint MPI_Point_disp[2] = { (char*)&point_tmp.location - (char*)&point_tmp,
		(char*)&point_tmp.speed - (char*)&point_tmp };
	MPI_Type_create_struct(2, point_blocklen, MPI_Point_disp, MPI_Point_type, &MPI_Point);
	MPI_Type_commit(&MPI_Point);

	if (rank != 0)
		*p = (point_t*)calloc(*n, sizeof(point_t));
	MPI_Bcast(*p, *n, MPI_Point, 0, MPI_COMM_WORLD);
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
	{
		getDataFromFile(/*argv[1]*/FILE_PATH, &n, &k, &limit, &qm, &t, &dt, &p);
	}
	Bcast(rank, &n, &k, &limit, &qm, &t, &dt, &p);
	printf_s("rank = %d, n = %d, k = %d, limit = %d, qm = %lf, t = %lf, dt = %lf\n", rank, n, k, limit, qm, t, dt);
	_flushall();

	for (int i = 0; i < n; i++)
	{
		p[i].location.x += (rank*dt) * p[i].speed.x;
		p[i].location.y += (rank*dt) * p[i].speed.y;
		p[i].location.z += (rank*dt) * p[i].speed.z;
	}

	double t1 = MPI_Wtime();
	findFirstGoodCluster(rank, numprocs, n, k, limit, qm, rank*dt, t, numprocs*dt, p);
	double t2 = MPI_Wtime();
	printf_s("\ntime took to compute: %lf\n", t2 - t1);

	free(p);
	_flushall();
	MPI_Finalize();
	system("pause");
	return 0;
}