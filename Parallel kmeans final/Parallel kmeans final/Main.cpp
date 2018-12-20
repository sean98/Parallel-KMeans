#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <omp.h>
#include "mpi.h"

#include "Cluster.h"
#include "Point.h"
#include "KMean.h"
#include "kernel.cuh"

#define CPU 0
#define CUDA 1

#define NOTIFY_TAG 0
#define CENTERS_TAG 1

MPI_Datatype MPI_VECTOR, MPI_POINT;
int rank, numprocs;

//This method initiate two MPI_Datatype(MPI_Vector, MPI_Point)
//to enable sending vector_t & point_t structs through mpi methods.
void initDataType()
{
	//init MPI_VECTOR
	vector_t vector_tmp;
	MPI_Datatype MPI_Vector_type[3] = { MPI_DOUBLE, MPI_DOUBLE , MPI_DOUBLE };
	int vector_blocklen[3] = { 1,1,1 };
	MPI_Aint MPI_Vector_disp[3] = { (char*)&vector_tmp.x - (char*)&vector_tmp,
		(char*)&vector_tmp.y - (char*)&vector_tmp, (char*)&vector_tmp.z - (char*)&vector_tmp };
	MPI_Type_create_struct(3, vector_blocklen, MPI_Vector_disp, MPI_Vector_type, &MPI_VECTOR);
	MPI_Type_commit(&MPI_VECTOR);

	//init MPI_POINT
	point_t point_tmp;
	MPI_Datatype MPI_Point_type[2] = { MPI_VECTOR, MPI_VECTOR };
	int point_blocklen[2] = { 1,1 };
	MPI_Aint MPI_Point_disp[2] = { (char*)&point_tmp.location - (char*)&point_tmp,
		(char*)&point_tmp.speed - (char*)&point_tmp };
	MPI_Type_create_struct(2, point_blocklen, MPI_Point_disp, MPI_Point_type, &MPI_POINT);
	MPI_Type_commit(&MPI_POINT);
}

//Move each point location from srcPoints array to dstPoints by delta*velocity.
//This method accelerated by OpenMP.
void movePoints(point_t dstPoints[], point_t srcPoints[], int n, double delta)
{
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		dstPoints[i].location.x = srcPoints[i].location.x + delta * srcPoints[i].speed.x;
		dstPoints[i].location.y = srcPoints[i].location.y + delta * srcPoints[i].speed.y;
		dstPoints[i].location.z = srcPoints[i].location.z + delta * srcPoints[i].speed.z;
	}
}

//Print and write to file the found centers and first time with it's quality.
void writeCentersToFile(char* outPath, vector_t centers[], int k, double time, double quality)
{
	FILE* f;
	fopen_s(&f, outPath, "w");
	if (time != INFINITY)
	{
		fprintf_s(f, "First occurrence t = %lf  with q = %lf\n", time, quality);
		printf_s("First occurrence t = %lf  with q = %lf\n", time, quality);

		fprintf_s(f, "Centers of the clusters:\n");
		printf_s("Centers of the clusters:\n");

		for (int i = 0; i < k; i++)
		{
			fprintf_s(f, "%lf %lf %lf\n", centers[i].x, centers[i].y, centers[i].z);
			printf_s("%.3lf %.3lf %.3lf\n", centers[i].x, centers[i].y, centers[i].z);
		}
	}
	else
	{
		fprintf_s(f, "no good clusters");
		printf_s("no good clusters");
	}
	fclose(f);
}

//Read all relevant data from file.
int getDataFromFile(const char* path, int* n, int* k, int* limit, double* qm, double* t, double* dt, point_t** p)
{
	FILE* f;
	fopen_s(&f, path, "r");
	if (f == NULL)
	{
		printf_s("error reading from file\n");
		_flushall();
		return 0;
	}

	fscanf_s(f, "%d %d %lf %lf %d %lf", n, k, t, dt, limit, qm);
	*p = (point_t*)calloc(*n, sizeof(point_t));

	//read points
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

//Each proccess send his first time to all others proccess
//in order to help them finish earlier by lower the upper time boundary
void notifyBestTime(double timeResult[3])
{
	MPI_Request request;
	MPI_Status status;

	int minIndex = timeResult[0] < timeResult[1] ? 0 : 1;
	//the proccess check if his time is relevant to be sent by checking if his 
	//time is smaller than anything he recieved from any other proccess
	if (timeResult[minIndex] < timeResult[2])
	{
		for (int i = 0; i < numprocs; i++)
		{
			if (i != rank)
				MPI_Isend(&timeResult[minIndex], 1, MPI_DOUBLE, i, NOTIFY_TAG, MPI_COMM_WORLD, &request);
		}
	}
}

//this method is responsible for bringing the centers of
//the first good time to proccess 0
vector_t* getBestCenters(double timeResult[3], double finalQuality[2], cluster_t* clusters[2], int k, double backResult[2])
{
	//prepare all data that may be sent
	int minIndex = timeResult[0] < timeResult[1] ? 0 : 1;
	double sendData[2] = { timeResult[minIndex], finalQuality[minIndex] };
	double* bestResults = (double*)calloc(2 * numprocs, sizeof(double));
	//create array of vector to represent centers of clusters
	vector_t* centers = (vector_t*)malloc(k * sizeof(vector_t));
	for (int i = 0; i < k; i++)
		centers[i] = clusters[minIndex][i].location;

	for (int tid=0;tid<2;tid++)
	if (clusters[tid] != NULL)
	{
		for (int i = 0; i < k; i++)
			free(clusters[tid][i].pointArray);
		free(clusters[tid]);
	}

	//gather all data (time and quality) at proccess 0
	MPI_Gather(sendData, 2, MPI_DOUBLE, bestResults, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		//print all data time recieved
		printf_s("\np[0] recived:\n");
		for (int i = 0; i < numprocs; i ++)
			printf_s("from p[%d] got %lf\n", i, bestResults[2*i]);
		printf_s("\n");
		_flushall();

		//find which proccess have the first good centers
		minIndex = 0;
		for (int i = 1; i < numprocs; i++)
			if (bestResults[2 * i] < bestResults[2 * minIndex])
				minIndex = i;
		//update back result
		backResult[0] = bestResults[2 * minIndex];
		backResult[1] = bestResults[2 * minIndex + 1];
		//get the centers off the wanted proccess
		MPI_Status status;
		if (minIndex != 0)
			MPI_Recv(centers, k, MPI_VECTOR, minIndex, CENTERS_TAG, MPI_COMM_WORLD, &status);
	}
	else
	{
		//All proccess sends their centers to procces 0 but only 1 (p[minIndex]) will be taken
		MPI_Request request;
		MPI_Isend(centers, k, MPI_VECTOR, 0, CENTERS_TAG, MPI_COMM_WORLD, &request);
	}
	//create barrier to assert no changes will be made to centers before send is finished
	free(bestResults);
	MPI_Barrier(MPI_COMM_WORLD);
	return centers;
}

//The method return to proccess 0 first centers that statisfies q<qm
vector_t* findFirstGoodCluster(int n, int k, int limit, double qm, double start, double t, double dt, point_t p[], double backResult[2])
{
	//create a non blocking Recv
	MPI_Request request;
	MPI_Status status;
	int flag;
	double timeRecv;
	MPI_Irecv(&timeRecv, 1, MPI_DOUBLE, MPI_ANY_SOURCE, NOTIFY_TAG, MPI_COMM_WORLD, &request);

	//create variables for 2 threads
	double timeFound[3] = { INFINITY, INFINITY, INFINITY };//index 2 is timeRecived from other proccess
	double qualityFound[2] = { INFINITY, INFINITY };
	point_t* points[2] = { NULL, NULL };
	cluster_t* clusters[2] = { NULL, NULL };

	double curTime = start;
	int succ = 0;
	omp_set_nested(1);
	omp_set_dynamic(0);
	omp_set_num_threads(omp_get_max_threads() / 2);
	//This block creates two thread. one will calculate the 
	//kmean on the cpu and the second will calculte it on the gpu
	#pragma omp parallel num_threads(2)
	{
		double myTime;
		int tid = omp_get_thread_num();
		points[tid] = (point_t*)calloc(n, sizeof(point_t));

		while (curTime <= t && curTime < timeFound[2] && !succ)
		{
			//free clusters if they are full from last iteration
			if (clusters[tid] != NULL)
			{
				for (int i = 0; i < k; i++)
					free(clusters[tid][i].pointArray);
				free(clusters[tid]);
			}
			//each thread takes his time
			#pragma omp critical
			{
				myTime = curTime;
				curTime += dt;
			}

			//calculate kmean and quality
			double q;
			if (tid == 0)
			{
				movePoints(points[tid], p, n, myTime);
				clusters[tid] = kMeans(n, k, limit, points[tid]);
				q = quality(clusters[tid], k);
			}
			else
			{
				cudaMovePoints(points[tid], p, n, myTime);
				clusters[tid] = cudaKMeans(n, k, limit, points[tid], &q);
			}

			printf_s("rank = %d, thread = %d, time = %lf, quality = %lf\n", rank, tid, myTime, q);
			_flushall();

			if (q < qm)
			{
				qualityFound[tid] = q;
				timeFound[tid] = myTime;
				succ = 1;
				break;
			}

			//check if another proccess sent time where q<qm to lower the upper buondary
			MPI_Test(&request, &flag, &status);
			if (flag)
			{
				timeFound[2] = fmin(timeFound[2], timeRecv);
				MPI_Irecv(&timeRecv, 1, MPI_DOUBLE, MPI_ANY_SOURCE, NOTIFY_TAG, MPI_COMM_WORLD, &request);
			}
			
		}
	}
	free(points[0]);
	free(points[1]);
	notifyBestTime(timeFound);
	return getBestCenters(timeFound, qualityFound, clusters, k, backResult);
}

//Broadcast all paramaters to all proccesses
void Bcast(int* n, int* k, int* limit, double* qm, double* t, double* dt, point_t** p)
{
	MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(k, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(limit, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Bcast(qm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank != 0)
		*p = (point_t*)calloc(*n, sizeof(point_t));
	MPI_Bcast(*p, *n, MPI_POINT, 0, MPI_COMM_WORLD);
}


//The main should get 3 argument (including him self).
//The other two are the input file and the outpu location
int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	if (argc != 3)
	{
		printf_s("not enough arguments to start\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
	}
	else if(rank==0)
		printf("input=%s\noutput=%s\n", argv[0], argv[1], argv[2]);
	cudaInit();

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int n, k, limit; //number of points, number of clusters, iteration limit
	double qm, t, dt; //quality measure, end of time interval, time jump
	point_t* p;

	if (rank == 0)
		if (!getDataFromFile(argv[1], &n, &k, &limit, &qm, &t, &dt, &p))
			MPI_Abort(MPI_COMM_WORLD, 0);
	initDataType();
	Bcast(&n, &k, &limit, &qm, &t, &dt, &p);
	printf_s("rank = %d, n = %d, k = %d, limit = %d, qm = %lf, t = %lf, dt = %lf\n", rank, n, k, limit, qm, t, dt);
	_flushall();

	double t1 = MPI_Wtime();

	double backResult[2];
	vector_t* centers = findFirstGoodCluster(n, k, limit, qm, rank*dt, t, numprocs*dt, p, backResult);
	if (rank == 0)
	{
		writeCentersToFile(argv[2], centers, k, backResult[0], backResult[1]);
		double t2 = MPI_Wtime();
		printf_s("\ntime took to compute: %lf\n", t2 - t1);
		_flushall();
	}

	free(p);
	free(centers);
	MPI_Finalize();
	system("pause");
	return 0;
}