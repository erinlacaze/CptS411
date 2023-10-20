/* Cpt S 411, Introduction to Parallel Computing
* School of EECS, WSU
*/
//#define __DEBUG__
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

#define BIGPRIME 93563

struct timeval t1, t2;
int communication_time = 0, generation_time = 0, avg_generation_time = 0;

void GenerateInitialGoL(int rank,  int n,  int p,  int *cell)
{
	MPI_Status status;

	// 1) Rank 0 generates p different random numbers
	if (rank == 0)
	{
		
		for (int i = 1; i < p; i++)
		{
			int seed = (rand() % BIGPRIME) + 1;

			// send seed
			gettimeofday(&t1, NULL);
            MPI_Send(&seed, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
	    	gettimeofday(&t2, NULL);

	    	communication_time += (t2.tv_sec-t1.tv_sec)*1000000 + (t2.tv_usec-t1.tv_usec);
		}

		// 2) generates distinct sequence of n^2 / p random values
		int seed = (rand() % BIGPRIME) + 1;

		for (int i = 0; i < n * n / p; i++)
		{
			if ((rand() % seed) % 2) 
			{
				cell[i] = 0; // status == alive
			}
			else
			{
				cell[i] = 1; // status == dead
			}
		}
	}

	// do 2) for each rank
	else  
	{
		int seed;

		// get seed
		gettimeofday(&t1, NULL);
		MPI_Recv(&seed,1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        gettimeofday(&t2, NULL);

        communication_time += (t2.tv_sec-t1.tv_sec)*1000000 + (t2.tv_usec-t1.tv_usec);

        for ( int i = 0; i < n * n / p; i++)
		{
			if ((rand() % seed) % 2)
			{
				cell[i] = 0; // status == alive
			} 
			else
			{
				cell[i] = 1; // status == dead
			}
		}
	}

}

// runs GoL game for G generations
void Simulate( int rank,  int n,  int p,  int *cell)
{
	MPI_Status status;

	// top row of current sub-matrix
	int *top = ( int *)malloc(n * sizeof( int));
	// bottom row of current sub-matrix
	int *bottom = ( int *)malloc(n * sizeof( int));
	// last row of prev rank
	int *prev = ( int *)malloc(n * sizeof( int));
	// first row of next rank
	int *next = ( int *)malloc(n * sizeof( int));
	

	for (int i = 0; i < n; i++)
	{
		top[i] = cell[i]; // copies the top row of current sub-matrix
		bottom[i] = cell[((n * n) / p) - n  + i]; // copies the bottom row of current sub-matrix
	}

	// communication time to get top, bottom, next, and prev ptrs
	gettimeofday(&t1, NULL);
    MPI_Send(top, n, MPI_INT, (rank - 1 + p) % p, 0, MPI_COMM_WORLD);
    MPI_Send(bottom, n, MPI_INT, (rank + 1) % p, 0, MPI_COMM_WORLD);

    MPI_Recv(next, n, MPI_INT,(rank + 1) % p, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(prev, n, MPI_INT,(rank - 1 + p) % p, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    gettimeofday(&t2, NULL);

    communication_time += (t2.tv_sec-t1.tv_sec)*1000000 + (t2.tv_usec-t1.tv_usec);

    for (int i = 0; i < n * n / p; i++)
	{
    	cell[i] = DetermineState(i, next, prev, cell, n, p);
    }

	// deallocate memory
    free(prev);
    free(next);
    free(top);
    free(bottom);
}

int DetermineState( int pos,  int *next,  int *prev,  int *cell,  int n,  int p){

	int N, S, E, W, NE, NW, SE, SW, sum, state;
	if(pos < n)
	{
		// top of sub-matrix; use last row of previous rank;
		if (pos == 0) // first
		{
			N = prev[pos];
			NW = prev[n - 1];
			NE = prev[pos + 1];
			S = cell[pos + n];
			SW = cell[pos + n + n - 1];
			SE = cell[pos + n + 1];
			E = cell[pos + 1];
			W = cell[n - 1];
		}
		else if (pos == n - 1) // last
		{
			N = prev[pos];
			NW = prev[pos - 1];
			NE = prev[0];
			S = cell[pos + n];
			SW = cell[pos + n - 1];
			SE = cell[pos + 1];
			E = cell[0];
			W = cell[pos - 1];
		}
		else // all others
		{
			N = prev[pos];
			NW = prev[pos - 1];
			NE = prev[pos + 1];
			S = cell[pos + n];
			SW = cell[pos + n - 1];
			SE = cell[pos + n + 1];
			E = cell[pos + 1];
			W = cell[pos - 1];
		}
	}
	else if (pos > (n * n / p) - 1 - n)
	{
		// bottom of sub-matrix; use first row of next rank;
		if (pos % n == 0) // first
		{
			N = cell[pos - n];
			NW = cell[pos - 1];
			NE = cell[pos - n + 1];
			S = next[0];
			SW = next[n - 1];
			SE = next[1];
			E = cell[pos + 1];
			W = cell[pos + n - 1];
		}
		else if (pos % n == n - 1) // last
		{
			N = cell[pos - n];
			NW = cell[pos - n - 1];
			NE = cell[pos - n - n + 1];
			S = next[n - 1];
			SW = next[n - 1 - 1];
			SE = next[0];
			E = cell[pos - n + 1];
			W = cell[pos - 1];
		}
		else // all others
		{
			N = cell[pos - n];
			NW = cell[pos - n - 1];
			NE = cell[pos - n + 1];
			S = next[pos % n];
			SW = next[pos % n - 1];
			SE = next[pos % n + 1];
			E = cell[pos + 1];
			W = cell[pos - 1];
		}
	}
	else 
	{
		// middle of sub-matrix; just use cell;
		if (pos % n == 0) // first
		{
			N = cell[pos - n];
			NW = cell[pos - 1];
			NE = cell[pos - n + 1];
			S = cell[pos + n];
			SW = cell[pos + n + n - 1];
			SE = cell[pos + n + 1];
			E = cell[pos + 1];
			W = cell[pos + n - 1];
		}
		else if (pos % n == n - 1) // last
		{
			N = cell[pos - n];
			NW = cell[pos - n - 1];
			NE = cell[pos - n - n + 1];
			S = cell[pos + n];
			SW = cell[pos + n - 1];
			SE = cell[pos + 1];
			E = cell[pos - n + 1];
			W = cell[pos - 1];
		}
		else // all others
		{
			N = cell[pos - n];
			NW = cell[pos - n - 1];
			NE = cell[pos - n + 1];
			S = cell[pos + n];
			SW = cell[pos + n - 1];
			SE = cell[pos + n + 1];
			E = cell[pos + 1];
			W = cell[pos - 1];
		}
	}

	sum = N + NW + NE + S + SW + SE + E + W;

	// less than 3 living neighbors
	if (sum > -1 && sum < 3)
	{
		state = 0; // state == dead
	}
	// more than 5 living neighbors
	if (sum > 5)
	{
		state = 0; // state == dead
	}
	// between 3 and 5 living neighbors
	if (sum > 2 && sum < 6)
	{
		state = 1;
	}

	return state;
}

void DisplayGoL( int rank,  int n,  int p,  int *cell)
{
	MPI_Status status;

	if (rank == 0)
	{
		int *matrix = (int *)malloc(sizeof(int) * n * n); // allocate memory for entire matrix

		for (int i = 0; i < n * (( int) n / p); i++) // copy sub-matrix to entire matrix
		{
			matrix[i] = cell[i];
		}

		for (int i = 1; i < p; i++) 
		{
			int *temp = (int *)malloc(sizeof( int) * n * ((int)n / p)); // temp matrix to populate entire matrix

			MPI_Recv(temp, n * ((int) n / p), MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			int first = i * n * ((int) n / p);

			for (int j = 0; j < n * ((int) n / p); j++)
			{
				matrix[first + j] = temp[j];
			}

			free(temp); // deallocate for next proc
		}	

		// DISPLAY
		for ( int i = 0; i < n * n; i++)
		{
			if (i % n == 0) // new row
			{
				printf("\n");
			}
			printf(" %d ", matrix[i]);
		}	
		printf("\n\n");
	}
	else 
	{
		MPI_Send(cell, n * ((int) n / p), MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
}

int main(int argc, char *argv[])
{
	int rank, p;
	struct timeval g1, g2;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	assert(argc == 3); // correct parameters
	int n = atoi(argv[1]); // size of matrix
    int G = atoi(argv[2]); // how many generations

	assert(p >= 1); // there is at least one process
	assert(n > p);	
    assert(n%p == 0); // n is divisible by p

    int *cell = (int*)malloc(sizeof(int) * n * ((int) n / p)); // allocate memory for sub-matrix
    
    gettimeofday(&g1, NULL);
    GenerateInitialGoL(rank, n, p, cell); // generate initial matrix
    gettimeofday(&g2, NULL);
    generation_time += (g2.tv_sec-g1.tv_sec)*1000000 + (g2.tv_usec-g1.tv_usec);

    for (int i = 0; i < G; i++)
	{
    	gettimeofday(&g1, NULL);
    	gettimeofday(&t1, NULL);    	

		// make sure that all processes are executing the same generation at any given time
    	MPI_Barrier(MPI_COMM_WORLD);

		gettimeofday(&t2, NULL); // comm time

		Simulate(rank, n, p, cell);

		gettimeofday(&g2, NULL); // gen time

		communication_time += (t2.tv_sec-t1.tv_sec)*1000000 + (t2.tv_usec-t1.tv_usec);
		generation_time += (g2.tv_sec-g1.tv_sec)*1000000 + (g2.tv_usec-g1.tv_usec);

		if (i == 0 || i == 25 || i == 50 || i == 75 || i == 99)
		{                                                                  
    		   DisplayGoL(rank, n, p, cell); // display contents of matrix every x generations
		}
    }

    avg_generation_time = generation_time / G;

    int *comm_time;
    if (rank == 0)
    {
    	comm_time = (int *)malloc(p * sizeof(int)); // allocate mem
    }
    // get comm times
    MPI_Gather(&communication_time, 1, MPI_INT, comm_time, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *gen_time;
    if (rank == 0)
    {
    	gen_time = (int *)malloc(p* sizeof(int)); // allocate mem
    }
    // get gen times
    MPI_Gather(&generation_time, 1, MPI_INT, gen_time, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0)
	{
		// for runtime, commtime, and avereages:

		int max_gen = generation_time;
    	for (int i = 0; i < p - 1; i++) 
		{
    		if (gen_time[i] > max_gen) 
			{
    			max_gen = gen_time[i];
			}
    	}

    	int max_comm = communication_time;
    	for (int i = 0; i < p - 1; i++) 
		{
    		if (comm_time[i] > max_comm) 
			{
    			max_comm = comm_time[i];
			}
    	}  

    	printf("Matrix Size: %d X %d\n", n, n);
    	printf("Procs: %d\n", p);
    	printf("Generations: %d\n", G);
    	printf("Total runtime (microseconds): %d\n", max_gen);
    	printf("Average time per generation (microseconds): %d\n", max_gen / G);
    	printf("Total Comm Time (microseconds): %d\n", max_comm);
    	printf("Total Compute Time (microseconds): %d\n", max_gen - max_comm); // runtime - commtime
    }
    MPI_Finalize();
    return 0;
}
