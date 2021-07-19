#define lowerb(id, p, n)  ( id * (n/p) + (id < (n%p) ? id : n%p) )
#define numElem(id, p, n) ( (n/p) + (id < (n%p)) )
#define upperb(id, p, n)  ( lowerb(id, p, n) + numElem(id, p, n) - 1 )

#define min(a, b) ( (a < b) ? a : b )
#define max(a, b) ( (a > b) ? a : b )
#include "omp.h"
// Function to copy one matrix into another
void copy_mat (double *u, double *v, unsigned sizex, unsigned sizey) {
    int numprocs = omp_get_num_threads();
	#pragma omp parallel
    {
	      int myid = omp_get_thread_num();
	      int i_start = lowerb(myid, numprocs, sizex);
	      int i_end = upperb(myid, numprocs, sizex);
	      for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++) {
			for (int j=1; j<=sizey-2; j++)
		    	v[i*sizey+j] = u[i*sizey+j];
	      }
    }
}

// 1D-blocked Jacobi solver: one iteration step
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey) {
    double diff, sum=0.0;
    int nblocksi=8;  
    int numprocs = omp_get_num_threads();
	#pragma parallel
	#pragma omp single
      #pragma omp taskloop private(diff) reduction(+:sum)
      for (int i=1; i<=sizex; i++) {
        for (int j=1; j<=sizey-2; j++) {
	      utmp[i*sizey+j] = 0.25 * ( u[ i*sizey     + (j-1) ] +  // left
	                                 u[ i*sizey     + (j+1) ] +  // right
                                     u[ (i-1)*sizey + j     ] +  // top
                                     u[ (i+1)*sizey + j     ] ) ;// bottom
	      diff = utmp[i*sizey+j] - u[i*sizey + j];
	      sum += diff * diff;
	    }
      }
    return sum;
}

// 2D-blocked Gauss-Seidel solver: one iteration step
double relax_gauss (double *u, unsigned sizex, unsigned sizey) {
    double unew, diff, sum=0.0;
    int numprocs=omp_get_max_threads();

    #pragma omp parallel for ordered(2) private(unew,diff) reduction(+:sum)	
    for (int r = 0; r < numprocs; ++r) { 
		for (int c = 0; c < numprocs; ++c) { 
			int r_start = lowerb(r, numprocs, sizex);
			int r_end = upperb(r, numprocs, sizex); 
			int c_start = lowerb(c, numprocs, sizey);
			int c_end = upperb(c, numprocs, sizey);   
			#pragma omp ordered depend(sink: r-1, c)           
			for (int i=max(1, r_start); i<= min(sizex-2, r_end); i++) { 
				for (int j=max(1, c_start); j<= min(sizey-2,c_end); j++) {
				unew= 0.25 * ( u[ i*sizey	+ (j-1) ]+  // left
					u[ i*sizey	+ (j+1) ]+  // right
					u[ (i-1)*sizey	+ j     ]+  // top
					u[ (i+1)*sizey	+ j     ]); // bottom
				diff = unew - u[i*sizey+ j];
				sum += diff * diff; 
				u[i*sizey+j]=unew;
				}
			}
			#pragma omp ordered depend(source)
		}
	}

    return sum;
}
