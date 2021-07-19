#include <tareador.h>

#define lowerb(id, p, n)  ( id * (n/p) + (id < (n%p) ? id : n%p) )
#define numElem(id, p, n) ( (n/p) + (id < (n%p)) )
#define upperb(id, p, n)  ( lowerb(id, p, n) + numElem(id, p, n) - 1 )

#define min(a, b) ( (a < b) ? a : b )
#define max(a, b) ( (a > b) ? a : b )

// Function to copy one matrix into another
void copy_mat (double *u, double *v, unsigned sizex, unsigned sizey) {
    int nblocksi=8;
    for (int blocki=0; blocki<nblocksi; ++blocki) {
            tareador_start_task("copy_mat");
      int i_start = lowerb(blocki, nblocksi, sizex);
      int i_end = upperb(blocki, nblocksi, sizex);
      for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++) {
        for (int j=1; j<=sizey-2; j++)
            v[i*sizey+j] = u[i*sizey+j];
      }
            tareador_end_task("copy_mat");
    }
}

// 1D-blocked Jacobi solver: one iteration step
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey) {
    double diff, sum=0.0;
    int nblocksi=8;
    tareador_disable_object(&sum);
    for (int blocki=0; blocki<nblocksi; ++blocki) {
	tareador_start_task("relax-jacobi");
      int i_start = lowerb(blocki, nblocksi, sizex);
      int i_end = upperb(blocki, nblocksi, sizex);
      for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++) {
        for (int j=1; j<=sizey-2; j++) {
	      utmp[i*sizey+j] = 0.25 * ( u[ i*sizey     + (j-1) ] +  // left
	                                 u[ i*sizey     + (j+1) ] +  // right
                                     u[ (i-1)*sizey + j     ] +  // top
                                     u[ (i+1)*sizey + j     ] ) ;// bottom
	      diff = utmp[i*sizey+j] - u[i*sizey + j];
	      sum += diff * diff;
	    }
      }
	tareador_end_task("relax-jacobi");
    }
    tareador_enable_object(&sum);
	
    return sum;
}

// 2D-blocked Gauss-Seidel solver: one iteration step
double relax_gauss (double *u, unsigned sizex, unsigned sizey) {
    double unew, diff, sum=0.0;
    //tareador_disable_object(&sum);
    int nblocksi=4;
    int nblocksj=4;
    for (int blocki=0; blocki<nblocksi; ++blocki) {
      int i_start = lowerb(blocki, nblocksi, sizex);
      int i_end = upperb(blocki, nblocksi, sizex);
      for (int blockj=0; blockj<nblocksj; ++blockj) {
	tareador_start_task("relax_gauss");
        int j_start = lowerb(blockj, nblocksj, sizey);
        int j_end = upperb(blockj, nblocksj, sizey);
        for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++) {
          for (int j=max(1, j_start); j<=min(sizey-2, j_end); j++) {
	        unew = 0.25 * ( u[ i*sizey	   + (j-1) ] +  // left
                            u[ i*sizey	   + (j+1) ] +  // right
                            u[ (i-1)*sizey + j     ] +  // top
                            u[ (i+1)*sizey + j     ] ); // bottom
	        diff = unew - u[i*sizey+ j];
	        sum += diff * diff;
	        u[i*sizey+j] = unew;
          }
        }
	tareador_end_task("relax_gauss");
      }
    }
    //tareador_enable_object(&sum);
    return sum;
}
