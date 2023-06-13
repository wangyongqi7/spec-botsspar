#ifndef SPARSELU_H
#define SPARSELU_H
#include <mpi.h>
#define EPSILON 1.0E-6
#define MIN(X, Y) (X > Y ? Y : X)
#include <stdarg.h>
#include "time.h"
#include <unistd.h>

extern int myid, numprocs;
// #define LOG(...) printf("[TRACE] "),printf("client-%d: ",myid) ,printf(__VA_ARGS__), fflush(stdout)
#define LOG( ... ) 
int checkmat(float *M, float *N);
void genmat(float *M[]);
void print_structure(char *name, float *M[]);
float *allocate_clean_block();
void lu0(float *diag);
void bdiv(float *diag, float *row);
void bmod(float *row, float *col, float *inner);
void fwd(float *diag, float *col);

void scatter_row_col(float **BENCH,float *row[], float *col[],int kk);
void free_row_col(float *row[], float *col[], int kk);
void scatter_data(float** BENCH);
void gather_data(float** BENCH);
int get_owner(int index);

#endif
