/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify                      */
/*  it under the terms of the GNU General Public License as published by                      */
/*  the Free Software Foundation; either version 2 of the License, or                         */
/*  (at your option) any later version.                                                       */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful,                           */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of                            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                             */
/*  GNU General Public License for more details.                                              */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License                         */
/*  along with this program; if not, write to the Free Software                               */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA            */
/**********************************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <libgen.h>
#include "bots.h"
#include "sparselu.h"

/***********************************************************************
 * checkmat:
 **********************************************************************/
int checkmat(float *M, float *N)
{
   int i, j;
   float r_err;

   for (i = 0; i < bots_arg_size_1; i++)
   {
      for (j = 0; j < bots_arg_size_1; j++)
      {
         r_err = M[i * bots_arg_size_1 + j] - N[i * bots_arg_size_1 + j];
         if (r_err < 0.0)
            r_err = -r_err;
         r_err = r_err / M[i * bots_arg_size_1 + j];
         if (r_err > EPSILON)
         {
            bots_message("Checking failure: A[%d][%d]=%f  B[%d][%d]=%f; Relative Error=%f\n",
                         i, j, M[i * bots_arg_size_1 + j], i, j, N[i * bots_arg_size_1 + j], r_err);
            return FALSE;
         }
      }
   }
   return TRUE;
}
/***********************************************************************
 * genmat:
 **********************************************************************/
void genmat(float *M[])
{
   int null_entry, init_val, i, j, ii, jj;
   float *p;
   float *prow;
   float rowsum;

   init_val = 1325;

   /* generating the structure */
   for (ii = 0; ii < bots_arg_size; ii++)
   {
      for (jj = 0; jj < bots_arg_size; jj++)
      {
         /* computing null entries */
         null_entry = FALSE;
         if ((ii < jj) && (ii % 3 != 0))
            null_entry = TRUE;
         if ((ii > jj) && (jj % 3 != 0))
            null_entry = TRUE;
         if (ii % 2 == 1)
            null_entry = TRUE;
         if (jj % 2 == 1)
            null_entry = TRUE;
         if (ii == jj)
            null_entry = FALSE;
         if (ii == jj - 1)
            null_entry = FALSE;
         if (ii - 1 == jj)
            null_entry = FALSE;
         /* allocating matrix */
         if (null_entry == FALSE)
         {
            M[ii * bots_arg_size + jj] = (float *)malloc(bots_arg_size_1 * bots_arg_size_1 * sizeof(float));
            if ((M[ii * bots_arg_size + jj] == NULL))
            {
               bots_message("Error: Out of memory\n");
               exit(101);
            }
            /* initializing matrix */
            /* Modify diagonal element of each row in order */
            /* to ensure matrix is diagonally dominant and  */
            /* well conditioned. */
            prow = p = M[ii * bots_arg_size + jj];
            for (i = 0; i < bots_arg_size_1; i++)
            {
               rowsum = 0.0;
               for (j = 0; j < bots_arg_size_1; j++)
               {
                  init_val = (3125 * init_val) % 65536;
                  (*p) = (float)((init_val - 32768.0) / 16384.0);
                  rowsum += abs(*p);
                  p++;
               }
               if (ii == jj)
                  *(prow + i) = rowsum * (float)bots_arg_size + abs(*(prow + i));
               prow += bots_arg_size_1;
            }
         }
         else
         {
            M[ii * bots_arg_size + jj] = NULL;
         }
      }
   }
}
/***********************************************************************
 * print_structure:
 **********************************************************************/
void print_structure(char *name, float *M[])
{
   int ii, jj;
   bots_message("Structure for matrix %s @ 0x%p\n", name, M);
   for (ii = 0; ii < bots_arg_size; ii++)
   {
      for (jj = 0; jj < bots_arg_size; jj++)
      {
         if (M[ii * bots_arg_size + jj] != NULL)
         {
            bots_message("x");
         }
         else
            bots_message(" ");
      }
      bots_message("\n");
   }
   bots_message("\n");
}
/***********************************************************************
 * allocate_clean_block:
 **********************************************************************/
float *allocate_clean_block()
{
   int i, j;
   float *p, *q;

   p = (float *)malloc(bots_arg_size_1 * bots_arg_size_1 * sizeof(float));
   q = p;
   if (p != NULL)
   {
      for (i = 0; i < bots_arg_size_1; i++)
         for (j = 0; j < bots_arg_size_1; j++)
         {
            (*p) = 0.0;
            p++;
         }
   }
   else
   {
      bots_message("Error: Out of memory\n");
      exit(101);
   }
   return (q);
}

/***********************************************************************
 * lu0:
 **********************************************************************/
void lu0(float *diag)
{
   int i, j, k;

   for (k = 0; k < bots_arg_size_1; k++)
      for (i = k + 1; i < bots_arg_size_1; i++)
      {
         diag[i * bots_arg_size_1 + k] = diag[i * bots_arg_size_1 + k] / diag[k * bots_arg_size_1 + k];
         for (j = k + 1; j < bots_arg_size_1; j++)
            diag[i * bots_arg_size_1 + j] = diag[i * bots_arg_size_1 + j] - diag[i * bots_arg_size_1 + k] * diag[k * bots_arg_size_1 + j];
      }
}

/***********************************************************************
 * bdiv:
 **********************************************************************/
void bdiv(float *diag, float *row)
{
   int i, j, k;
   for (i = 0; i < bots_arg_size_1; i++)
      for (k = 0; k < bots_arg_size_1; k++)
      {
         row[i * bots_arg_size_1 + k] = row[i * bots_arg_size_1 + k] / diag[k * bots_arg_size_1 + k];
         for (j = k + 1; j < bots_arg_size_1; j++)
            row[i * bots_arg_size_1 + j] = row[i * bots_arg_size_1 + j] - row[i * bots_arg_size_1 + k] * diag[k * bots_arg_size_1 + j];
      }
}
/***********************************************************************
 * bmod:
 **********************************************************************/
void bmod(float *row, float *col, float *inner)
{
   int i, j, k;
   for (i = 0; i < bots_arg_size_1; i++)
      for (j = 0; j < bots_arg_size_1; j++)
         for (k = 0; k < bots_arg_size_1; k++)
            inner[i * bots_arg_size_1 + j] = inner[i * bots_arg_size_1 + j] - row[i * bots_arg_size_1 + k] * col[k * bots_arg_size_1 + j];
}
/***********************************************************************
 * fwd:
 **********************************************************************/
void fwd(float *diag, float *col)
{
   int i, j, k;
   for (j = 0; j < bots_arg_size_1; j++)
      for (k = 0; k < bots_arg_size_1; k++)
         for (i = k + 1; i < bots_arg_size_1; i++)
            col[i * bots_arg_size_1 + j] = col[i * bots_arg_size_1 + j] - diag[i * bots_arg_size_1 + k] * col[k * bots_arg_size_1 + j];
}

void sparselu_init(float ***pBENCH, char *pass)
{
   *pBENCH = (float **)malloc(bots_arg_size * bots_arg_size * sizeof(float *));
   genmat(*pBENCH);
   /* spec  print_structure(pass, *pBENCH);  */
}

void print_matrix(float** BENCH){
   for(int ii=0;ii<bots_arg_size;ii++){
      for(int jj=0;jj<bots_arg_size;jj++){
         printf("\t%d",BENCH[ii*bots_arg_size+jj]==NULL?0:1);
      }
      printf("\n");
   }
}

void print_matrix_exist(int BENCH[]){
   for(int ii=0;ii<bots_arg_size;ii++){
      for(int jj=0;jj<bots_arg_size;jj++){
         printf("\t%d",BENCH[ii*bots_arg_size+jj]);
      }
      printf("\n");
   }
}

void print_array(float* array[]){
   for(int ii=0;ii<bots_arg_size;ii++){
      printf("\t%d",array[ii]==NULL?0:1);
   }
   printf("\n");
}

int get_owner(int index)
{
   int i = index / bots_arg_size;
   int j = index % bots_arg_size;
   int k = MIN(i, j);
   return k % numprocs;
}

int matrix_size ;
int block_size ;

void sparselu_par_call(float **BENCH)
{
   matrix_size = bots_arg_size * bots_arg_size;
   block_size = bots_arg_size_1 * bots_arg_size_1;
   int ii, jj, kk;
   // printf("client %d 进入并行计算区域\n",myid);
   scatter_data(BENCH);
   
   // printf("INFO:%d 数据分发完成\n",myid);
   fflush(stdout);
   // print_matrix(BENCH);
   bots_message("Computing SparseLU Factorization (%dx%d matrix with %dx%d blocks) ",
                bots_arg_size, bots_arg_size, bots_arg_size_1, bots_arg_size_1);
   float diag[block_size];

   // 存储kk所在的右行和下列数据
   float *row[bots_arg_size];
   float *col[bots_arg_size];
   
   for(ii=0;ii<bots_arg_size;ii++){
      row[ii]=NULL;
      col[ii]=NULL;
   }
#pragma omp parallel
#pragma omp single nowait
   for (kk = 0; kk < bots_arg_size; kk++)
   {
      MPI_Request requests[2 * bots_arg_size];
      LOG("第%d次循环\n",kk);
      if(myid==get_owner(kk * bots_arg_size + kk)){
         lu0(BENCH[kk * bots_arg_size + kk]);
         LOG("LU分解结束\n");
         memcpy(diag, BENCH[kk * bots_arg_size + kk], block_size * sizeof(float));
      }
      
      
      MPI_Bcast(diag, block_size, MPI_FLOAT, get_owner(kk * bots_arg_size + kk), MPI_COMM_WORLD);
      LOG("kk矩阵广播结束\n");
      // diag存储lu分解后的矩阵
      scatter_row_col(BENCH, row, col, kk);
      LOG("分发行列结束\n");
      LOG("row array:\n");
      // print_array(row);
      LOG("col array:\n");
      // print_array(col);
      LOG("myid=%d numprocs=%d\n",myid,numprocs);
      // 并行计算fwd bdiv 并且 非阻塞广播计算完成的行列数据
      for (jj = 0; jj < bots_arg_size - kk - 1; jj++)
         if (row[jj] != NULL && jj%numprocs==myid)
#pragma omp task untied firstprivate(jj) shared(BENCH,requests)
         {
            LOG("fwd %d 开始计算\n",jj);
            fwd(diag, row[jj]);
         }

      for (ii = 0; ii < bots_arg_size - kk - 1; ii++)
         if (col[ii] != NULL && ii%numprocs==myid)
#pragma omp task untied firstprivate(ii) shared(BENCH,requests)
         {
            LOG("bdiv %d 开始计算\n",ii);
            bdiv(diag, col[ii]);
         }

#pragma omp taskwait
      LOG("行列计算结束\n");
      for(ii=0;ii<bots_arg_size - kk - 1;ii++){
         if (row[ii] != NULL){
            LOG("row[%d] 广播接收  请求 %p root=%d\n",ii,&requests[ii],ii % numprocs);
            MPI_Ibcast(row[ii], block_size, MPI_FLOAT, ii % numprocs, MPI_COMM_WORLD, &requests[ii]);

         }
         if (col[ii] != NULL){
            LOG("col[%d] 广播接收  请求 %p root= %d\n",ii,&requests[ii+bots_arg_size],ii % numprocs);
            MPI_Ibcast(col[ii], block_size, MPI_FLOAT, ii % numprocs, MPI_COMM_WORLD, &requests[ii + bots_arg_size]);
         }               
      }

      // 等待全部传输完成
      for(ii = 0; ii < bots_arg_size - kk - 1; ii++)
      {
         if (col[ii] != NULL){
            LOG("col[%d] wait bcast start %p \n",ii,&requests[ii + bots_arg_size]);
            MPI_Wait(&requests[ii + bots_arg_size], MPI_STATUS_IGNORE);
            LOG("col[%d] wait bcast end\n",ii);
         }
      }
      for (ii = 0; ii < bots_arg_size - kk - 1; ii++)
      {
         if (row[ii] != NULL){
            LOG("row[%d] wait bcast start %p \n",ii,&requests[ii]);
            MPI_Wait(&requests[ii], MPI_STATUS_IGNORE);
            LOG("row[%d] wait bcast end\n",ii);
         }            
      }

      LOG("行列广播结束\n");

      // 已知行列数据 并行计算下矩阵各block的bmod
      int begin = kk + 1;
      while (get_owner(begin * begin) != myid && begin < bots_arg_size)
         begin++;
      LOG("开始计算bmod\n");
      // begin为当前proc第一次要计算的层
      for (ii = begin; ii < bots_arg_size; ii += numprocs)
      {
         // 计算当前层的行+角
         for (jj = 0; jj < bots_arg_size - ii; jj++)
         {
            if (row[jj] != NULL && col[ii - kk - 1] != NULL)
#pragma omp task untied firstprivate(kk, jj, ii) shared(BENCH)
            {
               int inner_index = ii * bots_arg_size + ii + jj;
               if (BENCH[inner_index] == NULL)
                  BENCH[inner_index] = allocate_clean_block();
               bmod(col[ii - kk - 1], row[jj], BENCH[inner_index]);
            }
         }
         // 计算当前层的列
         for (jj = 1; jj < bots_arg_size - ii; jj++)
         {
            if (row[ii - kk - 1] != NULL && col[jj] != NULL)
#pragma omp task untied firstprivate(kk, jj, ii) shared(BENCH)
            {
               int inner_index = (ii + jj) * bots_arg_size + ii;
               if (BENCH[inner_index] == NULL)
                  BENCH[inner_index] = allocate_clean_block();
               bmod(col[jj], row[ii - kk - 1], BENCH[inner_index]);
            }
         }
      }
      LOG("bmod 计算结束\n");
#pragma omp taskwait
      LOG("释放行列内存\n");
      // 释放row/col内存
      free_row_col(row, col, kk);
      LOG("成功释放行列内存\n");
   }
   // 计算完成 聚合数据
   LOG("开始聚合数据\n");
   gather_data(BENCH);
   LOG("成功聚合数据\n");
   printf("并行计算结束");
   bots_message(" completed!\n");
   
}

void scatter_row_col(float **BENCH, float *row[], float *col[], int kk)
{

   // 存储kk所在左行和下列是否为NULL 0为NULL 1为存在block
   int row_exist[bots_arg_size];
   int col_exist[bots_arg_size];
   int ii, jj;
   if (get_owner(kk * bots_arg_size + kk) == myid)
   {
      for (ii = 0; ii + kk + 1 < bots_arg_size; ii++)
      {
         row_exist[ii] = BENCH[kk * bots_arg_size + kk + 1 + ii] == NULL ? 0 : 1;
         col_exist[ii] = BENCH[(kk + 1 + ii) * bots_arg_size + kk] == NULL ? 0 : 1;
      }
   }
   int index;
   // 传输lu后矩阵和所在行列矩阵信息

   MPI_Bcast(row_exist, bots_arg_size - 1 - kk, MPI_INT, get_owner(kk * bots_arg_size + kk), MPI_COMM_WORLD);
   MPI_Bcast(col_exist, bots_arg_size - 1 - kk, MPI_INT, get_owner(kk * bots_arg_size + kk), MPI_COMM_WORLD);
   MPI_Request requests[bots_arg_size * 2];

   // 接收行列块
   if (myid != get_owner(kk * bots_arg_size + kk))
   {
      for (ii = 0; ii + kk + 1 < bots_arg_size; ii++)
      {
         if (row_exist[ii] != 0)
         {
            row[ii] = (float *)malloc(block_size * sizeof(float));
            if (ii % numprocs == myid)
            {
               MPI_Irecv(row[ii], block_size, MPI_FLOAT, get_owner(kk * bots_arg_size + kk), ii, MPI_COMM_WORLD, &requests[ii]);
               LOG("Irecv ii=%d \n",ii);
            }
         }

         if (col_exist[ii] != 0)
         {
            col[ii] = (float *)malloc(block_size * sizeof(float));
            if (ii % numprocs == myid)
            {
               MPI_Irecv(col[ii], block_size, MPI_FLOAT, get_owner(kk * bots_arg_size + kk), bots_arg_size + ii, MPI_COMM_WORLD, &requests[bots_arg_size + ii]);
            }
         }
      }
      for (ii = 0; ii + kk + 1 < bots_arg_size; ii++)
      {
         if (row_exist[ii] != 0 && ii % numprocs == myid)
         {
            MPI_Wait(&requests[ii], MPI_STATUS_IGNORE);
         }

         if (col_exist[ii] != 0 && ii % numprocs == myid)
         {
            MPI_Wait(&requests[ii + bots_arg_size], MPI_STATUS_IGNORE);
         }
      }
   }
   // 分发行列块数据
   else
   {
      for (ii = 0; ii + kk + 1 < bots_arg_size; ii++)
      {
         if (row_exist[ii] != 0)
         {
            index = kk * bots_arg_size + kk + 1 + ii;
            row[ii] = BENCH[index];
            if (ii % numprocs != myid)
               MPI_Isend(row[ii], block_size, MPI_FLOAT, ii % numprocs, ii, MPI_COMM_WORLD, &requests[ii]);
         }
         if (col_exist[ii] != 0)
         {
            index = (kk + 1 + ii) * bots_arg_size + kk;
            col[ii] = BENCH[index];
            if (ii % numprocs != myid)
               MPI_Isend(col[ii], block_size, MPI_FLOAT, ii % numprocs, ii+bots_arg_size, MPI_COMM_WORLD, &requests[bots_arg_size + ii]);
         }
      }

      for (ii = 0; ii + kk + 1 < bots_arg_size; ii++)
      {
         if (row_exist[ii] != 0 && ii % numprocs != myid)
         {
            MPI_Wait(&requests[ii], MPI_STATUS_IGNORE);
         }

         if (col_exist[ii] != 0 && ii % numprocs != myid)
         {
            MPI_Wait(&requests[ii + bots_arg_size], MPI_STATUS_IGNORE);
         }
      }
   }
}

void free_row_col(float *row[], float *col[], int kk)
{
   int ii;
   if (get_owner(kk * bots_arg_size + kk) != myid)
   {
      // 释放临时存储的行列块内存
      for (ii = 0; ii < bots_arg_size; ii++)
      {
         if (row[ii] != NULL)
         {
            free(row[ii]);
            row[ii] = NULL;
         }
         if (col[ii] != NULL)
         {
            free(col[ii]);
            col[ii] = NULL;
         }
      }
   }
   else
   {
      // 行列数据直接指向自己的内存，置为NULL
      for (ii = 0; ii < bots_arg_size; ii++)
      {
         row[ii] = NULL;
         col[ii] = NULL;
      }
   }
}

void scatter_data(float **BENCH)
{
   // 0 表示 对应位置矩阵为NULL 否则不为NULL
   int ii, jj;
   int matrix_exist[matrix_size];
   MPI_Request requests[matrix_size];

   if (myid == 0)
   {
      for (ii = 0; ii < matrix_size; ii++)
      {
         matrix_exist[ii] = BENCH[ii] == NULL ? 0 : 1;
      }
   }
   MPI_Bcast(matrix_exist, matrix_size, MPI_INT, 0, MPI_COMM_WORLD);

   // 分发matrix_exist状态 和 BENCH数组

   if (myid != 0)
   {
      // 为分配到当前机器上数据申请内存并接收
      for (ii = 0; ii < matrix_size; ii++)
      {
         if (matrix_exist[ii] && get_owner(ii) == myid)
         {
            if (BENCH[ii] == NULL)
               BENCH[ii] = (float *)malloc(block_size * sizeof(float));
            MPI_Irecv(BENCH[ii], block_size, MPI_FLOAT, 0, ii, MPI_COMM_WORLD, &requests[ii]);
         }
      }
      // 等待数据分配完成
      for (ii = 0; ii < matrix_size; ii++)
      {
         if (matrix_exist[ii] && get_owner(ii) == myid)
            MPI_Wait(&requests[ii], MPI_STATUS_IGNORE);
      }
      
   }
   else
   {
      MPI_Request requests[matrix_size];
      for (ii = 0; ii < matrix_size; ii++)
      {
         if (BENCH[ii] != NULL && get_owner(ii) != 0)
         {
            MPI_Isend(BENCH[ii], block_size, MPI_FLOAT, get_owner(ii), ii, MPI_COMM_WORLD, &requests[ii]);
         }
      }

      for (ii = 0; ii < matrix_size; ii++)
      {
         if (BENCH[ii] != NULL && get_owner(ii) != 0)
         {
            MPI_Wait(&requests[ii], MPI_STATUS_IGNORE);
         }
      }
   }
}

void gather_data(float **BENCH)
{
   int ii, jj;
   // 聚合数据

   int matrix_exist_tmp[matrix_size];
   int matrix_exist[matrix_size];
   if(myid==0){
      for(ii=0;ii<matrix_size;ii++)
         matrix_exist[ii] = 0;
   }
   MPI_Request requests[matrix_size];

   for (ii = 0; ii < matrix_size; ii++)
   {
      if (get_owner(ii) == myid)
         matrix_exist_tmp[ii] = BENCH[ii] == NULL ? 0 : 1;
      else{
         matrix_exist_tmp[ii]=0;
      }
   }
   
   // LOG("matrix_exist: \n");
   //    print_matrix_exist(matrix_exist_tmp);
   
   MPI_Reduce(matrix_exist_tmp, matrix_exist, matrix_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

   // if(myid==0){
   //    LOG("matrix_exist: ");
   //    print_matrix_exist(matrix_exist);
   // }
   
   
   if (myid != 0)
   {
      // 为分配到当前机器上数据申请内存并接收
      for (ii = 0; ii < matrix_size; ii++)
      {
         if (get_owner(ii) == myid && BENCH[ii] != NULL)
            MPI_Isend(BENCH[ii], block_size, MPI_FLOAT, 0, ii, MPI_COMM_WORLD, &requests[ii]);
      }
      // 等待数据分配完成
      for (ii = 0; ii < matrix_size; ii++)
      {
         if (get_owner(ii) == myid && BENCH[ii] != NULL)
            MPI_Wait(&requests[ii], MPI_STATUS_IGNORE);
      }
   }
   else
   {
      for (ii = 0; ii < matrix_size; ii++)
      {
         if (matrix_exist[ii] != 0 && get_owner(ii) != 0)
         {
            if (BENCH[ii] == NULL)
               BENCH[ii] = (float *)malloc(block_size * sizeof(float));
            MPI_Irecv(BENCH[ii], block_size, MPI_FLOAT, get_owner(ii), ii, MPI_COMM_WORLD, &requests[ii]);
         }
      }

      for (ii = 0; ii < matrix_size; ii++)
      {
         if (matrix_exist[ii] != 0 && get_owner(ii) != 0)
         {
            MPI_Wait(&requests[ii], MPI_STATUS_IGNORE);
         }
      }
   }
}

void sparselu_seq_call(float **BENCH)
{
   int ii, jj, kk;

   for (kk = 0; kk < bots_arg_size; kk++)
   {
      lu0(BENCH[kk * bots_arg_size + kk]);
      for (jj = kk + 1; jj < bots_arg_size; jj++)
         if (BENCH[kk * bots_arg_size + jj] != NULL)
         {
            fwd(BENCH[kk * bots_arg_size + kk], BENCH[kk * bots_arg_size + jj]);
         }
      for (ii = kk + 1; ii < bots_arg_size; ii++)
         if (BENCH[ii * bots_arg_size + kk] != NULL)
         {
            bdiv(BENCH[kk * bots_arg_size + kk], BENCH[ii * bots_arg_size + kk]);
         }
      for (ii = kk + 1; ii < bots_arg_size; ii++)
         if (BENCH[ii * bots_arg_size + kk] != NULL)
            for (jj = kk + 1; jj < bots_arg_size; jj++)
               if (BENCH[kk * bots_arg_size + jj] != NULL)
               {
                  if (BENCH[ii * bots_arg_size + jj] == NULL)
                     BENCH[ii * bots_arg_size + jj] = allocate_clean_block();
                  bmod(BENCH[ii * bots_arg_size + kk], BENCH[kk * bots_arg_size + jj], BENCH[ii * bots_arg_size + jj]);
               }
   }
}

void sparselu_fini(float **BENCH, char *pass)
{
   /* spec  print_structure(pass, BENCH); */
   return;
}

/*
 * changes for SPEC, original source
 *
int sparselu_check(float **SEQ, float **BENCH)
{
   int ii,jj,ok=1;

   for (ii=0; ((ii<bots_arg_size) && ok); ii++)
   {
      for (jj=0; ((jj<bots_arg_size) && ok); jj++)
      {
         if ((SEQ[ii*bots_arg_size+jj] == NULL) && (BENCH[ii*bots_arg_size+jj] != NULL)) ok = FALSE;
         if ((SEQ[ii*bots_arg_size+jj] != NULL) && (BENCH[ii*bots_arg_size+jj] == NULL)) ok = FALSE;
         if ((SEQ[ii*bots_arg_size+jj] != NULL) && (BENCH[ii*bots_arg_size+jj] != NULL))
            ok = checkmat(SEQ[ii*bots_arg_size+jj], BENCH[ii*bots_arg_size+jj]);
      }
   }
   if (ok) return BOTS_RESULT_SUCCESSFUL;
   else return BOTS_RESULT_UNSUCCESSFUL;
}
*/

/*
 * SPEC modified check, print out values
 *
 */
int checkmat1(float *N)
{
   int i, j;

   for (i = 0; i < bots_arg_size_1; i += 20)
   {
      for (j = 0; j < bots_arg_size_1; j += 20)
      {
         bots_message("Output Matrix: A[%d][%d]=%8.12f \n",
                      i, j, N[i * bots_arg_size_1 + j]);
      }
   }
   return TRUE;
}
int sparselu_check(float **SEQ, float **BENCH)
{
   int i, j, ok;

   bots_message("Output size: %d\n", bots_arg_size);
   for (i = 0; i < bots_arg_size; i += 50)
   {
      for (j = 0; j < bots_arg_size; j += 40)
      {
         ok = checkmat1(BENCH[i * bots_arg_size + j]);
      }
   }
   return BOTS_RESULT_SUCCESSFUL;
}
