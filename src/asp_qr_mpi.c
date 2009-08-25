/*$Id$*/

/* Copyright (c) 2005 Alexey Ozeritsky
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by Alexey Ozeritsky.
 * 4. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#ifndef _DEBUG
#define _DEBUG
#endif

#include "asp_lib.h"


/*!умножение матрицы на вектор
  \param w-ответ
  \param v-начальное условие
*/
void av(MPI_Comm comm, int n,
		int myid, int nprocs,
		const double *a, const double *v, double *w)
{
	int f_raw    = (n) * myid / nprocs;
	int l_raw    = (n) * (myid + 1) / nprocs - 1;
	/*int raws     = l_raw - f_raw + 1;*/
	int max_raws = (n) / nprocs + (n) % nprocs;

	int l, k, i, j;
	int f_raw_k;
	int l_raw_k;
	int raws_k;
	int dest, source, tag = 0;

	double * b = malloc(max_raws * sizeof(double));
	memcpy(b, v, max_raws * sizeof(double));

	MPI_Status status;

	memset(w, 0, max_raws * sizeof(double));
	MPI_Barrier(comm);

	for (l = 0; l < nprocs; l++) {
		k        = (myid + l) % nprocs;
		f_raw_k  = (n) * k;
		f_raw_k /= nprocs;
		l_raw_k  = (n) * (k + 1);
		l_raw_k  = l_raw_k / nprocs - 1;
		raws_k   = l_raw_k - f_raw_k + 1;

		for (i = f_raw; i <= l_raw && i < n; i++) {
			for (j = f_raw_k; j <= l_raw_k && j < n; j++) {
				w[i - f_raw] += a[(i - f_raw) * n + j] * b[j - f_raw_k];
			}
		}

		source = (myid + 1) % nprocs;
		if (myid == 0) dest = nprocs - 1;
		else dest = myid - 1;

		MPI_Sendrecv_replace(b, max_raws, MPI_DOUBLE,
							 dest, tag, source, tag, comm, &status);
	}
	free(b);
}

void avT(MPI_Comm comm, int n,
		int myid, int nprocs,
		const double *a, const double *v, double *w)
{
	int f_raw    = (n) * myid / nprocs;
	int l_raw    = (n) * (myid + 1) / nprocs - 1;
	int raws     = l_raw - f_raw + 1;
	int max_raws = (n) / nprocs + (n) % nprocs;

	int i, j;
	//int l, k;
	//int f_raw_k;
	//int l_raw_k;
	//int raws_k;
	//int dest, source, tag = 0;

	double * b = malloc(n * sizeof(double));
	double * c = 0;

	//	int * scounts = malloc(nprocs * sizeof(int));
	//int * disps   = malloc(nprocs * sizeof(int));

	//double * b = malloc(max_raws * sizeof(double));
	//memcpy(b, v, max_raws * sizeof(double));

	//MPI_Status status;

	//if (myid == 0)
		c = malloc(n * sizeof(double));
	memset(w, 0, max_raws * sizeof(double));
	MPI_Barrier(comm);


	for (i = 0; i < n; i++) {
		b[i] = 0.0;
		for (j = 0; j < raws; j++) {
			b[i] += a[j * n + i] * v[j];
		}
	}

	//MPI_Reduce(b, c, n, MPI_DOUBLE, MPI_SUM, 0, comm);
	MPI_Allreduce(b, c, n, MPI_DOUBLE, MPI_SUM,  comm);
	memcpy(w, &c[f_raw], raws * sizeof(double));

	//MPI_Barrier(comm);
	//for (i = 0; i < nprocs; i++) {
	//	int f = n * i / nprocs;
	//	int l = n * (i + 1) / nprocs - 1;
	//	scounts[i] = l - f + 1;
	//	disps[i]   = f;
	//}

	MPI_Barrier(comm);
	//MPI_Scatterv(c, scounts, disps, MPI_DOUBLE, w, max_raws, MPI_DOUBLE,
	//			 0, comm);

	//if (myid == 0)
		free(c);
		//free(scounts); free(disps);
	free(b);
}

/*!вычисление X подпространства с помощью MPI
  методом Арнольди.
  \param comm - MPI коммуникатор
  \param A    - строки матрицы, с которыми работает данный процесс
  \param l    - собственные значения, соответсвующие пространству X
  \param Ex   - подпространство X, память выделяется malloc'ом
  \param n    - размерность пространства
  \param nx   - оценка с верху размерность пространства nx,
  сюда же кладется реальная размерность nx,
*/
void ev_x_general_parpack_mpi(MPI_Comm comm,
							  double *A,
							  double **l,
							  double **Ex, int n,
							  int *nx)
{
	int myid;
	int nprocs;
	int f_raw, l_raw; /*!<номера первой и последней строки реальной матрицы*/
	int raws; /*!<число строк с которыми работает текущий процесс*/
	int max_raws; /*!<максимальное число строк с которыми работают процессы*/
	int ldv;

	double *dr; /*!<вещественная часть собственных значений*/
	double *di; /*!<мнимая часть собственных значений*/

	double *resid, *workd, *v, *workl, *workev;

	char   bmat[] ="I";
	/*          BMAT = 'I' -> standard eigenvalue problem A*x = lambda*x*/
    /*          BMAT = 'G' -> generalized eigenvalue problem A*x = lambda*B*x*/

	char which[] = "LM";
	/*          'LM' -> want the NEV eigenvalues of largest magnitude.*/
	/*          'SM' -> want the NEV eigenvalues of smallest magnitude.*/
	/*          'LR' -> want the NEV eigenvalues of largest real part.*/
	/*          'SR' -> want the NEV eigenvalues of smallest real part.*/
	/*          'LI' -> want the NEV eigenvalues of largest imaginary part.*/
	/*          'SI' -> want the NEV eigenvalues of smallest imaginary part.*/

	int iparam[11], ipntr[14];
    int ishfts = 1;
	int maxitr = _ARPACK_MAX_ITER;
	int mode   = 1;
	int nev    = *nx;
    int ncv    = ((2 * nev + 1) < n)?
			(2 * nev + 1):n; /*  Number of columns of the matrix V.
								 NCV must satisfy the two
						               inequalities 2 <= NCV-NEV and NCV <= N.*/
	int lworkl     = 3 * ncv * ncv + 6 * ncv;
	int status;


	int * rcounts = 0;/* для объединения данных в нулевом*/
	int * disps   = 0;  /* --''--*/
	int * c = malloc(n * sizeof(int));
	int ido  = 0;
	int info = 0;
	int i, j;
	double tol = 0; /*точность 0 - машинная*/
	int f_comm;

	ERR((1 <= *nx && *nx <= n - 2), "nx must be 1 <= nx <= n - 2");

	MPI_Comm_rank(comm, &myid);
	MPI_Comm_size(comm, &nprocs);

	f_raw = n * myid / nprocs;
	l_raw = n * (myid + 1) / nprocs - 1;
	raws     = l_raw - f_raw + 1;
	max_raws = n / nprocs + n % nprocs;
	ldv = max_raws;

	/*в фортране нумерация массива с 1*/
	iparam[1-1] = ishfts;
	iparam[3-1] = maxitr;
	iparam[7-1] = mode;

	dr     = malloc((nev + 1) * sizeof(double));
	di     = malloc((nev + 1) * sizeof(double));
	resid  = malloc(n * sizeof(double));   /*(n)*/
	workd  = malloc(3 * n * sizeof(double));
	workl  = malloc(lworkl * sizeof(double));
	workev = malloc(3 * ncv*sizeof(double));
	v      = malloc(ldv * ncv * sizeof(double)); /*(ldv,maxncv) (n, ncv)*/

	f_comm = MPI_Comm_c2f(comm);

	do {
		pdnaupd_ (&f_comm, &ido, bmat, &raws, which,
				   &nev, &tol, resid, &ncv, v, &ldv,
				   iparam, ipntr, workd, workl, &lworkl,
				   &info );

		ERR((check_dnaupd_status(info) >= 0), "error in arpack's dnaupd_ procedure");
		if (ido == -1 || ido == 1) {
			int addr2 = ipntr[2-1]-1;
			int addr1 = ipntr[1-1]-1;
			double *w1 = &workd[addr2]; /*ответ*/
			double *v1 = &workd[addr1]; /*начальное условие*/

#ifdef _PARPACK_TRANSPOSE_MATRIX
			avT(comm, n, myid, nprocs, A, v1, w1);
#else
			av(comm, n, myid, nprocs, A, v1, w1);
#endif
		}
	} while (ido == -1 || ido == 1);

	{
		char c[] = "A";
		int rvec   = 1;
		int *select = malloc(ncv * sizeof(int));
		double sigmar, sigmai;
		int ierr = 0;

		pdneupd_ (&f_comm, &rvec, c, select, dr, di, v, &ldv,
				   &sigmar, &sigmai, workev, bmat, &raws, which, &nev, &tol,
				   resid, &ncv, v, &ldv, iparam, ipntr, workd, workl,
				   &lworkl, &ierr );
		ERR((check_dneupd_status(ierr) >= 0), "error in arpack's dneupd_ procedure\n");
		free(select);
	}

	free(resid); free(workd); free(workl); free(workev);

	MPI_Barrier(comm);

	/*классифицируем найденные вектора*/
	if (myid == 0) {
		*nx = 0;
		for (i = 0; i < nev; i++) {
			double dist = sqrt(dr[i] * dr[i] + di[i] * di[i]);
			if (dist > 1)    {c[i] = 1; (*nx)++;} /* x */
			else /*dist< 1*/ {c[i] = 0;} /* y */
		}
	}

	MPI_Barrier(comm);
	MPI_Bcast(nx, 1, MPI_INT, 0, comm);
	MPI_Bcast(c, nev, MPI_INT, 0, comm);

	/*if (myid == 0) {
		printf("sz:\n");
		printfvector(dr, nev, 10, "%.3le ");
		printf("\n");
	}*/

	*Ex = (double*)malloc(n * (*nx) * sizeof(double));

	if (myid == 0) {
		disps   = (int *)malloc(nprocs * sizeof(int));
		rcounts = (int *)malloc(nprocs * sizeof(int));

		for (i = 0; i < nprocs; i++) {
			int f = n * i / nprocs;
			int l = n * (i + 1) / nprocs - 1;
			rcounts[i] = l - f + 1;
			disps[i]   = f;
		}
		/*на первом этапе вектора лежат по строкам*/
		/*в i'ю строку собираем информацию с процессов*/
	}

	MPI_Barrier(comm);

	if (l) *l = (double*)malloc(*nx * sizeof(double));
	for (j = 0, i = 0; i < nev; i++) {
		if (c[i] == 1) {
			MPI_Barrier(comm);
			status = MPI_Gatherv(&v[i * ldv], raws, MPI_DOUBLE,
								 &(*Ex)[j * n], rcounts, disps,
								 MPI_DOUBLE, 0, comm);
			ERR((status == MPI_SUCCESS), "MPI_Gather error");
			(*l)[j] = dr[i];
			j ++;
		}
	}

	free(v); free(c); free(dr); free(di);

	if (myid == 0) {
		matrix_transposew(*Ex, *nx, n);
	}

	MPI_Barrier(comm);
	MPI_Bcast(*Ex, n * (*nx), MPI_DOUBLE, 0, comm);

	if (myid == 0) {
		free(rcounts); free(disps);
	}
}
