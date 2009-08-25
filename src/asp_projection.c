/* $Id$ */

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

/**
 * Алгоритмы проектирования.
 * Проектирование вдоль подпространства, ортогональное проектирование,
 * проектирование как нахождение минимизирующего вектора (lapack'based),
 * Ортогонализация Грамма-Шмидта
 */

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "asp_macros.h"
#include "asp_misc.h"
#include "asp_gauss.h"
#include "asp_projection.h"

/**
 * по образу и прообразу n n'мерных линейнонезависимых векторов
 * находит матрицу линеаризации отображения
 * @param A - искомая матрица линеаризации
 * @param X - n'векторов прообразов (хранятся по столбцам)
 * @param Y - n'векторов образов (хранятся по столбцам)
 * @param n - размерность пространства
 */
void linearization(double *A, double *X, double *Y, int n) {
	/**
	 * надо найти такую матрицу, что AXi = Yi, где
	 * Xi, Yi - i'е стобцы матриц X и Y
	 * то есть AX = Y и A = YX^{-1}
	 */
	int i, j, k;
	double *X1 = (double*)malloc(n*n*sizeof(double)); NOMEM(X1);
	inverse_general_matrix(X1, X, n);
	//gauss(n, X, X1);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			A[i * n + j] = 0;
			for (k = 0; k < n; k++) {
				A[i * n + j] += Y[i * n + k] * X1[k * n + j];
			}
		}
	}
	free(X1);
}


/**
 * Процесс ортогонализации Грамма-Шмидта
 * основная формула:
 * \f$E_{k+1}=e_{k+1}-\sum_{j=1}^{k} \lambda_j E_j\f$,
 * \f$E_{k+1}=E_{k+1}/||E_{k+1}||\f$ - нормировка,
 * где \f$\lambda_j=(E_j,e_{k+1})\f$,
 * \f${E_i}\f$ - новый базис (ортонормированный),
 * \f${e_i}\f$ - старый базис(неортонормированный)
 * 
 * @param X - вектора, которые иртогонализуем, хранящиеся по строкам
 * @param n - размерность пространства
 * @param m - число векторов
 */
void ortogonalize_gsch(double *X, int n, int m) {
	int i = 0, j = 0, k = 0;
	double sum;
	double *la  = (double*)malloc(m * sizeof(double));
	double *tX  = X;
	double *oX  = X;
	double norm = 1.0;

	/*вычисляем вектора E_{k}*/
	/*векторов 'm' штук !!!*/
	for (k = 0; k < m; k++) {
		for (i = 0; i < k; i++) {
			la[i] = scalar(&oX[i * n], &tX[k * n],n);
		}
		/*по координатам*/
		for (j = 0; j < n; j++) {
			sum=0.0;
			for (i = 0; i < k; i++) {
				sum += la[i] * oX[i * n + j];
			}
 			oX[k * n + j] = tX[k * n + j] - sum;
		}

		norm = 1.0 / sqrt(scalar(&oX[k * n], &oX[k * n], n));
		vector_mult_scalar(&oX[k * n], &oX[k * n], norm, n);
	}
	
	free(la);
}

double volume(double *X, double *l, int n, int m) {
	//матрица Грамма
	int i, j, k, r;
	double * M = (double*)malloc(m * m * sizeof(double));

	double * X1 = (double*)malloc(n * m * sizeof(double));
	double * A = M;
	double vol = 1.0;

	memcpy(X1, X, n * m * sizeof(double));
	ortogonalize_gsch(X1, n, m);

	for (i = 0; i < m; ++i) {
		for (j = 0; j < m; ++j) {
			M[i * m + j] = scalar(&X[i * n], &X1[j * n], n);
		}
	}

/*	inverse_general_matrix(M, M, m);*/

/*gauss*/
	n = m;
    for (j = 0; j < n; j++) {
    	double a;
     /*поиск главного значения*/
        findmain(&r,j,n,A);
        swop(r,j,n,A);
        swop2(r,j,n,X);

        a = A[j * n + j];
        for (i = j + 1; i < n; i++) {
            double ba = A[i * n + j] / a;

            for (k = j; k < n; k++)
                A[i * n + k] -= A[j * n + k] * ba;
        }
    }

    for (j = n - 1; j >= 0; j--) {
        double a = A[j * n + j];
        for (i = j - 1; i >= 0; i--) {
            double ba = A[i * n + j] / a;
            for (k = n - 1; k >= j; k--)
                A[i * n + k] -= A[j * n + k] * ba;
        }
    }

	for(i = 0; i < n; i++) {
		//printf("%.16le\n", A[i * n + i]);
		vol *= A[i * n + i];
		if (l) {
			l[i] = A[i * n + i];
		}
	}

	free(X1); free(M);

	return vol;
}

/* модифицированный алгоритм Грамма-Шмидта
 * см. Голуб "Матричные Вычисления"
 */
void ortogonalize_cols_mgsch(double *A, int n, int m) {
	int i = 0, j = 0, k = 0;
	/*double sum;*/
	double * R = (double*)malloc(m * m * sizeof(double));

	for (k = 0; k < m; ++k) {
		R[k * m + k] = sqrt(matrix_cols_scalar(A, A, k, k, m, m, n));
		for (i = 0; i < n; ++i) {
			A[i * m + k] /= R[k * m + k];
		}

		for (j = k + 1; j < m; ++j) {
			R[k * m + j] = 0;
			/*произведение строки на столбец*/
			for (i = 0; i < n; ++i) {
				R[k * m + j] += A[i * m + k] * A[i * m + j];
			}

			for (i = 0; i < n; ++i) {
				A[i * m + j] -= A[i * m + k] * R[k * m + j];
			}
		}
	}

	free(R);
}

void ortogonalize_mgsch(double *A, int n, int m)
{
	matrix_transposew(A, n, m);
	ortogonalize_cols_mgsch(A, n, m);
	matrix_transposew(A, n, m);
}

/**
 * Процесс ортогонализации Грамма-Шмидта
 * с хранением векторов по стобцам
 * 
 * @param X - вектора, которые иртогонализуем, хранящиеся по столбцам
 * @param n - размерность пространства
 * @param m - число векторов
 */
void ortogonalize_cols_gsch(double *e, int n, int m) {
	int i = 0, j = 0, k = 0, u = 0;
//      double vlen;
	double sum;
	double *la=(double*)malloc(m*sizeof(double));
	/*можно уменьшить использование памяти в 2 раза !!!*/

/*временная матрица,
	для хранения базиса сдвинутого в 0*/
	double *ez=e;//new double [n*m]; //ортонормированный базис
	double norm;

	NOMEM(la);


	/*вычисляем вектора E_{k}*/
	/*векторов 'm' штук !!!*/
	for (k = 0; k < m; k++) {
		for (i = 0; i < k; i++) {
			la[i] = matrix_cols_scalar(ez, e, i, k, m, m, n);
		}
		/*по координатам*/
		for (j = 0; j < n; j++) {
			sum = 0.0;
			for (i = 0; i < k; i++) {
				sum += la[i] * ez[j * m + i];
			}

			ez[j * m + k]  = e[j * m + k] - sum;
		}

		norm = 1.0 / sqrt(matrix_cols_scalar(ez, ez, k, k, m, m, n));
		for (j = 0; j < n; j++) {
			ez[j * m + k] *= norm;
		}

                //check
/*		if (k > 0) {
			printf("%.16le\n",
				   matrix_cols_scalar(ez, ez, k, k - 1, m, m, n));
		}*/
	}

	free(la);
}

/**
 * ортопроекция на подпространство \f$<e_1,...,e_m>\f$
 * где \f$(e_i, e_j) = \delta_i^j \f$
 * @param h  - вектор который проецируем
 * @param h1 - проекция
 * @param e  - ортонормированный базис подпространства, записанный по столбцам
 * @param n  - размерность пространства
 * @param m  - размерность подпространства
 */
void ortoproj_ortonormbasis(double *h1, double *h, double *e, int n, int m) {
	int i, j;
	double *c = (double*)malloc(m*sizeof(double)); NOMEM(c);

	/* c[i]=(h,ei) */
	for (i = 0; i < m ; i++) {
		c[i] = 0.0;
		for (j = 0; j < n; j++) {
			c[i] += h[j] * e[j * m + i];
		}
	}
	
	/**
	 * если \f$(e_i,e_j)=0, i!=j, (e_i,e_i)=1\f$
	 * \f$h1 = \sum (h,e_i)e_i\f$
	 */
	for (j = 0; j < n; j++) {
		h1[j] = 0;
		for (i = 0; i < m; i++) {
			h1[j] += c[i] * e[j * m + i];
		}
	}
	
	free(c);
}

/**
 * ортопроекция на подпространство \f$<e_1,...,e_m>\f$
 * где \f$(e_i, e_j) = 0\f$, при \f$i != j\f$
 * @param h  - вектор который проецируем
 * @param h1 - проекция
 * @param e  - ортонормированный базис подпространства, записанный по столбцам
 * @param n  - размерность пространства
 * @param m  - размерность подпространства
 */
void ortoproj_ortobasis(double *h1, double *h, double *e, int n, int m) {
	int i, j;
	double *c = (double*)malloc(m*sizeof(double)); 
	double e_i_e_i; //скалярное произведение (ei, ei)
	NOMEM(c);

	/* c[i]=(h,ei) */
	for (i = 0; i < m ; i++) {
		e_i_e_i = matrix_cols_scalar(e, e, i, i, m, m, n);
		c[i] = 0.0;
		for (j = 0; j < n; j++) {
			c[i] += h[j] * e[j * m + i];
		}
		c[i] /= e_i_e_i;
	}
	
	/**
	 * если \f$(e_i,e_j)=0, i!=j\f$
	 * \f$h1 = \sum (h,e_i)e_i / (e_i, e_i)\f$
	 */
	for (j = 0; j < n; j++) {
		h1[j] = 0;
		for (i = 0; i < m; i++) {
			h1[j] += c[i] * e[j * m + i];
		}
	}
	
	free(c);
}

void projection(double *h1, double *h, double *e,
					double * e1, int n, int k, int l)
{
	/**
	 * стандартный базис подпространства fi = (0,...,1,...0)
	 * ______________________________________________i______
	 * h[i] - i'я координата в этом базисе
	 * h = \sum hi fi = \sum ci ei = sum ^nx ci ei + \sum ^n ci ei
	 * (h, fj) = hj = \sum_i ci (ei, fj) = \sum_i ci eij
	 * eij = E[j * n + i]
	 * c = \sum E^-1 h
	 * в качестве проекции берем часть суммы \sum ci ei
	 */
	int i, j;
	int free_e1 = 0;

	double * c;

	ERR((0 <= k && k < l && l <= n)," 0 <= k < l <= n must be ");

	c       = (double*)malloc(n*sizeof(double));   NOMEM(c);

	if (e1 == 0) {
		free_e1 = 1;
		e1      = (double*)malloc(n*n*sizeof(double)); NOMEM(e1);

		/*
		  транспонированная к обратной равно обратная к транспонированной
		  norm(inv(e')-inv(e)') = 0
		*/
		inverse_general_matrix(e1, e, n);
//		matrix_transpose(e1, n);
	}

	/* c = e^1 h */
	for (i = 0; i < n; ++i) {
		c[i] = 0;
		for (j = 0; j < n; ++j) {
			c[i] += e1[i * n + j] * h[j];
		}
	}
	
	for (j = 0; j < n; ++j) {
		h1[j] = 0;
		for (i = k; i < l; ++i) {
			h1[j] += c[i] * e[j * n + i];
		}
	}

	free(c);

	if (free_e1) free(e1);
}

double gauss( double * a, double *b, double *x,int n)
{   
	int i,j,k;
	double p;
	int imax;
	double nev=0;
	double Eps = 1.e-15;


	for(k = 0;k<n;k++)
	{
		imax=k;
		for(i=k+1;i<n;i++)
			if(fabs(a[i*n+k])>fabs(a[imax*n+k]))imax = i;

		for(j=k;j<n;j++)
		{p = a[imax*n+j]; a[imax*n+j] = a[k*n+j]; a[k*n+j]=p;}
		p = b[imax]; b[imax] = b[k]; b[k]=p;

		p=a[k*n+k];
		if(fabs(p) < Eps){printf("Gauss:Error!!!\n"); return -1.;}
		for(j=k;j<n;j++)
			a[k*n+j] = a[k*n+j]/p;
		b[k] = b[k]/p;

		for(i=k+1;i<n;i++)
		{
			p = a[i*n+k];
			for(j=k;j<n;j++)
				a[i*n+j] = a[i*n+j] -a[k*n+j]*p;
			b[i] = b[i] - b[k]*p;
		}

	}

	for(k = n-1;k>=0;k--)
	{
		x[k] = b[k];
		for(j=k+1;j<n;j++)
			x[k] = x[k] - x[j]*a[k*n+j];
	}

	nev=0.;

	for(i=0;i<n;i++)
	{
		p=0.;
		for(j=0;j<n;j++)
			p+=a[i*n+j]*x[j];

		p=b[i]-p;
		nev+=p*p;
	}

	return p;
}


int projection_test(const double *h, double *Pph, double *ex, double * ext, int m, int n)
{
	// fk.c
	static  double *A=NULL;
	static  double *B=NULL;
	static  double *C=NULL;


	int i,j,k;
	double r;
	double nev,p;

	static double * Ppbase = 0; 
	static double * Pmort  = 0;

	//   initP();

	if(B==NULL)
	{
		A = malloc(m * m * sizeof(double));
		B = malloc(m * sizeof(double));
		C = malloc(m * sizeof(double));

		Ppbase = malloc(m * sizeof(double));
		Pmort  = malloc(m * sizeof(double));
	}



	for(j=0;j<m;j++)
	{
		for (k = 0; k < n; ++k) {
			Ppbase[k] = ex [k * m + j];
		}

		for(i=0;i<m;i++)
		{

			for (k = 0; k < n; ++k) {
				Pmort [k] = ext[k * m + i];
			}

			A[i*m+j]=scalar(Ppbase, Pmort, n);
			//     A[i][j]=scalar(z.Pmort[i],z.Pmort[j],s);
			//     A[i][j]=scalar(z.Ppbase[i],z.Ppbase[j],s);
		}
	}


	_fprintfwmatrix("out/_gram2.txt", A, m, m, m, "%23.16le ");

	for(j=0;j<m;j++)
	{
		for (i = 0; i < n; ++i) {
			Ppbase[i] = ex [i * m + j];
			Pmort [i] = ext[i * m + j];
		}

		r=scalar(h,Pmort, n);
		//      r=scalar(h,z.Ppbase[j],s);
		B[j]=r;
	}

	//printf("\n Before Gauss B\n");
	//for(i=0;i<s.Np;i++)
	//printf("%lf ",B[i]);
	//printf("\n");


	//print_matr_double(stdout, "Before Gauss:A", A,s.Np,s.Np);

	nev = gauss( A, B, C, m);

	// printf("Gauss ERR = %g\n",nev);

	//print_matr_double(stdout, "After Gauss:A", A,s.Np,s.Np);

	//printf("C\n");
	//for(i=0;i<s.Np;i++)
	//printf("%lf ",C[i]);
	//printf("\n");

	for(i=0;i<n;i++)
		Pph[i]=0.;

	for(k=0;k<m;k++)
		for(i=0;i<n;i++)
			Pph[i]=Pph[i]+ex[i*m+k]*C[k];
	//   Pph[i]=Pph[i]+z.Pmort[k][i]*C[k];



	//for(j=0;j<s.Np;j++)
	//{
	//	for(i=0;i<s.Np;i++)
	//	{
	//		A[i][j]=kor_scalar(z.Ppbase[j],z.Pmort[i],s);
	//	}
	//}


	//for(j=0;j<s.Np;j++)
	//{
	//	r=kor_scalar(h,z.Pmort[j],s);
	//	B[j]=r;
	//}

	//nev=0.;

	//for(i=0;i<s.Np;i++)
	//{
	//	p=0.;
	//	for(j=0;j<s.Np;j++)
	//		p+=A[i][j]*C[j];

	//	p=B[i]-p;
	//	nev+=p*p;

	//}

	// printf("Pp =%f\n", nev);
	return 0;
}


/**
 * проекция на подпространство \f$e_y=<e_1,...,e_{n_y}>\f$ пространства \f$<e_1,...,e_n>\f$
 * @param h   - вектор который проецируем
 * @param h1  - проекция
 * @param e   - базис подпространства на которое проецируем
 * @param et  - сопряженный базис (с. в. транспонированной матрицы)
 * @param ete - матрица обратная к \f$(e_i, et_j)\f$ если 0, то вычисляем
 * @param m   - размерность подпространства проектирования
 * @param n   - размерность всего пространства
 */
void projection2(double *h1, const double *h, const double *e, const double *et,
					 double * ete, int m, int n)
{
	/**
	 * Теорема \f$A = <e_i ... e_nx >\f$ \f$<e_{nx+1} ... e_{nx + ny}>\f$
	 * \f$A^t = <e*_i ... e*_nx >\f$ \f$<e*_{nx+1} ... e*_{nx + ny}>\f$
	 * (транспонированная матрица собственные значения те же, 
	 *  собственные вектора отличаются)
	 * тогда \f$<e_i ... e_nx >\f$ ортогонально \f$<e*_{nx+1} ... e*_{nx + ny}>\f$ 
	 * \f$<e*_i ... e*_nx >\f$ ортогонально \f$<e_{nx+1} ... e_{nx + ny}>\f$
	 */

	int i, j;
	int do_not_free_a = 0;
	double *c  = 0;
	double *rp = 0;
	double *a  = 0;

//	assert(0&&"unverified code, use SDS alternative!");

	c  = (double *)malloc(m * sizeof(double));     NOMEM(c);
	rp = (double *)malloc(m * sizeof(double));     NOMEM(rp);
	if (ete == 0) {
		a = (double *)malloc(m * m * sizeof(double)); NOMEM(a);
		inverse_gramm_matrix(a, e, et, m, n);
	} else {
		a = ete;
		do_not_free_a = 1;
	}

	//rp = (h, e*_j)
	for (i = 0; i < m; i++) {
		rp[i] = matrix_cols_scalar(h, et, 0, i, 1, m, n);
	}
	

	//c = a * rp
	for (i = 0; i < m; i++) {
		c[i] = 0;
		for (j = 0; j < m; j++) {
			c[i] += rp[j] * a[j * m + i];
		}
	}

	for (j = 0; j < n; j++) {
		h1[j] = 0;
		for (i = 0; i < m; i++) {
			h1[j] += c[i] * e[j * m + i];
		}
	}
	free(c); free(rp);
	if (do_not_free_a != 1) free(a);
}

void projection2_ext(double *c, const double *h, const double *e, 
					 const double *et,
					 double * ete, int m, int n)
{	
	/**
	 * Теорема \f$A = <e_i ... e_nx >\f$ \f$<e_{nx+1} ... e_{nx + ny}>\f$
	 * \f$A^t = <e*_i ... e*_nx >\f$ \f$<e*_{nx+1} ... e*_{nx + ny}>\f$
	 * (транспонированная матрица собственные значения те же, 
	 *  собственные вектора отличаются)
	 * тогда \f$<e_i ... e_nx >\f$ ортогонально \f$<e*_{nx+1} ... e*_{nx + ny}>\f$ 
	 * \f$<e*_i ... e*_nx >\f$ ортогонально \f$<e_{nx+1} ... e_{nx + ny}>\f$
	 */

	int i, j;
	int do_not_free_a = 0;
	double *rp = 0;
	double *a  = 0;

	assert(1&&"unverified code, use SDS alternative!");

	rp = (double *)malloc(m * sizeof(double));     NOMEM(rp);
	if (ete == 0) {
		a = (double *)malloc(m * m * sizeof(double)); NOMEM(a);
		inverse_gramm_matrix(a, e, et, m, n);
	} else {
		a = ete;
		do_not_free_a = 1;
	}

	//rp = (h, e*_j)
	for (i = 0; i < m; i++) {
		rp[i] = matrix_cols_scalar(h, et, 0, i, 1, m, n);
	}
	

	//c = a * rp
	for (i = 0; i < m; i++) {
		c[i] = 0;
		for (j = 0; j < m; j++) {
			c[i] += rp[j] * a[j * m + i];
		}
	}

	free(rp);
	if (do_not_free_a != 1) free(a);
}

/**
 * ортопроекция на подпространство \f$e_y=<e_1,...,e_{n_y}>\f$ пространства \f$<e_1,...,e_n>\f$
 * @param h  - вектор который проецируем
 * @param h1 - проекция
 * @param e  - базис подпространства на которое проецируем
 * @param ete - матрица обратная к \f$(e_i, e_j)\f$ если 0, то вычисляем
 */
void ortoproj_general(double *h1, double *h, double *e, 
					 double * ete, int m, int n)
{
	int i, j;
	int do_not_free_a = 0;
	double *c  = 0;
	double *rp = 0;
	double *a  = 0;

	c  = (double *)malloc(m * sizeof(double));     NOMEM(c);
	rp = (double *)malloc(m * sizeof(double));     NOMEM(rp);
	if (ete == 0) {
		a  = (double *)malloc(m * m * sizeof(double)); NOMEM(a);
		inverse_gramm_matrix(a, e, e, m, n);
	} else {
		a  = ete;
		do_not_free_a = 1;
	}

	//rp = (h, e*_j)
	for (i = 0; i < m; i++) {
		rp[i] = matrix_cols_scalar(h, e, 0, i, 1, m, n);
	}
	

	//c = a * rp
	for (i = 0; i < m; i++) {
		c[i] = 0;
		for (j = 0; j < m; j++) {
			c[i] += rp[j] * a[j * m + i];
		}
	}

	for (j = 0; j < n; j++) {
		h1[j] = 0;
		for (i = 0; i < m; i++) {
			h1[j] += c[i] * e[j * m + i];
		}
	}
	free(c); free(rp);
	if (do_not_free_a != 1) free(a);
}

/**
 * LAPACK'based
 * ортопроекция на подпространство \f$<e_1,...,e_m>\f$
 * минимизируем \f$\|h - e h_1\|\f$
 * используя QR разложение
 * @param h  - вектор который проецируем
 * @param h1 - проекция
 * @param e  - ортонормированный базис подпространства, записанный по столбцам
 * @param n  - размерность пространства
 * @param m  - размерность подпространства
 */
void ortoproj_lls_qr_la(double *h1, double *h, double *e, int n, int m) {
	char trans = 'T'; //транспонированная матрица? 'N'
	int info;
	int lwork = n * n;
	int ldb  = n;
	int lda  = n;
	int nrhs = 1;
	double * a    = 0;
	double * work = 0;
	
	a    = (double *) malloc(n * m * sizeof(double)); NOMEM(a);
	work = (double *) malloc(lwork * sizeof(double)); NOMEM(work);
	memcpy(a, e, n * m * sizeof(double));
	memcpy(h1, h, n * sizeof(double));
	dgels_(&trans, &m, &n, &nrhs, a, &lda, h1, &ldb, work, &lwork, &info);
	ERR((info == 0), "error in lapack's dgels_ procedure");
	free(a); free(work);
}

///**
// * LAPACK'based
// * ортопроекция на подпространство \f$<e_1,...,e_m>\f$
// * минимизируем \f$\|h - e h_1\|\f$
// * используя complete orthogonal factorization
// * @param h  - вектор который проецируем
// * @param h1 - проекция
// * @param e  - ортонормированный базис подпространства, записанный по столбцам
// * @param n  - размерность пространства
// * @param m  - размерность подпространства
// */
//void ortoproj_lls_cof_la(double *h1, double *h, double *e, int n, int m) {
//	int info;
//	int lwork = n * n;
//	int ldb  = n;
//	int lda  = n;
//	int nrhs = 1;
//	double * a    = 0;
//	double * work = 0;
//	int * jpvt = 0;
//	
//	a    = (double *) malloc(n * m * sizeof(double)); NOMEM(a);
//	work = (double *) malloc(lwork * sizeof(double)); NOMEM(work);
//	jpvt = (int *) malloc(n * sizeof(int)); NOMEM(jpvt);
//	memcpy(a, e, n * m * sizeof(double));
//	memcpy(h1, h, n * sizeof(double));
//
//	dgelsy_(&m, &n, &nrhs, a, &lda, n, &ldb, jpvt, &rcond, rank, work, &lwork, &info);
//	ERR((info == 0), "error in lapack's dgelsy_ procedure");
//	free(a); free(work); free(jpvt);
//}
//
///**
// * LAPACK'based
// * ортопроекция на подпространство \f$<e_1,...,e_m>\f$
// * минимизируем \f$\|h - e h_1\|\f$
// * используя SVD
// * @param h  - вектор который проецируем
// * @param h1 - проекция
// * @param e  - ортонормированный базис подпространства, записанный по столбцам
// * @param n  - размерность пространства
// * @param m  - размерность подпространства
// */
//	void ortoproj_lls_svd_la(double *h1, double *h, double *e, int n, int m);
//
///**
// * LAPACK'based
// * ортопроекция на подпространство \f$<e_1,...,e_m>\f$
// * минимизируем \f$\|h - e h_1\|\f$
// * используя divide-and-conquer SVD
// * @param h  - вектор который проецируем
// * @param h1 - проекция
// * @param e  - ортонормированный базис подпространства, записанный по столбцам
// * @param n  - размерность пространства
// * @param m  - размерность подпространства
// */
//	void ortoproj_lls_dacsvd_la(double *h1, double *h, double *e, int n, int m);
