#ifndef _QR_CORE_H
#define _QR_CORE_H
/*$Id$*/

/* Copyright (c) 2002, 2004, 2005 Alexey Ozeritsky
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

#ifdef _MPI_BUILD
#include <mpi.h>
#endif

#define _TEST_LAPACK

#ifdef __cplusplus
extern "C" {
#endif
/**
 * QR разложение
 * Алгоритм взят из книги К. Ю. Богачёв, Практикум на ЭВМ.
 * Методы решения линейных систем и нахождения собственных значений.
 * Москва 1999.
 * стр 63-75
 */

 /**
  * @param A - матрица, которую вращаем
  * @param Q - транспонированное произведение матриц элементарного вращения
  * если 0, то не вычисляем
  */
	void rotate_general(double*A,double *Q,int n);

/**
 * Приведение матрицы к почти треугольному виду
 * унитарным подобием методом вращений
 * @param A - матрица
 * @param n - размерность
 */
	void rotateu_to_ptriang(double *A,int n);

/**
 * Приведение матрицы к почти треугольному виду
 * унитарным подобием методом вращений
 * @param A - матрица, которую приводим к почти треугольному
 * @param Z - матрица, которую вращаем также
 * @param n - размерность
 */
	void rotateu_to_ptriang_z(double *A, double *Z, int n);

/**
 * Приведение симметричной матрицы к трёхдиагональному виду
 * унитарным подобием методом вращений
 *
 * @param A - матрица
 * @param n - размерность
 */
	void rotateu_to_tdiag(double *A,int n);

/**
 * Приведение матрицы к почти треугольному виду
 * унитарным подобием методом отражений
 *
 * @param A - матрица
 * @param n - размерность
 *
 * источник этой функции - программа Ерошина, поэтому правильность
 * результата не гарантирована
 */
	void reflectu_to_ptriang(double *A,int n);


/**
 * метод вращений для почти треугольной матрицы. Вращение подматрицы
 * @param A - матрица
 * @param C - вектор для хранения косинусов матрицы вращений Q
 * @param S - вектор для хранения синусов
 * @param n - размерность
 * @param m - размерность подматрицы
 */
	void rotate_ptriang_sub(double *A,double *C,double *S,int n,int i);
	void rotate_ptriang_sub_z(double *A,double *Z, double *C,double *S,
							int n,int i);

/**
 * метод отражений для трёхдиагональной сим матрицы. отражение подматрицы
 * @param A - матрица
 * @param tmp - вектор для хранения
 * @param tmp2 - вектор для хранения
 * @param n - размерность
 * @param m - размерность подматрицы
 */
	void reflect_tdiag_sub(double*A,double*tmp,double*tmp2,int n,int m);

/**
 * решение проблеммы собственных значений для
 * общего случая
 * @param A - матрица
 * @param Xr - вещественные части собственных значений
 * @param Xi - мнимые части собственных значений
 * @param n - размерность
 * @param eps - точность
 */
	void ev_general(double *A,double *Xr,double *Xi,int n,double eps);

	void sdvig(double*A,int n,int m,double s);

/**
 * нахождение матрицы A^{k+1}=RQ в случае когда, для подматрицы
 * @param Q задана векторами косинусов и синусов
 * @param A - матрица
 * @param C - вектор для хранения косинусов матрицы вращений Q
 * @param S - вектор для хранения синусов
 * @param n - размерность
 * @param m - размерность подматрицы
 */
	void RQ_rotate_step(double *A,double *C,double *S,int n,int i);
	void RQ_rotate_step_z(double *A,double *Z,double *C,double *S,int n,int i);

/**
 * нахождение матрицы A^{k+1}=RQ
 * @param Q - получена с помощью отражений
 * @param A - матрица
 * @param C - вектор для хранения
 * @param S - вектор для хранения
 * @param n - размерность
 * @param m - размерность подматрицы
 */
	void RQ_reflect_step(double *A,double *tmp,double *tmp2,int n,int m);

/**
 * нахождение матрицы A=QR в случае когда, для подматрицы
 * @param Q задана векторами косинусов и синусов
 * @param A - матрица
 * @param C - вектор для хранения косинусов матрицы вращений Q
 * @param S - вектор для хранения синусов
 * @param n - размерность
 * @param m - размерность подматрицы
 */
	void QR_rotate_step(double *A,double *C,double *S,int n,int m);


/**
 * решение проблеммы собственных значений для
 * симметричной матрицы
 * @param A - матрица
 * @param X - собственные значения
 * @param n - размерность
 * @param eps - точность
 */
	void ev_sym(double*A,double*X,int n,double eps);

/**
 * решение проблемы собственных значений
 * и функций для общего случая
 * @param A - матрица
 * @param X - вектор собственных значений
 * @param V - базис подпространств
 * @param n - размерность пространства
 * @param eps - точность
 */
	void ev_vectors_general_old(double *A, double *X, double *V
								, int n, double eps);

/**
 * решение проблеммы собственных значений для
 * общего случая
 * @param A - матрица
 * @param Z - собственные вектора
 * @param Xr - вещественные части собственных значений
 * @param Xi - мнимые части собственных значений
 * @param n - размерность
 * @param eps - точность
 */
	void ev_vectors_general(double *A,double *Z, double *Xr,double *Xi,int n
							,double eps);

/**
 * решение проблемы собственных значений для общего случая и
 * нахождение подпространств, отвечающих собственным значениям
 * по модулю больше единицы и меньше единицы
 * @param A - матрица
 * @param X - вектор собственных значений
 * @param V - базис подпространств
 * @param n - размерность пространства
 * @param nx - размерность подпространства, отвечающего собственному
 *      значению по модулю больше 1
 * @param ny - размерность подпространства, отвечающего собственному
 *      значению по модулю меньше 1
 * @param eps - точность
 */
	void ev_subspace_general(double *A, double *X, double *V, int n
							, int nx, int ny
							, double eps);

/**
 * по образу и прообразу n n'мерных линейнонезависимых векторов
 * находит матрицу линеаризации отображения
 * @param A - искомая матрица линеаризации
 * @param X - n'векторов прообразов (хранятся по столбцам)
 * @param Y - n'векторов образов (хранятся по столбцам)
 * @param n - размерность пространства
 */
	void linearization(double *A, double *x, double *y, int n);

/**
 * LAPACK'based
 * решение проблемы собственных значений для общего случая и
 * нахождение подпространств, отвечающих собственным значениям
 * по модулю больше единицы и меньше единицы
 * память на Ex, Ey выделяется внутри, ибо не знаем размеры!
 * при выделении памяти под матрицу классификации внутри функции
 * надо обязательно выставить исходный указатель в нуль!
 * @param A  - матрица
 * @param X  - собственные значения
 * @param V  - собственные вектора (если 0, то не выч)
 * @param Ex - собственные вектора, отвечающие подпространству X (если 0, то не выч)
 * @param Ey - собственные вектора, отвечающие подпространству Y (если 0, то не выч)
 * @param n  - размерность пространства
 * @param nx - размерность подпространства, отвечающего собственному
 *      значению по модулю больше 1 (если 0, то не вычисляем. в этом случае Ex, Ey Тоже не выч)
 * @param ny - размерность подпространства, отвечающего собственному
 *      значению по модулю меньше 1 (если 0, то не вычисляем. в этом случае Ex, Ey Тоже не выч)
 * @param criteria - критерий принадлежности подпространствам
 * @param absolute - критерий считается по модулю?
 */
#define _LAPACK_PRINT_VALUES
void ev_subspace_general_lapack(double *A, double *X, double *V
								, double **lx, double **ly
								, double **Ex, double **Ey
								, int n, int *nx, int *ny
								, double criteria, int absolute);

/*!тоже самое что и ev_subspace_general_lapack но на базе arpack'а*/
void ev_subspace_general_arpack(double *A, double *X, double *V
								, double **lx, double **ly
								, double **Ex, double **Ey
								, int n, int *nx, int *ny
								, double criteria, int absolute);

int check_dnaupd_status(int info);
int check_dneupd_status(int info);

#define _ARPACK_MAX_ITER 1000
#ifdef _MPI_BUILD
	/*работа с транспонированной матрицей в процедуре av?*/
	#define _PARPACK_TRANSPOSE_MATRIX
	void pdnaupd_(int * comm, int*ido, char*bmat, int*n,
			  char *which, int*nev, double *tol,
			  double *resid, int *ncv, double *v, int*ldv,
			  int *iparam, int *ipntr, double *workd,
			  double *workl, int *lworkl, int *info);
	void    pdneupd_ ( int *comm, int *vec, char * c, int * select,
					   double * d, double * /*d(1,2)*/,
					   double *v, int *ldv,
					   double * sigmar, double * sigmai,
					   double * workev, char * bmat, int *n,
					   char * which, int *nev, double * tol,
					   double * resid, int *ncv, double *vv,
					   int * ldvv, int * iparam, int * ipntr, double * workd,
					   double * workl, int * lworkl, int *ierr );

	/*!вычисление X подпространства с помощью MPI
	  методом Арнольди.
	  \param comm - MPI коммуникатор
	  \param A    - строки матрицы, с которыми работает данный процесс
	  \param l    - собственные значения, соответсвующие пространству X
	  \param Ex   - подпространство X, память выделяется malloc'ом
	  ответ кладется в нулевой процесс
	  \param n    - размерность пространства
	  \param nx   - оценка с верху размерность пространства nx,
	  сюда же кладется реальная размерность nx
	  ограничение на nx - 1 < nx <= n - 2
	*/
	void ev_x_general_parpack_mpi(MPI_Comm comm,
								  double *A,
								  double **l,
								  double **Ex, int n,
								  int *nx);
#endif
/*! вычисляет базис пространства X
\param A  - матрица линеаризации
\param Ex - базис X (память выделяется malloc'ом)
\param n  - размерность A
\param nx - предположительная размерность X (может увеличится
            за счет кратных значений)
*/
void ev_x_subspace_general_arnoldi(double *A, double **Ex, int n,
								  int *nx);
void ev_x_subspace_general_lanczos(double *A, double **Ex, int n,
								  int *nx);

/*E - полный базис
  если 0, то не возвращаем
*/
void ev_x_subspace_general_lapack2(const double * A, double **Ex, 
								   double ** E, int n, int *nx); 

void ev_x_subspace_general_lapack3(const double * A, double **Ex, int n, int *nx); 

/*mode=3*/
void ev_x_subspace_general_arnoldi2(double *A1, double **Ex, int n,
								  int *nx);

void ev_x_subspace_iteration(const double * A, double **Ex, int n, int *nx);

/**
 * заголовки для фортрановской библиотеки lapack,
 * подключение -llapack
 * заголовки для остальных функций пишутся аналогично
 * аргументы в фортране передаются по ссылке
 */
/**
 * DGEEV - compute for an N-by-N real nonsymmetric matrix A, the
 * eigenvalues and, optionally, the left and/or right eigenvectors
 *
 * ARGUMENTS
 *      @param JOBVL   (input) CHARACTER*1
 *              = 'N': left eigenvectors of A are not computed;
 *              = 'V': left eigenvectors of A are computed.
 *
 *      @param JOBVR   (input) CHARACTER*1
 *              = 'N': right eigenvectors of A are not computed;
 *              = 'V': right eigenvectors of A are computed.
 *
 *      @param N       (input) INTEGER
 *              The order of the matrix A. N >= 0.
 *
 *     @param  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *              On  entry,  the N-by-N matrix A.  On exit, A has been
 *              overwritten.
 *
 *      @param LDA     (input) INTEGER
 *              The leading dimension of the array A.  LDA >= max(1,N).
 *
 *      @param WR      (output) DOUBLE PRECISION array, dimension (N)
 *     @param  WI      (output) DOUBLE PRECISION array, dimension (N)  WR  and
 *              WI  contain  the real and imaginary parts, respectively, of the
 *              computed eigenvalues.  Complex conjugate pairs  of  eigenvalues
 *              appear  consecutively  with  the eigenvalue having the positive
 *              imaginary part first.
 *
 *     @param  VL      (output) DOUBLE PRECISION array, dimension (LDVL,N)
 *              If JOBVL = 'V', the left eigenvectors u(j) are stored one after
 *              another in the columns of VL, in the same order as their eigen-
 *              values.  If JOBVL = 'N', VL is not referenced.  If the j-th ei-
 *              genvalue  is  real, then u(j) = VL(:,j), the j-th column of VL.
 *              If the j-th and (j+1)-st eigenvalues form a  complex  conjugate
 *              pair, then u(j) = VL(:,j) + i*VL(:,j+1) and
 *              u(j+1) = VL(:,j) - i*VL(:,j+1).
 *
 *    @param   LDVL    (input) INTEGER
 *              The  leading  dimension of the array VL.  LDVL >= 1; if JOBVL =
 *              'V', LDVL >= N.
 *
 *     @param  VR      (output) DOUBLE PRECISION array, dimension (LDVR,N)
 *              If JOBVR = 'V', the right  eigenvectors  v(j)  are  stored  one
 *              after  another in the columns of VR, in the same order as their
 *              eigenvalues.  If JOBVR = 'N', VR is not referenced.  If the  j-
 *              th  eigenvalue is real, then v(j) = VR(:,j), the j-th column of
 *              VR.  If the j-th and (j+1)-st eigenvalues form a complex conju-
 *              gate pair, then v(j) = VR(:,j) + i*VR(:,j+1) and
 *              v(j+1) = VR(:,j) - i*VR(:,j+1).
 *
 *    @param   LDVR    (input) INTEGER
 *              The  leading  dimension of the array VR.  LDVR >= 1; if JOBVR =
 *              'V', LDVR >= N.
 *
 *     @param  WORK    (workspace/output) DOUBLE PRECISION array, dimension (LWORK)
 *              On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *    @param  LWORK   (input) INTEGER
 *              The dimension of the array WORK.  LWORK >= max(1,3*N),  and  if
 *              JOBVL  =  'V'  or  JOBVR = 'V', LWORK >= 4*N.  For good perfor-
 *              mance, LWORK must generally be larger.
 *
 *              If LWORK = -1, then a workspace query is assumed;  the  routine
 *              only  calculates  the  optimal  size of the WORK array, returns
 *              this value as the first entry of the WORK array, and  no  error
 *              message related to LWORK is issued by XERBLA.
 *
 *     @param  INFO    (output) INTEGER
 *              = 0:  successful exit
 *              < 0:  if INFO = -i, the i-th argument had an illegal value.
 *              >  0:   if INFO = i, the QR algorithm failed to compute all the
 *              eigenvalues, and no eigenvectors have been  computed;  elements
 *              i+1:N of WR and WI contain eigenvalues which have converged.
 */
	void dgeev_(char* JOBVL, char* JOBVR, int* N, double* A, int* LDA,
				double* WR, double* WI, double* VL, int* LDVL, double* VR,
				int* LDVR, double* WORK, int* LWORK, int *INFO );


	/*@{ svd*/
	void ev_subspace_svd_lapack(double *A, double **lx 
							, double **Ex, double **Ext
							, int n, int *nx 
							, double criteria);

	/*! угол между подпространствами Y1 и Y2
	    Y2 задается базисом сопряженного пространства
		n  - размерность пространства
		nx - размерность подпространств
	 */
	double ev_subspace_angle(const double * Y1, const double * Y2t, int n, int nx);

	/*нахождение сингулярных значений*/
	void ev_singular_svd_lapack(double * A, double *lx, int n);

	void dgesdd_(
		char * jobz, 
		int *M /*number of rows of A*/,
		int *N /*number of cols*/,
		double * A,
		int *lda,
		double *S, /*singular values (S[i+1]>S[i])*/
		double *U, int *ldu, double *VT,
		int * ldvt, double *WORK,
		int *lwork,
/*          The dimension of the array WORK. LWORK >= 1.
*          If JOBZ = 'N',
*            LWORK >= max(14*min(M,N)+4, 10*min(M,N)+2+
*                     SMLSIZ*(SMLSIZ+8)) + max(M,N)
*          where SMLSIZ is returned by ILAENV and is equal to the
*          maximum size of the subproblems at the bottom of the
*          computation tree (usually about 25).
*          If JOBZ = 'O',
*            LWORK >= 5*min(M,N)*min(M,N) + max(M,N) + 9*min(M,N).
*          If JOBZ = 'S' or 'A'
*            LWORK >= 4*min(M,N)*min(M,N) + max(M,N) + 9*min(M,N).
*          For good performance, LWORK should generally be larger.
*
*          If LWORK = -1, a workspace query is assumed.  The optimal
*          size for the WORK array is calculated and stored in WORK(1),
*          and no other work except argument checking is performed.*/
		int *IWORK,/* (workspace) INTEGER array, dimension (8*min(M,N))*/
		int *info);

      void dgesvd_( 
		  char * JOBU, 
/*          = 'A':  all M columns of U are returned in array U:
 *          = 'S':  the first min(m,n) columns of U (the left singular
 *                  vectors) are returned in the array U;
 *          = 'O':  the first min(m,n) columns of U (the left singular
 *                  vectors) are overwritten on the array A;
 *          = 'N':  no columns of U (no left singular vectors) are
 *                  computed.*/

		  char * JOBVT, 
/*          Specifies options for computing all or part of the matrix
 *          V**T:
 *          = 'A':  all N rows of V**T are returned in the array VT;
 *          = 'S':  the first min(m,n) rows of V**T (the right singular
 *                  vectors) are returned in the array VT;
 *          = 'O':  the first min(m,n) rows of V**T (the right singular
 *                  vectors) are overwritten on the array A;
 *          = 'N':  no rows of V**T (no right singular vectors) are
 *                  computed. */
		  
		  int * M, /*The number of rows of the input matrix A.  M >= 0.*/
		  int * N, /*The number of columns of the input matrix A.  N >= 0.*/
		  double * A, 
/*          On entry, the M-by-N matrix A.
 *          On exit,
 *          if JOBU = 'O',  A is overwritten with the first min(m,n)
 *                          columns of U (the left singular vectors,
 *                          stored columnwise);
 *          if JOBVT = 'O', A is overwritten with the first min(m,n)
 *                          rows of V**T (the right singular vectors,
 *                          stored rowwise);
 *          if JOBU .ne. 'O' and JOBVT .ne. 'O', the contents of A
 *                          are destroyed. */
		  		  
		  int * LDA, /*The leading dimension of the array A.  LDA >= max(1,M).*/
		  double * S, /*The singular values of A, sorted so that S(i) >= S(i+1).*/
		  double * U, /*array, dimension (LDU,UCOL) */
/*          (LDU,M) if JOBU = 'A' or (LDU,min(M,N)) if JOBU = 'S'.
 *          If JOBU = 'A', U contains the M-by-M orthogonal matrix U;
 *          if JOBU = 'S', U contains the first min(m,n) columns of U
 *          (the left singular vectors, stored columnwise);
 *          if JOBU = 'N' or 'O', U is not referenced. */		  
		  int * LDU, 
/*          The leading dimension of the array U.  LDU >= 1; if
 *          JOBU = 'S' or 'A', LDU >= M. */		  
		  double * VT, /*array, dimension (LDVT,N)*/
/*          If JOBVT = 'A', VT contains the N-by-N orthogonal matrix
 *          V**T;
 *          if JOBVT = 'S', VT contains the first min(m,n) rows of
 *          V**T (the right singular vectors, stored rowwise);
 *          if JOBVT = 'N' or 'O', VT is not referenced. */
		  int * LDVT,
/*          The leading dimension of the array VT.  LDVT >= 1; if
 *          JOBVT = 'A', LDVT >= N; if JOBVT = 'S', LDVT >= min(M,N). */
          double * WORK, /*array, dimension (LWORK)*/
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK;
 *          if INFO > 0, WORK(2:MIN(M,N)) contains the unconverged
 *          superdiagonal elements of an upper bidiagonal matrix B
 *          whose diagonal is in S (not necessarily sorted). B
 *          satisfies A = U * B * VT, so it has the same singular values
 *          as A, and singular vectors related by U and VT. */		 
		  int * LWORK, 
/*          The dimension of the array WORK. LWORK >= 1.
 *          LWORK >= MAX(3*MIN(M,N)+MAX(M,N),5*MIN(M,N)).
 *          For good performance, LWORK should generally be larger.
 *
 *          If LWORK = -1, a workspace query is assumed.  The optimal
 *          size for the WORK array is calculated and stored in WORK(1),
 *          and no other work except argument checking is performed. */
		  int * INFO 
/*          = 0:  successful exit.
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          > 0:  if DBDSQR did not converge, INFO specifies how many
 *                superdiagonals of an intermediate bidiagonal form B
 *                did not converge to zero. See the description of WORK
 *                above for details.*/		  
		  );

	/*@}*/


	  /*вычисляет upper Heisenberg матрицу*/
      void dgehrd_(int * N, /*The order of the matrix A.  N >= 0.*/

/*          It is assumed that A is already upper triangular in rows
 *          and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
 *          set by a previous call to DGEBAL; otherwise they should be
 *          set to 1 and N respectively. See Further Details.
 *          1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0. */

		  int * ILO,  /*1*/
		  int * IHI,  /*N*/
		  double * A, 
		  int * LDA, 

/*  TAU     (output) DOUBLE PRECISION array, dimension (N-1)
 *          The scalar factors of the elementary reflectors (see Further
 *          Details). Elements 1:ILO-1 and IHI:N-1 of TAU are set to
 *          zero. */

		  double * TAU, 

/*          The length of the array WORK.  LWORK >= max(1,N).
 *          For optimum performance LWORK >= N*NB, where NB is the
 *          optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA. */

		  double * WORK, 
		  int * LWORK, 
		  int * INFO);

	  /*собственные значения матрицы H*/
	  void dhseqr_( 
/*           = 'E':  compute eigenvalues only;
 *           = 'S':  compute eigenvalues and the Schur form T. */
		  char * JOB, 
/*           = 'N':  no Schur vectors are computed;
 *           = 'I':  Z is initialized to the unit matrix and the matrix Z
 *                   of Schur vectors of H is returned;
 *           = 'V':  Z must contain an orthogonal matrix Q on entry, and
 *                   the product Q*Z is returned. */
		  char * COMPZ, 
		  int * N, 

/*           It is assumed that H is already upper triangular in rows
 *           and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
 *           set by a previous call to DGEBAL, and then passed to DGEHRD
 *           when the matrix output by DGEBAL is reduced to Hessenberg
 *           form. Otherwise ILO and IHI should be set to 1 and N
 *           respectively.  If N.GT.0, then 1.LE.ILO.LE.IHI.LE.N.
 *           If N = 0, then ILO = 1 and IHI = 0. */
		  int * ILO, 
		  int * IHI, 
		  double * H, 
		  int * LDH, 

		  /*собственные значения вещественная и мнимые части*/
		  double * WR, 
		  double * WI, 

/*           If COMPZ = 'N', Z is not referenced.
 *           If COMPZ = 'I', on entry Z need not be set and on exit,
 *           if INFO = 0, Z contains the orthogonal matrix Z of the Schur
 *           vectors of H.  If COMPZ = 'V', on entry Z must contain an
 *           N-by-N matrix Q, which is assumed to be equal to the unit
 *           matrix except for the submatrix Z(ILO:IHI,ILO:IHI). On exit,
 *           if INFO = 0, Z contains Q*Z.
 *           Normally Q is the orthogonal matrix generated by DORGHR
 *           after the call to DGEHRD which formed the Hessenberg matrix
 *           H. (The output value of Z when INFO.GT.0 is given under
 *           the description of INFO below.) */

		  double * Z,
		  int * LDZ, 
		  double * WORK, 

/*           The dimension of the array WORK.  LWORK .GE. max(1,N)
 *           is sufficient, but LWORK typically as large as 6*N may
 *           be required for optimal performance.  A workspace query
 *           to determine the optimal workspace size is recommended.
 *
 *           If LWORK = -1, then DHSEQR does a workspace query.
 *           In this case, DHSEQR checks the input parameters and
 *           estimates the optimal workspace size for the given
 *           values of N, ILO and IHI.  The estimate is returned
 *           in WORK(1).  No error message related to LWORK is
 *           issued by XERBLA.  Neither H nor Z are accessed. */

		  int * LWORK, 
		  int * INFO );

/*подпространство, базис*/
      void dtrsen_(
/*          Specifies whether condition numbers are required for the
 *          cluster of eigenvalues (S) or the invariant subspace (SEP):
 *          = 'N': none;
 *          = 'E': for eigenvalues only (S);
 *          = 'V': for invariant subspace only (SEP);
 *          = 'B': for both eigenvalues and invariant subspace (S and
 *                 SEP). */

		  char * JOB, 
/*          = 'V': update the matrix Q of Schur vectors;
 *          = 'N': do not update Q. */
		  char * COMPQ, 
/*          SELECT specifies the eigenvalues in the selected cluster. To
 *          select a real eigenvalue w(j), SELECT(j) must be set to
 *          .TRUE.. To select a complex conjugate pair of eigenvalues
 *          w(j) and w(j+1), corresponding to a 2-by-2 diagonal block,
 *          either SELECT(j) or SELECT(j+1) or both must be set to
 *          .TRUE.; a complex conjugate pair of eigenvalues must be
 *          either both included in the cluster or both excluded. */
		  int * SELECT, 
		  int * N, 
		  /*матрица которая возвращает dhseqr_ -- матрица H */
		  double * T, 
		  int * LDT, 
		  /*вектора Шура -- матрица Z из dhseqr_ */
		  double * Q, 
		  int * LDQ,
		  /*собственные числа*/
		  double * WR, 
		  double * WI,
		  /*размерность инвариантного подпространства*/
		  int * M, 
/*          If JOB = 'E' or 'B', S is a lower bound on the reciprocal
 *          condition number for the selected cluster of eigenvalues.
 *          S cannot underestimate the true reciprocal condition number
 *          by more than a factor of sqrt(N). If M = 0 or N, S = 1.
 *          If JOB = 'N' or 'V', S is not referenced. */
		  double * S, 
/*          If JOB = 'V' or 'B', SEP is the estimated reciprocal
 *          condition number of the specified invariant subspace. If
 *          M = 0 or N, SEP = norm(T).
 *          If JOB = 'N' or 'E', SEP is not referenced. */
		  double * SEP, 
		  
		  double * WORK, 
/*          The dimension of the array WORK.
 *          If JOB = 'N', LWORK >= max(1,N);
 *          if JOB = 'E', LWORK >= max(1,M*(N-M));
 *          if JOB = 'V' or 'B', LWORK >= max(1,2*M*(N-M)).
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA. */
		  int * LWORK, 
		  int * IWORK, 
/*          The dimension of the array IWORK.
 *          If JOB = 'N' or 'E', LIWORK >= 1;
 *          if JOB = 'V' or 'B', LIWORK >= max(1,M*(N-M)).
 *
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates the optimal size of the IWORK array,
 *          returns this value as the first entry of the IWORK array, and
 *          no error message related to LIWORK is issued by XERBLA. */
		  int * LIWORK, 
		  int * INFO );

	  /*балансировка матрицы*/
      void dgebal_( 
/*          Specifies the operations to be performed on A:
 *          = 'N':  none:  simply set ILO = 1, IHI = N, SCALE(I) = 1.0
 *                  for i = 1,...,N;
 *          = 'P':  permute only;
 *          = 'S':  scale only;
 *          = 'B':  both permute and scale. */
		  char * JOB, 
		  int * N, 
		  double * A, 
		  int * LDA, 
		  int * ILO, 
		  int * IHI, 
		  double * SCALE, 
		  int * INFO );

	  /*восстановление матрицы Q после приведения к трех диагональному виду*/
       void dorghr_( int * N, int * ILO, int * IHI, 
		   double * A, int * LDA, double * TAU, 
		   double * WORK, int * LWORK, int * INFO );

      void  dhsein_( 
/*          = 'R': compute right eigenvectors only;
 *          = 'L': compute left eigenvectors only;
 *          = 'B': compute both right and left eigenvectors. */

		  char * SIDE, 

/*          Specifies the source of eigenvalues supplied in (WR,WI):
 *          = 'Q': the eigenvalues were found using DHSEQR; thus, if
 *                 H has zero subdiagonal elements, and so is
 *                 block-triangular, then the j-th eigenvalue can be
 *                 assumed to be an eigenvalue of the block containing
 *                 the j-th row/column.  This property allows DHSEIN to
 *                 perform inverse iteration on just one diagonal block.
 *          = 'N': no assumptions are made on the correspondence
 *                 between eigenvalues and diagonal blocks.  In this
 *                 case, DHSEIN must always perform inverse iteration
 *                 using the whole matrix H. */

		  char * EIGSRC, 
/*          = 'N': no initial vectors are supplied;
 *          = 'U': user-supplied initial vectors are stored in the arrays
 *                 VL and/or VR. */
		  char * INITV, 

/*          Specifies the eigenvectors to be computed. To select the
 *          real eigenvector corresponding to a real eigenvalue WR(j),
 *          SELECT(j) must be set to .TRUE.. To select the complex
 *          eigenvector corresponding to a complex eigenvalue
 *          (WR(j),WI(j)), with complex conjugate (WR(j+1),WI(j+1)),
 *          either SELECT(j) or SELECT(j+1) or both must be set to
 *          .TRUE.; then on exit SELECT(j) is .TRUE. and SELECT(j+1) is
 *          .FALSE.. */
		  int * SELECT, 
		  int * N, 
		  double * H, 
		  int * LDH, 
		  double * WR, double *WI,
		  double * VL, int * LDVL, 
		  /*правые собственные вектора*/
		  double * VR, int * LDVR, 
/*          (input) The number of columns in the arrays VL and/or VR. MM >= M.*/
		  int * MM, 
/* (output) The number of columns in the arrays VL and/or VR required to
 *          store the eigenvectors; each selected real eigenvector
 *          occupies one column and each selected complex eigenvector
 *          occupies two columns. */
		  int *M, 
		  /*DOUBLE PRECISION array, dimension ((N+2)*N)*/
		  double * WORK, 

/*  IFAILL  (output) INTEGER array, dimension (MM)
 *          If SIDE = 'L' or 'B', IFAILL(i) = j > 0 if the left
 *          eigenvector in the i-th column of VL (corresponding to the
 *          eigenvalue w(j)) failed to converge; IFAILL(i) = 0 if the
 *          eigenvector converged satisfactorily. If the i-th and (i+1)th
 *          columns of VL hold a complex eigenvector, then IFAILL(i) and
 *          IFAILL(i+1) are set to the same value.
 *          If SIDE = 'R', IFAILL is not referenced. */

		  double * IFAILL,
/*  IFAILR  (output) INTEGER array, dimension (MM)
 *          If SIDE = 'R' or 'B', IFAILR(i) = j > 0 if the right
 *          eigenvector in the i-th column of VR (corresponding to the
 *          eigenvalue w(j)) failed to converge; IFAILR(i) = 0 if the
 *          eigenvector converged satisfactorily. If the i-th and (i+1)th
 *          columns of VR hold a complex eigenvector, then IFAILR(i) and
 *          IFAILR(i+1) are set to the same value.
 *          If SIDE = 'L', IFAILR is not referenced. */
		  double * IFAILR, 
		  int *INFO );

/*арпак*/
	void dnaupd_(int*ido, char*bmat, int*n, char *which, int*nev, double *tol, double *resid, int *ncv, double *v, int*ldv,
		int *iparam, int *ipntr, double *workd, double *workl, int *lworkl, int *info);

	void    dneupd_ ( int *vec, char * c, int * select, double * d, double * /*d(1,2)*/,
		double *v, int *ldv,
		double * sigmar, double * sigmai, double * workev, char * bmat, int *n,
		char * which, int *nev, double * tol,
		double * resid, int *ncv, double *vv, int * ldvv, int * iparam, int * ipntr, double * workd,
		double * workl, int * lworkl, int *ierr );


	void dsaupd_(int*ido, char*bmat, int*n, char *which, int*nev, double *tol, double *resid, int *ncv, double *v, int*ldv,
		int *iparam, int *ipntr, double *workd, double *workl, int *lworkl, int *info);

	void    dseupd_ ( int *vec, char * c, int * select, double * d,
		double *v, int *ldv,
		double * sigma, char * bmat, int *n,
		char * which, int *nev, double * tol,
		double * resid, int *ncv, double *vv, int * ldvv, int * iparam, int * ipntr, double * workd,
		double * workl, int * lworkl, int *ierr );

#ifdef __cplusplus
} //extern "C"
#endif
#endif
