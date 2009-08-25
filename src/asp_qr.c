/*$Id$*/

/* Copyright (c) 2000, 2002, 2004, 2005, 2007 Alexey Ozeritsky
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
 * Вычисление собственных векторов и значений
 */
/**
 * QR разложение
 * Алгоритм взят из книги К. Ю. Богачёв, Практикум на ЭВМ.
 * Методы решения линейных систем и нахождения собственных значений.
 * Москва 1999.
 * стр 63-75
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "asp_macros.h"
#include "asp_misc.h"
#include "asp_qr.h"
#include "asp_check.h"
#include "asp_gauss.h"

 /**
  * A - матрица, которую вращаем
  * Q - транспонированное произведение матриц элементарного вращения
  * если 0, то не вычисляем
  */
void rotate_general(double * A, double * Q, int n) {
	int i,j=0,k=0;
	double c,s,x,y,xi,xj;
	double r;

	if(Q==0) { /*Q=0, сохранять матрицу Q не нужно*/
		for(j=0;j<n-1;j++) {
			/*от j+2 - почти треугольный вид*/
			for(i=j+1;i<n;i++) {
				x=A[j*n+j]; /*j+1 * n - почти треугольный*/
				y=A[i*n+j];
				r=x*x+y*y;
				r=sqrt(r);
				if(r<1e-14) continue;
				c=x/r;
				s=-y/r;
				for(k=j;k<n;k++) {
					xi=A[j*n+k]; /*j+1 * n - почти треугольный*/
					xj=A[i*n+k];
					A[n*j+k]=xi*c-xj*s; /*j+1 * n	- почти	треугольный*/
					A[n*i+k]=xi*s+xj*c;
				}
			}
		}
		if(A[n*(n-1)+n-1]<0) {
			A[n*(n-1)+n-1]=-A[n*(n-1)+n-1];
		}
	} else {
		memset(Q,0,n*n*sizeof(double));
		for(i=0;i<n;i++) Q[i*n+i]=1.0;

		for(j=0;j<n-1;j++) {
			/*от j+2 - почти треугольный вид*/
			for(i=j+1;i<n;i++) {
				x=A[j*n+j]; /*j+1 * n - почти треугольный*/
				y=A[i*n+j];
				r=x*x+y*y;
				r=sqrt(r);
				if(r<1e-14) continue;
				c=x/r;
				s=-y/r;
				for(k=0;k<n;k++) {
					xi=Q[k*n+j];
					xj=Q[k*n+i];
					Q[k*n+j]=xi*c-xj*s;
					Q[k*n+i]=xi*s+xj*c;
				}

				for(k=j;k<n;k++) {
					xi=A[(j)*n+k]; /*j+1 * n - почти треугольный*/
					xj=A[i*n+k];
					A[n*(j)+k]=xi*c-xj*s; /*j+1 * n - почти	треугольный*/
					A[n*i+k]=xi*s+xj*c;
				}
			}
		}
		if(A[n*(n-1)+n-1]<0) {
			A[n*(n-1)+n-1]=-A[n*(n-1)+n-1];
			for(k=0;k<n;k++)
				Q[k*(n-1)+n-1]=-Q[k*(n-1)+n-1];
		}
	}
}

/*!метод для матриц A и Q размерности n * p
   p столбцов
   n размерность строк
  */
void rotate_general2(double * A, double * Q, int n, int p) {
	int i, j = 0, k = 0;
	double c, s, x, y, xi, xj;
	double r;

	
	memset(Q, 0, n * p * sizeof(double));
	for (i = 0; i < n; i++) Q[i * p + i] = 1.0;

	for (j = 0; j < n - 1; j++) {
		for (i = 0; i < n; i++) {
			x = A[j * p + j];
			y = A[i * p + j];
			r = x * x + y * y;
			r = sqrt(r);
			if (r < 1e-14) continue;
			c =  x / r;
			s = -y / r;
			for (k = 0; k < n; k++) {
				xi = Q[k * p + j];
				xj = Q[k * n + i];
				Q[k * p + j] = xi * c - xj * s;
				Q[k * p + i] = xi * s + xj * c;
			}

			for(k = j; k < n; k++) {
				xi = A[(j) * p + k];
				xj = A[i *   p + k];
				A[p * (j) + k] = xi * c - xj * s;
				A[p * i   + k] = xi * s + xj * c;
			}
		}
	}

	/* ? */
	if (A[n * (p - 1) + n - 1] < 0) {
		A[n * (p - 1) + n - 1] = -A[n * (p - 1) + n - 1];
		for (k = 0; k < n; k++)
			Q[k * (p - 1) + n - 1] = -Q[k * (p - 1) + n - 1];
	}
}

/**
 * Приведение матрицы к почти треугольному виду
 * унитарным подобием методом вращений
 * @param A - матрица
 * @param n - размерность
 */
void rotateu_to_ptriang(double *A, int n) {
	int i,j,k;
	double x,y,r,xi,xj,c,s;
	for(j=0;j<n-1;j++) {
		for(i=j+2;i<n;i++) {
			x=A[(j+1)*n+j];
			y=A[i*n+j];
			r=x*x+y*y;
			r=sqrt(r);
			if(r<1e-14) continue;
			c=x/r;
			s=-y/r;
			for(k=j;k<n;k++) {
				xi=A[(j+1)*n+k];
				xj=A[i*n+k];
				A[n*(j+1)+k]=xi*c-xj*s;
				A[n*i+k]=xi*s+xj*c;
			}
			for(k=0;k<n;k++) {
  				xi=A[k*n+j+1];
				xj=A[k*n+i];
				A[n*k+j+1]=xi*c-xj*s;
				A[n*k+i]=xi*s+xj*c;
			}
		}
	}
}

/**
 * Приведение матрицы к почти треугольному виду
 * унитарным подобием методом вращений
 * @param A - матрица, которую приводим к почти треугольному
 * @param Z - матрица, которую вращаем также
 * @param n - размерность
 */
void rotateu_to_ptriang_z(double *A, double *Z, int n) {
	int i,j,k;
	double x,y,r,xi,xj,c,s;
	for(j=0;j<n-1;j++) {
		for(i=j+2;i<n;i++) {
			x=A[(j+1)*n+j];
			y=A[i*n+j];
			r=x*x+y*y;
			r=sqrt(r);
			if(r<1e-14) continue;
			c=x/r;
			s=-y/r;
			for(k=j;k<n;k++) {
				xi=A[(j+1)*n+k];
				xj=A[i*n+k];
				A[n*(j+1)+k]=xi*c-xj*s;
				A[n*i+k]=xi*s+xj*c;
				Z[n*(j+1)+k]=xi*c-xj*s;
				Z[n*i+k]=xi*s+xj*c;
			}
			for(k=0;k<n;k++) {
  				xi=A[k*n+j+1];
				xj=A[k*n+i];
				A[n*k+j+1]=xi*c-xj*s;
				A[n*k+i]=xi*s+xj*c;
				Z[n*k+j+1]=xi*c-xj*s;
				Z[n*k+i]=xi*s+xj*c;
			}
		}
	}
}

/**
 * Приведение симметричной матрицы к трёхдиагональному виду
 * унитарным подобием методом вращений
 *
 * @param A - матрица
 * @param n - размерность
 */
void rotateu_to_tdiag(double *A,int n) {
	int i,j,k;
	double x,y,r,xi,xj,c,s;
	for(j=0;j<n-1;j++) {
		for(i=j+2;i<n;i++) {
			x=A[(j+1)*n+j];
			y=A[i*n+j];
			r=x*x+y*y;
			r=sqrt(r);
			if(r<1e-14) continue;
			c=x/r;
			s=-y/r;
			for(k=j;k<n;k++) {
				xi=A[(j+1)*n+k];
				xj=A[i*n+k];
				A[n*(j+1)+k]=xi*c-xj*s;
				A[n*i+k]=xi*s+xj*c;
			}
			for(k=j;k<n;k++) {
				xi=A[k*n+j+1];
				xj=A[k*n+i];
				A[n*k+j+1]=xi*c-xj*s;
				A[n*k+i]=xi*s+xj*c;
			}
		}
	}
}

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
void reflectu_to_ptriang(double *a,int n) {
	int j,k=1,u,v;
	double *d=(double*)malloc(n*sizeof(double));
	double l,s,m=0;
	NOMEM(d);
	for(k=1;k<n-1;k++) {
		s=0;
		for(j=k+1;j<n;j++) {
			s+=a[j*n+k-1]*a[j*n+k-1];
		}
		if(s!=0) {
			l=sqrt((double)(a[k*n+k-1]*a[k*n+k-1]+s));
			d[k-1]=a[k*n+k-1]-l;
			for(j=k+1;j<n;j++)
				d[j-1]=a[j*n+k-1];
			l=sqrt(d[k-1]*d[k-1]+s);
			for(j=k-1;j<n-1;j++) {
				d[j]/=l;
			}
			for(j=0;j<k-1;j++) {
				d[j]=0;
			}
			for(u=0;u<n;u++) {
				for(v=k;v<n;v++)
					m+=d[v-1]*a[n*v+u];
				m*=2;
				for(v=k;v<n;v++)
					a[v*n+u]-=m*d[v-1];
				m=0;
			}
			for(u=0;u<n;u++) {
				for(v=k;v<n;v++)
					m+=d[v-1]*a[n*u+v];
				m*=2;
				for(v=k;v<n;v++)
					a[u*n+v]-=m*d[v-1];
				m=0;
			}
		}
	}
	free(d);
}

/**
 * решение проблеммы собственных значений для
 * общего случая
 * @param A - матрица
 * @param Xr - вещественные части собственных значений
 * @param Xi - мнимые части собственных значений
 * @param n - размерность
 * @param eps - точность
 */
void ev_general(double *A,double *Xr,double *Xi,int n,double eps) {
	int i;
	double s;
	double b,c,D,*C,*S;
	int it=0;
	if(n==1) {(*Xr)=A[0];(*Xi)=0;return;}
/* приводим матрицу к почти треугольному виду */
//	reflectu_to_ptriang(A,n);
	rotateu_to_ptriang(A,n);

	C=(double*)malloc(n*sizeof(double)); NOMEM(C);
	S=(double*)malloc(n*sizeof(double)); NOMEM(S)

	memset(Xi,0,n*sizeof(double));
	for(i=n;i>2;i--) {
		while (fabs(A [(i-1)*n+i-2])>eps) {
			s=A[(i-1)*n+i-1]+1./2.*A[(i-1)*n+i-2];
			sdvig(A,n,i,s);   /*сдвигаем*/
			rotate_ptriang_sub(A,C,S,n,i); /* находим A=R=A*Q*/
			RQ_rotate_step(A,C,S,n,i); /*R*Q*/

			sdvig(A,n,i,-s); /*сдвигаем*/
			it++;

			/**
			 * Случай блочной матрицы
			 */
			if(fabs(A[(i-2)*n+i-3])<eps) {
				b=-(A[(i-2)*n+i-2]+A[(i-1)*n+i-1]);
				c=A[(i-2)*n+i-2]*A[(i-1)*n+i-1]-
					A[(i-2)*n+i-1]*A[(i-1)*n+i-2];
				D=b*b-4*c;
				if (D<0) {/*есть комплексные значения*/
					A[(i-2)*n+i-2]=-b/2.0;
					A[(i-1)*n+i-1]=A[(i-2)*n+i-2];
					A[(i-1)*n+i-2]=sqrt(-D)/2.0;
					A[(i-2)*n+i-1]=-A[(i-1)*n+i-2];
					Xi[i-1]=A[(i-1)*n+i-2];
					Xi[i-2]=-A[(i-1)*n+i-2];
				} else {
					D=sqrt(D);
					A[(i-2)*n+i-2]=(-b+D)/2; /*диагонали*/
					A[(i-1)*n+i-1]=(-b-D)/2; /*диагонали*/
					A[(i-1)*n+i-2]=0; /*околодиагональные*/
					A[(i-2)*n+i-1]=0; /*элементы*/
				}
				i-=1;
				break;
			}
		}
	}
	for (i=2; i<n; i++)
		Xr[i]=A[i*n+i];

	/**
	 * Нахождение последних 2-х с.з. путём решения кв. ур-я
	 */
	b = -(A[0*n+0]+A[1*n+1]); c=A[0*n+0]*A[1*n+1]-A[0*n+1]*A[1*n+0];
	D = b*b-4.0*c;

	if (D<0) {
		Xr[0]=-b/2.;
		Xr[1]=Xr[0];
		Xi[0]=sqrt(-D)/2.;
		Xi[1]=-Xi[0];
	} else {
		D=sqrt(D);
		Xr[0]=(-b+D)/2.;
		Xr[1]=(-b-D)/2.;
	}
	free(C);free(S);
}

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
						,double eps) {
	int i,j;
	double s;
	double b,c,D,*C,*S;
	int it=0;
	if(n==1) {(*Xr)=A[0];(*Xi)=0;return;}

	/*!\todo fixme*/
	for(i=0;i<n;i++) {
		for(j=0;j<n;j++) {
			if(i==j) Z[i*n+j]=1;
			else Z[i*n+j]=0;
		}
	}

	/**
	 * приводим матрицу к почти треугольному виду
	 */
	rotateu_to_ptriang_z(A,Z,n);

	C=(double*)malloc(n*sizeof(double)); NOMEM(C);
	S=(double*)malloc(n*sizeof(double)); NOMEM(S)
	memset(Xi,0,n*sizeof(double));

	for(i=n;i>2;i--) {
		while (fabs(A [(i-1)*n+i-2])>eps) {
			s=A[(i-1)*n+i-1]+1./2.*A[(i-1)*n+i-2];
			sdvig(A,n,i,s);   /*сдвигаем*/
			rotate_ptriang_sub_z(A,Z,C,S,n,i); /* находим A=R=A*Q*/
			RQ_rotate_step_z(A,Z,C,S,n,i); /*R*Q*/
			sdvig(A,n,i,-s); /*сдвигаем*/
			it++;
			/**
			 * Случай блочной матрицы
			 */
			if(fabs(A[(i-2)*n+i-3])<eps) {
				b=-(A[(i-2)*n+i-2]+A[(i-1)*n+i-1]);
				c=A[(i-2)*n+i-2]*A[(i-1)*n+i-1]-
					A[(i-2)*n+i-1]*A[(i-1)*n+i-2];
				D=b*b-4*c;
				if (D<0) {/*есть комплексные значения*/
					A[(i-2)*n+i-2]=-b/2.0;
					A[(i-1)*n+i-1]=A[(i-2)*n+i-2];
					A[(i-1)*n+i-2]=sqrt(-D)/2.0;
					A[(i-2)*n+i-1]=-A[(i-1)*n+i-2];
					Xi[i-1]=A[(i-1)*n+i-2];
					Xi[i-2]=-A[(i-1)*n+i-2];
				} else {
					D=sqrt(D);
					A[(i-2)*n+i-2]=(-b+D)/2; /*диагонали*/
					A[(i-1)*n+i-1]=(-b-D)/2; /*диагонали*/
					A[(i-1)*n+i-2]=0; /*околодиагональные*/
					A[(i-2)*n+i-1]=0; /*элементы*/
				}
				i-=1;
				break;
			}
		}
	}
	for (i=2; i<n; i++)
		Xr[i]=A[i*n+i];

	/**
	 * Нахождение последних 2-х с.з. путём решения кв. ур-я
	 */
	b = -(A[0*n+0]+A[1*n+1]); c=A[0*n+0]*A[1*n+1]-A[0*n+1]*A[1*n+0];
	D = b*b-4.0*c;

	if (D<0) {
		Xr[0]=-b/2.;
		Xr[1]=Xr[0];
		Xi[0]=sqrt(-D)/2.;
		Xi[1]=-Xi[0];
	} else {
		D=sqrt(D);
		Xr[0]=(-b-D)/2.;
		Xr[1]=(-b+D)/2.;
	}
	free(C);free(S);
}

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
						, double eps)
{
	int i, j;
	double *Xi = (double*)malloc(n*sizeof(double));/*мнимая часть с.з.*/
	double *Xr = (double*)malloc(n*sizeof(double));/*комплексная часть с.з.*/
	double *AA = (double*)malloc(n*n*sizeof(double));
	NOMEM(Xi); NOMEM(Xr); NOMEM(AA);
	NOTIMPL ;

	memset(Xi,0,n*sizeof(double));

	ev_general(A,Xr,Xi,n,eps);

	for(i = 0; i < n; i++) {
		if(fabs(Xi[i]) > eps) {
			printf("комплексные собственные значения\n");
			exit(1);
		}
	}

	/*надо решить систему с матрицей A-lE*/
	for(i = 0; i < n; i++) {
		for(j = 0; j < n; j++) {
			if(i == j) {
				AA[i*n+j]=A[i*n+j] - Xr[i];
			} else {
				AA[i*n+j]=A[i*n+j];
			}
		}
	}
	/*решаем (A-lE)x=0*/
	//gauss(n,AA,Xi);
	/*!\todo fixme*/
	V = AA;
	/*----------------*/
}

/**
 * решение проблемы собственных значений
 * и функций для общего случая
 * @param A - матрица
 * @param X - вектор собственных значений
 * @param V - базис подпространств
 * @param n - размерность пространства
 * @param eps - точность
 */
void ev_vectors_general_old(double *A, double *X, double *V, int n, double eps) {
	int m, i, j, k;
	double *Xi = (double*)malloc(n*sizeof(double));/*мнимая часть с.з.*/
	double *T  = (double*)malloc(n*n*sizeof(double));
	NOMEM(Xi); NOMEM(T);

	memset(Xi,0,n*sizeof(double));

	ev_general(A,X,Xi,n,eps);

	for (i = 0; i < n; i++) {
		if(fabs(Xi[i]) > eps) {
			printf("комплексные собственные значения\n");
			exit(1);
		}
	}

	printf("A::\n"); printfvmatrix(A,n,8,"%.6e "); printf("\n");
	for (m = 0; m < n; m++) {
		printf("l_m::%lf\n",X[m]);
		/*надо решить систему с матрицей T = A-l_m E*/
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				if (i == j) {
					T[i*n+j]=A[i*n+j] - X[m];
				} else {
					T[i*n+j]=A[i*n+j];
				}
			}
		}
		/**
		 * приводим T к треугольному виду
		 */
		for (j = 0; j < n; j++) {
	    	double a;
		    /*поиск главного значения*/
	       	int l, r = j;
			double max = T[j*n+j];
			double c;
			double temp;
			for (l = j; l < n; l++) {
				c = fabs(T[l*n+j]);
				if(max < c) {
					r = l;
					max  = c;
				}
			}
	        if(fabs(max) < eps) break;
	        /*меняем r'ю и j'ю строки*/

			for(l = j; l < n; l++) {
				temp = T[j*n+l]; T[j*n+l] = T[r*n+l]; T[r*n+l] = temp;
			}

	        a=T[j*n+j];
	        for (i = j + 1; i < n; i++) {
	            double ba=T[i*n+j]/a;

	            for (k = j; k < n; k++)
	                T[i*n+k]-=A[j*n+k]*ba;
	        }
	    }
	    /**
	     * обратный ход гаусса
	     */
//	    for (j = n - 1; j >= 0; j--) {
//	        double a = T[j*n+j];
//	        if (fabs(a)<eps) {
//	        	V[m*n+j] = 1;
//	        	continue;
//	        }
//	        for (i = j - 1; i >= 0; i--) {
//	            double ba = T[i*n+j]/a;
//	            for (k = n-1; k >= j; k--)
//	                T[i*n+k]-=T[j*n+k]*ba;
//	    	}
//    	}

		printf("T %d::\n",m); printfvmatrix(T,n,8,"%.6e "); printf("\n");
	}
	free(Xi);
	free(T);
}

/**
 * LAPACK'based
 * решение проблемы собственных значений для общего случая и
 * нахождение подпространств, отвечающих собственным значениям
 * по модулю больше единицы и меньше единицы
 * память на Ex, Ey выделяется внутри, ибо не знаем размеры!
 * при выделении памяти под матрицу классификации внутри функции
 * надо обязательно выставить исходный указатель в нуль!
 * @param A  - матрица
 * @param X  - собственные значения (если 0, то не выч)
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
void ev_subspace_general_lapack(double *A, double *X, double *V
								, double **lx, double **ly
								, double **Ex, double **Ey
								, int n, int *nx, int *ny
								, double criteria, int absolute)
{
	char jobvl = 'N'; /*не вычислять левые собственные вектора*/
	char jobvr = 'V'; /*вычислять правые собственные вектора  */

	int i,j,k;
	int info;          /* статус возврата */
	int lda   = n;
	int lwork = 4 * n; /* хз что?*/
	int ldvl  = n;     /* не используется */
	int ldvr  = n;     /* не используется */
	int *c    = 0;     /* матрица классификации */

	int v_was_null  = 0; /*флаги для корректного удаления ресурсов*/
	int x_was_null  = 0;
	int ex_was_null = 0;
	int ey_was_null = 0;
	int calc_ex_ey  = 1;
	double *work = (double*)malloc(lwork*sizeof(double)); /*хз что?*/
	double *wr   = X; /*с. в.: вещественная часть*/
	double *wi   = (double*)malloc(n * sizeof(double));/*мнимая часть*/
	double *vr;       /*с. вектора Ax  =lx*/
	double *T    = (double*)malloc(n * n * sizeof(double));/*мат с которой работаем*/
	double *ex   = 0; /* собственные вектора, отвечающие X, Y, если надо то */
	double *ey   = 0; /* потом сохраняем их в Ex и Ey */

	ERR(n, "n must be > 0!"); NOMEM(work); NOMEM(wi); NOMEM(T);
	if (!V) {
		vr = (double*)malloc(n * n * sizeof(double)); NOMEM(vr);
		v_was_null = 1;
	} else {
		vr = V;
	}
	if (!X) {
		wr = (double*)malloc(n * sizeof(double)); NOMEM(wr);
		x_was_null = 1;
	} else {
		wr = X;
	}

	if (nx == 0 || ny == 0) calc_ex_ey = 0;
	if (calc_ex_ey) {
		if (!Ex) { ex_was_null = 1;}
		if (!Ey) { ey_was_null = 1;}
		c = (int*)malloc(n*sizeof(int)); NOMEM(c);
	}

	/**
	 * копирование с транспонированием
	 * dgeev_ фортрановская функция, работает с матрицами
	 * заданными по строкам, более того она изменяет исходную
	 * матрицу
	 */
	matrix_copy_transpose(T, A, n);
	dgeev_( &jobvl, &jobvr, &n, T, &lda,  wr,  wi, 0 /*vl*/,  &ldvl,  vr,
 		&ldvr, work, &lwork, &info);
	ERR((info == 0), "error in lapack's dgeev_ procedure");
	matrix_transpose(vr, n);

	//_fprintfmatrix("vr_lapack.txt", vr, n, n, "%.3le ");
	//_fprintfvector("wr_lapack.txt", wr, n, n, "%.3le ");

	if (!absolute) {
		for (i = 0; i < n; i++) {
			if (fabs(wi[i]) > EPS32) {ERR(0, "Complex eigen values");}
		}
	}

	if (!calc_ex_ey) {
		if (v_was_null) free(vr);
		free(work); free(wi);
		free(T); free(c);
		return;
	}

	if (lx) *lx = (double*)malloc(n * sizeof(double));
	if (ly) *ly = (double*)malloc(n * sizeof(double));

	(*nx) = 0; (*ny) = 0;
	if (absolute) {
		for (i = 0; i < n; i++) {
			double dist = sqrt(wr[i] * wr[i] + wi[i] * wi[i]);
#ifdef _LAPACK_PRINT_VALUES
			printf("%.16le=(%.16le, %.16le) \n", dist, wr[i], wi[i]);
#endif
			if (dist > criteria)    {
				if (lx) (*lx)[*nx] = wr[i];
				c[i] = 1; (*nx)++;
			} /* x */
			else /*fabs(wr[i])< criteria*/ {
				if (ly) (*ly)[*ny] = wr[i];
				c[i] = 0; (*ny)++;
			} /* y */
		}
	} else {
		for (i = 0; i < n; i++) {
			if (wr[i] > criteria)    {
				if (lx) (*lx)[*nx] = wr[i];
				c[i] = 1; (*nx)++;
			} /* x */
			else /*wr[i]< criteria*/ {
				if (ly) (*ly)[*ny] = wr[i];
				c[i] = 0; (*ny)++;
			} /* y */
		}
	}


	/* отделяем часть, отвечающую X
	 * отделение нужно только если считаем матрицу V (её надо перегруппировать)
	 * или если считаем матрицу Ex */
	if (!v_was_null || !ex_was_null) {
		size_t size = (*nx) * n * sizeof(double); MEMSIZE(size);
		ex = (double*)malloc(size); NOMEM(ex);
		for (i = 0, k = 0; i < n; i++) {
			if (c[i] == 1) {
				for (j = 0; j < n; j++) {
					ex[j*(*nx)+k] = vr[j*n+i];
				}
				k++;
			}
		}
	}
	/* отделяем часть, отвечающую Y */
	if (!v_was_null || !ex_was_null) {
		size_t size = (*ny) * n * sizeof(double); MEMSIZE(size);
		ey = (double*)malloc(size); NOMEM(ey);
		for (i = 0, k = 0; i < n; i++) {
			if (c[i] == 0) {
				for (j = 0; j < n; j++) {
					ey[j*(*ny)+k] = vr[j*n+i];
				}
				k++;
			}
		}
	}

	/* выставляем собственные вектора и собственные значения
	 * в порядке (Ex, Ey) */
	if (!v_was_null) {
		for (i = 0; i < *nx; i++) {
			for (j = 0; j < n; j++) {
				V[j * n + i] = ex[j * (*nx) + i];
			}
		}

		for (i = 0; i < *ny; i++) {
			for (j = 0; j < n; j++) {
				V[j * n + i + (*nx)] = ey[j * (*ny) + i];
			}
		}
		/* временно скопируем wr в wi в упорядоченном состоянии
		 * x компоненты */
		for (i = 0, k = 0; i < n; i++) {
			if (c[i] == 1) {
				wi[k] = wr[i];
				k++;
			}
		}
		/* y компоненты */
		for (i = 0, k = 0; i < n; i++) {
			if (c[i] == 0) {
				wi[k + (*nx)] = wr[i];
				k++;
			}
		}
		//!\todo можно обойтись без копирования, присвоив X = wi, и удалив wr
		memcpy(wr, wi, n * sizeof(double));
	}

	if (!ex_was_null) *Ex = ex;
	if (!ey_was_null) *Ey = ey;

	free(work); free(wi); free(T); free(c);
	if (ex_was_null && !v_was_null) free(ex);
	if (ey_was_null && !v_was_null) free(ey);
	if (v_was_null) free(V);
	if (x_was_null) free(X);
#ifdef _TEST_LAPACK
	if (*lx && ex) {
		double * e = malloc(n * sizeof(double));
		double *e1 = malloc(n * sizeof(double));
		printf("testing lapack, A X = lx X\n");
		for (i = 0; i < *nx; i++) {
			printf("lx[%d]=%.16lf norm=", i, (*lx)[i]);
			for (j = 0; j < n; j++) {
				e[j] = ex[j * (*nx) + i];
			}
			matrix_mult_vector(e1, A, e, n);
			for (j = 0; j < n; j++) {
				e[j] *= (*lx)[i];
			}
			printf("%.16lf\n", dist(e1, e, n));
		}
		free(e); free(e1);
	}
#endif
}

#include "asp_projection.h"

double ev_subspace_angle(const double * ex, const double * ext, int n, int nx)
{
	int i, j, k;
	double * y   = malloc(nx * nx * sizeof(double));
	double * s   = malloc(nx * sizeof(double));
	double * Y1  = malloc(nx * n * sizeof(double));
	double * Y2  = malloc(nx * n * sizeof(double));
	double angle = 0;
	double fang;

	memcpy(Y1, ex,  nx * n * sizeof(double));
	memcpy(Y2, ext, nx * n * sizeof(double));

	printf("%.16lf\n", dist1(Y1, Y2, nx * n));

	ortogonalize_cols_gsch(Y1, n, nx);
	ortogonalize_cols_gsch(Y2, n, nx);

#ifdef _DEBUG
	check_vectors_ortogonalization(Y1, n, nx);
	check_vectors_ortogonalization(Y2, n, nx);
#endif

	for (i = 0; i < nx; ++i) {
		for (j = 0; j < nx; ++j) {
			y[i * nx + j] = 0;
			for (k = 0; k < n; ++k) {
				y[i * nx + j] += Y2[k * nx + i] * Y1[k * nx + j];
			}
		}
	}

	ev_singular_svd_lapack(y, s, nx);

	angle = s[0];
	/*for (k = 1; k < nx; ++k) {
		if (s[k] > angle) angle = s[k];
	}*/

	fang = fabs(angle);
	if (fang < 1) {
		angle = acos(angle);
	} else if (fabs(fang - 1) < 1e-12) {
		angle = 0;
	} else {
		WARN(1, "cos > 1");
	}
	free(y); free(s);
	return angle;
}

void ev_singular_svd_lapack(double * A, double *lx, int n)
{
	char jobu = 'N';  /*не вычилять вектора*/
	char jobv = 'N';  /*не вычилять вектора*/
	int M = n, N = n; /*число строк и столбцов матрицы*/
	int lda = n;
	double * U  = 0;  /*не используется*/
	int ldu     = n;
	double * VT = 0;  /*не используется*/
	int ldvt    = n;
	int lwork   = 10 * n * n;
	double * work = malloc(lwork * sizeof(double));
	int info      = 0;

	dgesvd_(&jobu, &jobv, &M, &N, A, &lda, lx, 
		U, &ldu, VT, &ldvt, 
		work, &lwork, &info);

	ERR(!info, "dgesdd_ error");
	free(work);
}

void ev_subspace_svd_lapack(double *A, double **lx 
							, double **Ex, double **Ext
							, int n, int *nx 
							, double criteria)
{
	char jobz = 'A';
	int M = n, N = n;
	int lda = n;
	double * W = malloc(n * sizeof(double));
	double * U = malloc(n * n * sizeof(double));
	double *VT = malloc(n * n * sizeof(double));
	int ldu    = n, ldvt = n;
	int lwork  = 20 * n * n;
	double * work = malloc(lwork * sizeof(double));
	int * iwork   = malloc(8 * n * sizeof(int));
	int info      = 0;
	int i, j;

	dgesdd_(&jobz, &M, &N, A, &lda, W, U, &ldu, VT, &ldvt, 
		work, &lwork, iwork, &info);

	*nx = 0;
	for (i = 0; i < n; ++i, *nx++) {
		printf("W[%d]=%.16le\n", i, W[i]);
		if (W[i] < criteria)
			break;
	}

	*Ex = malloc(*nx * n * sizeof(double));
	*Ext= malloc(*nx * n * sizeof(double));

	printf("nx=%d\n", *nx);
	for (i = 0; i < *nx; ++i) {
		for (j = 0; j < n; ++j) {
			(*Ex) [j * *nx + i] = U [i * n   + j];
			(*Ext)[j * *nx + i] = VT[j * *nx + i];
		}
	}

	free(W); free(U); free(VT); free(iwork);
}

int check_dnaupd_status(int info)
{
	switch (info) {
	case 0:
//          =  0: Normal exit.
		break;
	case 1:
		printf("1: Maximum number of iterations taken.\n");
		printf("All possible eigenvalues of OP has been found. IPARAM(5)\n");
		printf("returns the number of wanted converged Ritz values.\n");
		break;
	case 2:
		printf("2: No longer an informational error. Deprecated starting\n");
		printf("with release 2 of ARPACK.\n");
		break;
	case 3:
		printf("3: No shifts could be applied during a cycle of the\n");
		printf("Implicitly restarted Arnoldi iteration. One possibility\n");
		printf("is to increase the size of NCV relative to NEV.\n");
	case -1:
		printf("-1: N must be positive.\n");
		break;
	case -2:
		printf("-2: NEV must be positive.\n");
		break;
	case -3:
		printf("-3: NCV-NEV >= 2 and less than or equal to N.\n");
		break;
	case -4:
		printf("-4: The maximum number of Arnoldi update iteration\n");
		printf("must be greater than zero.\n");
		break;
	case -5:
		printf("5: WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'\n");
		break;
	case -6:
		printf("6: BMAT must be one of 'I' or 'G'.\n");
		break;
	case -7:
		printf("-7: Length of private work array is not sufficient.\n");
		break;
	case -8:
		printf("-8: Error return from LAPACK eigenvalue calculation;\n");
		break;
	case -9:
		printf("-9: Starting vector is zero.\n");
		break;
	case -10:
		printf("-10: IPARAM(7) must be 1,2,3,4.\n");
		break;
	case -11:
		printf("-11: IPARAM(7) = 1 and BMAT = 'G' are incompatable.\n");
		break;
	case -12:
		printf("-12: IPARAM(1) must be equal to 0 or 1.\n");
		break;
	case -9999:
		printf("-9999: Could not build an Arnoldi factorization.\n");
		printf("IPARAM(5) returns the size of the current Arnoldi\n");
		printf("factorization.\n");
		break;
	default:
		printf("unknown error\n");
		return -100000;
		break;
	}
	return info;
}

int check_dneupd_status(int info)
{
	switch(info) {
	case 0:
//c          =  0: Normal exit.
			break;
	case 1:
		printf("1: The Schur form computed by LAPACK routine dlahqr\n");
		printf("could not be reordered by LAPACK routine dtrsen .\n");
		printf("Re-enter subroutine dneupd  with IPARAM(5)=NCV and \n");
		printf("increase the size of the arrays DR and DI to have \n");
		printf("dimension at least dimension NCV and allocate at least NCV \n");
		printf("columns for Z. NOTE: Not necessary if Z and V share \n");
		printf("the same space. Please notify the authors if this error\n");
		printf("occurs.\n");
		break;
	case -1:
		printf("-1: N must be positive.\n");
		break;
	case -2:
		printf("= -2: NEV must be positive.\n");
		break;
	case -3:
		printf("-3: NCV-NEV >= 2 and less than or equal to N.\n");
		break;
	case -5:
		printf("-5: WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'\n");
		break;
	case -6:
		printf("-6: BMAT must be one of 'I' or 'G'.\n");
		break;
	case -7:
		printf("-7: Length of private work WORKL array is not sufficient.\n");
		break;
	case -8:
		printf("-8: Error return from calculation of a real Schur form.\n");
		printf("Informational error from LAPACK routine dlahqr .\n");
		break;
	case -9:
		printf("-9: Error return from calculation of eigenvectors.\n");
		printf("Informational error from LAPACK routine dtrevc .\n");
		break;
	case -10:
		printf("-10: IPARAM(7) must be 1,2,3,4.\n");
		break;
	case -11:
		printf("-11: IPARAM(7) = 1 and BMAT = 'G' are incompatible.\n");
		break;
	case -12:
		printf("-12: HOWMNY = 'S' not yet implemented\n");
		break;
	case -13:
		printf("-13: HOWMNY must be one of 'A' or 'P' if RVEC = .true.\n");
		break;
	case -14:
		printf("-14: DNAUPD  did not find any eigenvalues to sufficient\n");
		printf("accuracy.\n");
		break;
	case -15:
		printf("-15: DNEUPD got a different count of the number of converged\n");
		printf("Ritz values than DNAUPD got.  This indicates the user\n");
		printf("probably made an error in passing data from DNAUPD to\n");
		printf("DNEUPD or that the data was modified before entering\n");
		printf("DNEUPD\n");
		break;
	default:
		printf("unknown error\n");
		return -10000;
	}
	return info;
}

/*!тоже самое что и ev_subspace_general_lapack но на базе arpack'а*/
void ev_subspace_general_arpack(double *A, double *X, double *V
								, double **lx, double **ly
								, double **Ex, double **Ey
								, int n, int *nx, int *ny
								, double criteria, int absolute)
{
	int N     = n + 2;
	int i,j,k;
	int info  = 0;          /* статус возврата */
	int maxn  = N;
	int maxncv= N;
    int  nev  = n;/*Number of eigenvalues of OP to be computed. 0 < NEV < N-1.*/
    int  ncv  = N; /*  Number of columns of the matrix V. NCV must satisfy the two
                        inequalities 2 <= NCV-NEV and NCV <= N.*/

	double  tol    = 0; //точность 0 - машинная
	int lworkl= 3 * ncv * ncv + 6 * ncv; /* хз что?*/
	int ldv   = maxn;
	int *c    = 0;     /* матрица классификации */

	int v_was_null  = 0; /*флаги для корректного удаления ресурсов*/
	int x_was_null  = 0;
	int ex_was_null = 0;
	int ey_was_null = 0;
	int calc_ex_ey  = 1;
	double * resid  = (double*)malloc(maxn * sizeof(double));
	double * v      = (double*)malloc(ldv * maxncv * sizeof(double));
	double * workd  = (double*)malloc(3*maxn * sizeof(double));
	double * workev = (double*)malloc(3*maxncv*sizeof(double));
	double * workl  = (double*)malloc((3*maxncv*maxncv+6*maxncv)*sizeof(double));

	double *wr   = X; /*с. в.: вещественная часть*/
	double *wi   = (double*)malloc(n * sizeof(double));/*мнимая часть*/
	double *vr;       /*с. вектора Ax  =lx*/
	double *ex   = 0; /* собственные вектора, отвечающие X, Y, если надо то */
	double *ey   = 0; /* потом сохраняем их в Ex и Ey */
	/* BMAT = 'I' -> standard eigenvalue problem A*x = lambda*x*/
    /* BMAT = 'G' -> generalized eigenvalue problem A*x = lambda*B*x*/
	char   bmat[] ="I";
/* 'LM' -> want the NEV eigenvalues of largest magnitude. */
/* 'SM' -> want the NEV eigenvalues of smallest magnitude.*/
/* 'LR' -> want the NEV eigenvalues of largest real part. */
/* 'SR' -> want the NEV eigenvalues of smallest real part.*/
/* 'LI' -> want the NEV eigenvalues of largest imaginary part. */
/* 'SI' -> want the NEV eigenvalues of smallest imaginary part.*/
	char which[] = "LM";
	int  iparam[11], ipntr[14];
    int  ishfts = 1;
	int  maxitr = 300;
	int  mode   = 1;
	int  ido    = 0;

	if (!V) {
		vr = (double*)malloc(n * n * sizeof(double)); NOMEM(vr);
		v_was_null = 1;
	} else {
		vr = V;
	}
	if (!X) {
		wr = (double*)malloc(n * sizeof(double)); NOMEM(wr);
		x_was_null = 1;
	} else {
		wr = X;
	}

	if (nx == 0 || ny == 0) calc_ex_ey = 0;
	if (calc_ex_ey) {
		if (!Ex) { ex_was_null = 1;}
		if (!Ey) { ey_was_null = 1;}
		c = (int*)malloc(n*sizeof(int)); NOMEM(c);
	}

	//в фортране нумерация массива с 1
	iparam[1-1] = ishfts;
	iparam[3-1] = maxitr;
	iparam[7-1] = mode;

	do {
		dnaupd_ ( &ido, bmat, &N, which, &nev, &tol, resid,
			&ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl,
			&info );

		ERR((info == 0), "error in arpack's dnaupd_ procedure");
		if (ido == -1 || ido == 1) {
			int addr2 = ipntr[2-1]-1;
			int addr1 = ipntr[1-1]-1;
			double *w1 = &workd[addr2]; //ответ
			double *v1 = &workd[addr1]; //начальное условие

			memset(w1, 0, n * sizeof(double));
			for(i = 0; i < n; i++) {
				for(j = 0; j < n; j++) {
					w1[i] += A[i * n + j] * v1[j];
				}
			}

			w1[n]     =  1.e-15*1.*v1[n];
			w1[n + 1] =  1.e-15*1.*v1[n + 1];
		}
	} while (ido == -1 || ido == 1);

	{
		char c[] = "A";
		int rvec   = 1;
		int *select = malloc(maxncv * sizeof(int));
		double sigmar, sigmai;
		int ierr;

		dneupd_ ( &rvec, c, select, wr, wi, v, &ldv,
			&sigmar, &sigmai, workev, bmat, &N, which, &nev, &tol,
			resid, &ncv, v, &ldv, iparam, ipntr, workd, workl,
			&lworkl, &ierr );
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				vr[i * n + j] = v[j * N + i];
			}
		}
		free(select);
	}

	//_fprintfmatrix("vr_arpack.txt", vr, n, n, "%.3le ");
	//_fprintfvector("wr_arpack.txt", wr, n, n, "%.3le ");
	if (!absolute) {
		for (i = 0; i < n; i++) {
			if (fabs(wi[i]) > EPS32) {ERR(0, "Complex eigen values");}
		}
	}

	if (!calc_ex_ey) {
		if (v_was_null) free(vr);
		free(resid); free(v); free(workd);
		free(workev); free(workl);
		free(wi); free(c);
		return;
	}

	if (lx) *lx = (double*)malloc(n * sizeof(double));
	if (ly) *ly = (double*)malloc(n * sizeof(double));

	(*nx) = 0; (*ny) = 0;
	if (absolute) {
		for (i = 0; i < n; i++) {
			double dist = sqrt(wr[i] * wr[i] + wi[i] * wi[i]);
			if (dist > criteria)    {
				if (lx) (*lx)[*nx] = wr[i];
				c[i] = 1; (*nx)++;
			} /* x */
			else /*fabs(wr[i])< criteria*/ {
				if (ly) (*ly)[*ny] = wr[i];
				c[i] = 0; (*ny)++;
			} /* y */
		}
	} else {
		for (i = 0; i < n; i++) {
			if (wr[i] > criteria)    {
				if (lx) (*lx)[*nx] = wr[i];
				c[i] = 1; (*nx)++;
			} /* x */
			else /*wr[i]< criteria*/ {
				if (ly) (*ly)[*ny] = wr[i];
				c[i] = 0; (*ny)++;
			} /* y */
		}
	}

	/* отделяем часть, отвечающую X
	 * отделение нужно только если считаем матрицу V (её надо перегруппировать)
	 * или если считаем матрицу Ex */
	if (!v_was_null || !ex_was_null) {
		size_t size = (*nx) * n * sizeof(double); MEMSIZE(size);
		ex = (double*)malloc(size); NOMEM(ex);
		for (i = 0, k = 0; i < n; i++) {
			if (c[i] == 1) {
				for (j = 0; j < n; j++) {
					ex[j*(*nx)+k] = vr[j*n+i];
				}
				k++;
			}
		}
	}
	/* отделяем часть, отвечающую Y */
	if (!v_was_null || !ex_was_null) {
		size_t size = (*ny) * n * sizeof(double); MEMSIZE(size);
		ey = (double*)malloc(size); NOMEM(ey);
		for (i = 0, k = 0; i < n; i++) {
			if (c[i] == 0) {
				for (j = 0; j < n; j++) {
					ey[j*(*ny)+k] = vr[j*n+i];
				}
				k++;
			}
		}
	}

	/* выставляем собственные вектора и собственные значения
	 * в порядке (Ex, Ey) */
	if (!v_was_null) {
		for (i = 0; i < *nx; i++) {
			for (j = 0; j < n; j++) {
				V[j * n + i] = ex[j * (*nx) + i];
			}
		}

		for (i = 0; i < *ny; i++) {
			for (j = 0; j < n; j++) {
				V[j * n + i + (*nx)] = ey[j * (*ny) + i];
			}
		}
		/* временно скопируем wr в wi в упорядоченном состоянии
		 * x компоненты */
		for (i = 0, k = 0; i < n; i++) {
			if (c[i] == 1) {
				wi[k] = wr[i];
				k++;
			}
		}
		/* y компоненты */
		for (i = 0, k = 0; i < n; i++) {
			if (c[i] == 0) {
				wi[k + (*nx)] = wr[i];
				k++;
			}
		}
		//!\todo можно обойтись без копирования, присвоив X = wi, и удалив wr
		memcpy(wr, wi, n * sizeof(double));
	}

	if (!ex_was_null) *Ex = ex;
	if (!ey_was_null) *Ey = ey;

	free(resid); free(v); free(workd);
	free(workev); free(workl);
	free(wi); free(c);
	if (ex_was_null && !v_was_null) free(ex);
	if (ey_was_null && !v_was_null) free(ey);
	if (v_was_null) free(V);
	if (x_was_null) free(X);
}

/*! вычисляет базис пространства X
\param A  - матрица линеаризации
\param Ex - базис X (память выделяется malloc'ом)
\param n  - размерность A
\param nx - предположительная размерность X (может увеличится
            за счет кратных значений)
*/
void ev_x_subspace_general_arnoldi(double *A, double **Ex, int n,
								  int *nx)
{
	int i, j, k;
	int info  = 0;          /* статус возврата */
    int  nev  =*nx;/*Number of eigenvalues of OP to be computed. 0 < NEV < N-1.*/
    int ncv    = ((2 * nev + 1) < n)?
		(2 * nev + 1):n; /*  Number of columns of the matrix V. NCV must satisfy the two
                        inequalities 2 <= NCV-NEV and NCV <= N.*/

	double  tol    = 0; //точность 0 - машинная
	int lworkl= 3 * ncv * ncv + 6 * ncv; /* хз что?*/
	int ldv   = n;
	int *c    = (int*)malloc(ncv * sizeof(int)); /* матрица классификации */

	double * resid  = (double*)malloc(n * sizeof(double));
	double * v      = (double*)malloc(ldv * ncv * sizeof(double));
	double * workd  = (double*)malloc(3 * n * sizeof(double));
	double * workev = (double*)malloc(3 * ncv * sizeof(double));
	double * workl  = (double*)malloc(lworkl * sizeof(double));

	double *wr   = (double*)malloc(n * sizeof(double));/*с. в.: вещественная часть*/
	double *wi   = (double*)malloc(n * sizeof(double));/*мнимая часть*/
	double *ex   = 0; /* собственные вектора, отвечающие X, Y, если надо то */
	/* BMAT = 'I' -> standard eigenvalue problem A*x = lambda*x*/
    /* BMAT = 'G' -> generalized eigenvalue problem A*x = lambda*B*x*/
	char   bmat[] ="I";
/* 'LM' -> want the NEV eigenvalues of largest magnitude. */
/* 'SM' -> want the NEV eigenvalues of smallest magnitude.*/
/* 'LR' -> want the NEV eigenvalues of largest real part. */
/* 'SR' -> want the NEV eigenvalues of smallest real part.*/
/* 'LI' -> want the NEV eigenvalues of largest imaginary part. */
/* 'SI' -> want the NEV eigenvalues of smallest imaginary part.*/
	char which[] = "LM";
	int  iparam[11], ipntr[14];
    int  ishfts = 1;
	int  maxitr = 300;
	int  mode   = 1;
	int  ido    = 0;

	//в фортране нумерация массива с 1
	iparam[1-1] = ishfts;
	iparam[3-1] = maxitr;
	iparam[7-1] = mode;

	do {
		dnaupd_ ( &ido, bmat, &n, which, &nev, &tol, resid,
			&ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl,
			&info );

		ERR((check_dnaupd_status(info) >= 0), "error in arpack's dnaupd_ procedure");
		if (ido == -1 || ido == 1) {
			int addr2 = ipntr[2-1]-1;
			int addr1 = ipntr[1-1]-1;
			double *w1 = &workd[addr2]; //ответ
			double *v1 = &workd[addr1]; //начальное условие

			memset(w1, 0, n * sizeof(double));
			for(i = 0; i < n; i++) {
				for(j = 0; j < n; j++) {
					w1[i] += A[i * n + j] * v1[j];
				}
			}
		}
	} while (ido == -1 || ido == 1);

	{
		char c[] = "A";
		int rvec   = 1;
		int *select = malloc(ncv * sizeof(int));
		double sigmar, sigmai;
		int ierr;

		dneupd_ ( &rvec, c, select, wr, wi, v, &ldv,
			&sigmar, &sigmai, workev, bmat, &n, which, &nev, &tol,
			resid, &ncv, v, &ldv, iparam, ipntr, workd, workl,
			&lworkl, &ierr );
		ERR((check_dneupd_status(ierr) >= 0), "error in arpack's dneupd_ procedure\n");
		free(select);
	}

	*nx = 0;
	for (i = 0; i < nev; i++) {
		double dist = sqrt(wr[i] * wr[i] + wi[i] * wi[i]);
		if (dist > 1)    {c[i] = 1; (*nx)++;} /* x */
		else /*dist< 1*/ {c[i] = 0;} /* y */
	}

	free(wr); free(wi);
	ex = (double*)malloc((*nx) * n * sizeof(double));
	for (j = 0, i = 0; i < nev; i++) {
		if (c[i] == 1) {
			for (k = 0; k < n; k++) {
				ex[k * (*nx) + j] = v[i * ldv + k];
			}
			j++;
		}
	}

	*Ex = ex;
	free(v); free(workd); free(workev); free(workl); free(resid);
	free(c);
}

void ev_x_subspace_general_lanczos(double *A, double **Ex, int n,
								  int *nx)
{
	int i, j, k;
	int info  = 0;          /* статус возврата */
    int  nev  =*nx;/*Number of eigenvalues of OP to be computed. 0 < NEV < N-1.*/
    int ncv    = ((2 * nev + 1) < n)?
		(2 * nev + 1):n; /*  Number of columns of the matrix V. NCV must satisfy the two
                        inequalities 2 <= NCV-NEV and NCV <= N.*/

	double  tol    = 0; //точность 0 - машинная
	int lworkl= 3 * ncv * ncv + 6 * ncv; /* хз что?*/
	int ldv   = n;
	int *c    = (int*)malloc(ncv * sizeof(int)); /* матрица классификации */

	double * resid  = (double*)malloc(n * sizeof(double));
	double * v      = (double*)malloc(ldv * ncv * sizeof(double));
	double * workd  = (double*)malloc(3 * n * sizeof(double));
	double * workl  = (double*)malloc(lworkl * sizeof(double));

	double *wr   = (double*)malloc(n * sizeof(double));/*с. в.: вещественная часть*/
	double *ex   = 0; /* собственные вектора, отвечающие X, Y, если надо то */
	/* BMAT = 'I' -> standard eigenvalue problem A*x = lambda*x*/
    /* BMAT = 'G' -> generalized eigenvalue problem A*x = lambda*B*x*/
	char   bmat[] ="I";
/* 'LM' -> want the NEV eigenvalues of largest magnitude. */
/* 'SM' -> want the NEV eigenvalues of smallest magnitude.*/
/* 'LR' -> want the NEV eigenvalues of largest real part. */
/* 'SR' -> want the NEV eigenvalues of smallest real part.*/
/* 'LI' -> want the NEV eigenvalues of largest imaginary part. */
/* 'SI' -> want the NEV eigenvalues of smallest imaginary part.*/
	char which[] = "LM";
	int  iparam[11], ipntr[14];
    int  ishfts = 1;
	int  maxitr = 300;
	int  mode   = 1;
	int  ido    = 0;

	//в фортране нумерация массива с 1
	iparam[1-1] = ishfts;
	iparam[3-1] = maxitr;
	iparam[7-1] = mode;

	do {
		dsaupd_ ( &ido, bmat, &n, which, &nev, &tol, resid,
			&ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl,
			&info );

		ERR((check_dnaupd_status(info) >= 0), "error in arpack's dnaupd_ procedure");
		if (ido == -1 || ido == 1) {
			int addr2 = ipntr[2-1]-1;
			int addr1 = ipntr[1-1]-1;
			double *w1 = &workd[addr2]; //ответ
			double *v1 = &workd[addr1]; //начальное условие

			memset(w1, 0, n * sizeof(double));
			for(i = 0; i < n; i++) {
				for(j = 0; j < n; j++) {
					w1[i] += A[i * n + j] * v1[j];
				}
			}
		}
	} while (ido == -1 || ido == 1);

	{
		char c[] = "A";
		int rvec   = 1;
		int *select = malloc(ncv * sizeof(int));
		double sigma;
		int ierr;

		dseupd_ ( &rvec, c, select, wr, v, &ldv,
			&sigma, bmat, &n, which, &nev, &tol,
			resid, &ncv, v, &ldv, iparam, ipntr, workd, workl,
			&lworkl, &ierr );
		ERR((check_dneupd_status(ierr) >= 0), "error in arpack's dneupd_ procedure\n");
		free(select);
	}

	*nx = 0;
	for (i = 0; i < nev; i++) {
		double dist = fabs(wr[i]);
		if (dist > 1)    {c[i] = 1; (*nx)++;} /* x */
		else /*dist< 1*/ {c[i] = 0;} /* y */
	}

	free(wr);
	ex = (double*)malloc((*nx) * n * sizeof(double));
	for (j = 0, i = 0; i < nev; i++) {
		if (c[i] == 1) {
			for (k = 0; k < n; k++) {
				ex[k * (*nx) + j] = v[i * ldv + k];
			}
			j++;
		}
	}

	*Ex = ex;
	free(v); free(workd); free(workl); free(resid);
	free(c);
}

/**
 * решение проблеммы собственных значений для
 * симметричной матрицы
 * @param A - матрица
 * @param X - собственные значения
 * @param n - размерность
 * @param eps - точность
 */
void ev_sym(double*A,double*X,int n,double eps) {
	int i;
	double s;
	double b,c,D,*X2;
	int it=0;
	if(n==1) {(*X)=A[0];return;}
	/*приводим матрицу к трёх диагогальному виду */
	rotateu_to_tdiag(A,n);

	X2=(double*)malloc(n*sizeof(double));
	for (i = n; i > 2; i--) {
		while (fabs(A[(i-1)*n+i-2])>eps) {
			s=A[(i-1)*n+i-1]-1./2.*A[(i-1)*n+i-2];
			sdvig(A,n,i,s);   /* сдвигаем */
			reflect_tdiag_sub(A,X,X2,n,i); /* отражаем */
			RQ_reflect_step(A,X,X2,n,i); /* R*Q */
			sdvig(A,n,i,-s); /* сдвигаем */
			it++;
			/**
			 * Случай блочной матрицы
			 */
			if(fabs(A[(i-2)*n+i-3])<eps) {
				b=-(A[(i-2)*n+i-2]+A[(i-1)*n+i-1]);
				c=A[(i-2)*n+i-2]*A[(i-1)*n+i-1]-
					A[(i-2)*n+i-1]*A[(i-1)*n+i-2];
				D=b*b-4*c;
				if (D<0){
					printf ("комплексные значения 1");
					exit(1);
				}
				else  D=sqrt(D);
				A[(i-2)*n+i-2]=(-b+D)/2;
				A[(i-1)*n+i-1]=(-b-D)/2;
				A[(i-1)*n+i-2]=0;
				A[(i-2)*n+i-1]=0;
				i-=1;
				break;
			}
		}
	}
	for (i=2; i<n; i++) X[i]=A[i*n+i];

	/**
	 * Нахождение последних 2-х с.з. путём решения кв. ур-я
	 */
	b = -(A[0]+A[n+1]); c=A[0]*A[n+1]-A[1]*A[n];
	D = b*b-4*c;

	if (D<0){printf ("комплексные значения 2");exit(1);}
	else  D=sqrt(D); X[0]=(-b+D)/2; X[1]=(-b-D)/2;
	free(X2);
}

/**
 * метод отражений для трёхдиагональной сим матрицы. отражение подматрицы
 * @param A - матрица
 * @param tmp - вектор для хранения
 * @param tmp2 - вектор для хранения
 * @param n - размерность
 * @param m - размерность подматрицы
 */
void reflect_tdiag_sub(double*A, double*tmp, double*tmp2, int n, int m) {
	int j,k;
	double norm,norm2;
	double x1,x2;
	double yx;
	for (j = 0; j < m - 1; j++) {
		x1 = A[j * n + j]; x2 = A[(j + 1) * n + j];
		if (fabs(x2)<EPS32) {
			x1 = 0; x2 = 0;
		} else {
			norm  = A[j * n + j] * A[j * n + j] + A[(j + 1) * n + j] * A[(j + 1) * n + j];
			norm  = sqrt(norm);
			x1    = (A[j * n + j] - norm);
			norm2 = x1 * x1 + A[(j + 1) * n + j] * A[(j + 1) * n + j];
			norm2 = sqrt(norm2);
			x1    = x1 / norm2;
			x2    = A[(j + 1) * n + j] / norm2;

			for (k = j; k < n && k < j + 3; k++) {
				yx            = x1 * A[j * n + k] + x2 * A[(j + 1) * n + k];
				A[j*n+k]     -= 2 * x1 * yx;
				A[(j+1)*n+k] -= 2 * x2 * yx;
			}
		}
		tmp2[j] = x1;
		tmp [j] = x2;
	}
}

void sdvig(double*A, int n, int m, double s) {
	int j;
	for (j = 0; j < m; j++) {
		A[j * n + j] -= s;
	}
}

/**
 * метод вращений для почти треугольной матрицы. Вращение подматрицы
 * @param A - матрица
 * @param C - вектор для хранения косинусов матрицы вращений Q
 * @param S - вектор для хранения синусов
 * @param n - размерность
 * @param m - размерность подматрицы
 */
void rotate_ptriang_sub(double *A, double *C, double *S, int n, int m) {
	int j, k;
	double c, s, x, y, r, xi, xj;
	for (j = 0; j < m - 1; j++) {
		x = A[(j) * n + j];
		y = A[(j + 1) * n + j];
		r = x * x + y * y;
		r = sqrt(r);
		if (r < EPS32) continue;
		c =  x / r;
		s = -y / r;
		for (k = j; k < m; k++) {
			xi           = A[(j) * n + k];
			xj           = A[(j + 1) * n + k];
			A[n*(j)+k]   = xi * c - xj * s;
			A[n*(j+1)+k] = xi * s + xj * c;
		}
		C[j] = c;
		S[j] = s;
	}
}

void rotate_ptriang_sub_z(double *A,double *Z, double *C,double *S
							,int n,int m) {
	int j,k;
	double c,s,x,y,r,xi,xj;
	for(j=0;j<m-1;j++) {
		x=A[(j)*n+j];
		y=A[(j+1)*n+j];
		r=x*x+y*y;
		r=sqrt(r);
		if(r<1e-14) continue;
		c=x/r;
		s=-y/r;
		for(k=j;k<m;k++) {
			xi=A[(j)*n+k];
			xj=A[(j+1)*n+k];
			A[n*(j)+k]=xi*c-xj*s;
			A[n*(j+1)+k]=xi*s+xj*c;
			Z[n*(j)+k]=xi*c-xj*s;
			Z[n*(j+1)+k]=xi*s+xj*c;
		}
		C[j]=c;
		S[j]=s;
	}
}

/**
 * нахождение матрицы A^{k+1}=RQ в случае когда, для подматрицы
 * @param Q задана векторами косинусов и синусов
 * @param A - матрица
 * @param C - вектор для хранения косинусов матрицы вращений Q
 * @param S - вектор для хранения синусов
 * @param n - размерность
 * @param m - размерность подматрицы
 */
void RQ_rotate_step(double *A, double *C, double *S, int n, int m) {
	int j, k;
	double xi, xj, c, s;
	for (j = 0; j < m - 1; j++) {
		A[(j + 1) * n + j] = 0;
		c = C[j];
		s = S[j];
		for (k = 0; k < j + 2; k++) {
			xi               = A[k * n + j];
			xj               = A[k * n + j + 1];
			A[k * n + j]     = xi * c - xj * s;
			A[k * n + j + 1] = xi * s + xj * c;
		}
	}
}

void RQ_rotate_step_z(double *A,double *Z, double *C,double *S,int n,int m) {
	int j,k;
	double xi,xj,c,s;
	for(j=0;j<m-1;j++) {
		A[(j+1)*n+j]=0;
		c=C[j];
		s=S[j];
		for(k=0;k<j+2;k++) {
			xi=A[k*n+j];
			xj=A[k*n+j+1];
			A[k*n+j]=xi*c-xj*s;
			A[k*n+j+1]=xi*s+xj*c;
			Z[k*n+j]=xi*c-xj*s;
			Z[k*n+j+1]=xi*s+xj*c;
		}
	}
}

/**
 * нахождение матрицы A^{k+1}=RQ
 * @param Q - получена с помощью отражений
 * @param A - матрица
 * @param C - вектор для хранения
 * @param S - вектор для хранения
 * @param n - размерность
 * @param m - размерность подматрицы
 */
void RQ_reflect_step(double *A, double *tmp, double *tmp2, int n, int m) {
	int j, k;
	double t, x1, x2;
	for (j = 0; j < m - 1; j++) {
		A[(j + 1) * n + j]=0;
		x2 = tmp [j];
		x1 = tmp2[j];
		for (k = 0; k <= j + 1; k++) {
			t                 = x1 * A[k * n + j] + x2 * A[k * n + j + 1];
			A[k * n + j]     -= 2 * x1 * t;
			A[k * n + j + 1] -= 2 * x2 * t;
		}
	}
}

void RQ_reflect_step_z(double *A,double *Z, double *tmp,double *tmp2
						,int n,int m) {
	int j,k;
	double t,x1,x2;
	for(j=0;j<m-1;j++) {
		A[(j+1)*n+j]=0;
		x2=tmp[j];
		x1=tmp2[j];
		for(k=0;k<=j+1;k++) {
			t=x1*A[k*n+j]+x2*A[k*n+j+1];
			A[k*n+j]-=2*x1*t;
			A[k*n+j+1]-=2*x2*t;
			Z[k*n+j]-=2*x1*t;
			Z[k*n+j+1]-=2*x2*t;
		}
	}
}

/**
 * нахождение матрицы A=QR в случае когда, для подматрицы
 * @param Q задана векторами косинусов и синусов
 * @param A - матрица
 * @param C - вектор для хранения косинусов матрицы вращений Q
 * @param S - вектор для хранения синусов
 * @param n - размерность
 * @param m - размерность подматрицы
 */
void QR_rotate_step(double *A, double *C, double *S, int n, int m) {
	int j, k;
	double xi, xj, c, s;
	for (j = m - 2; j >= 0; j--) {
		A[(j + 1) * n + j]=0;
		c = C[j];
		s = S[j];
		for (k = j + 1; k < m; k++) {
			xi                 = A[j * n + k];
			xj                 = A[(j + 1) * n + k];
			A[j * n + k]       =  xi * c + xj * s;
			A[(j + 1) * n + k] = -xi * s + xj * c;
		}
	}
}

void QR_rotate_step_z(double *A,double *Z, double *C,double *S
						,int n,int m) {
	int j,k;
	double xi,xj,c,s;
	for(j=m-2;j>=0;j--) {
		A[(j+1)*n+j]=0;
		c=C[j];
		s=S[j];
		for(k=j+1;k<m;k++) {
			xi=A[j*n+k];
			xj=A[(j+1)*n+k];
			A[j*n+k]=xi*c+xj*s;
			A[(j+1)*n+k]=-xi*s+xj*c;
			Z[j*n+k]=xi*c+xj*s;
			Z[(j+1)*n+k]=-xi*s+xj*c;
		}
	}
}


void ev_x_subspace_general_lapack2(const double * A1, double **Ex, 
								   double ** E, int n, int *nx)
{
	int i, j;
	int ilo = 1;
	int ihi = n;

	double * A    = malloc(n * n * sizeof(double));
	double * tau  = malloc(n * sizeof(double));
	int lwork     = n * n;
	double * work = malloc(lwork * sizeof(double));
	int info = 0;
	double * H;
/*	double * Q = malloc(n * n * sizeof(double));*/
	double * Z = calloc(n * n, sizeof(double));

	double * wr = malloc(n * sizeof(double));
	double * wi = malloc(n * sizeof(double));

	/*char * dgebal_job   = "B";*/
	char * dgebal_job   = "P";
	double * scale = malloc(n * n * sizeof(double));

	char * dhseqr_job   = "S";
	/*char * dhseqr_compz = "I";*/
	char * dhseqr_compz = "V";

	char * dtrsen_job   = "B";
	char * dtrsen_compq = "V";

	int * select = calloc(n, sizeof(int));

	int liwork  = n * n;
	int * iwork = malloc(liwork * sizeof(int));

	double S, SEP;

	double sigma = 0.0;

	matrix_copy_transpose(A, A1, n);

	printf("condition number=%.16le\n", condition_number(A, n));

	for (i = 0; i < n; ++i) A[i * n + i] -= sigma;

	printf("condition number=%.16le\n", condition_number(A, n));
	fflush(stdout);

	printfmatrix(A, n, 6, "%11.4le ");

	printf("running dgebal_ \n");
	dgebal_(dgebal_job, &n, A, &n, &ilo, &ihi, scale, &info);
	printf("done\n");

	//printfmatrix(A, n, 6, "%11.4le ");

	//if (info != 0) {
	//	fprintf(stderr, "error in dgebal_ \n");
	//	exit(1);
	//}

	/*приводим к верхнетреугольному виду*/
	printf("running dgehrd_ \n");
	dgehrd_(&n, &ilo, &ihi, A, &n, tau, work, &lwork, &info);
	printf("done\n");

	if (info != 0) {
		fprintf(stderr, "error in dgehrd_ \n");
		exit(1);
	}

	H = A;

/*	memcpy(Q, H, n * n * sizeof(double));*/
	memcpy(Z, H, n * n * sizeof(double));

	printf("running dorghr_ \n");
/*	dorghr_(&n, &ilo, &ihi, Q, &n, tau, work, &lwork, &info);*/
	dorghr_(&n, &ilo, &ihi, Z, &n, tau, work, &lwork, &info);
	printf("done\n");

	if (info != 0) {
		fprintf(stderr, "error in dorghr_ \n");
		exit(1);
	}

	/*for (i = 1; i < n; ++i) {
		for (j = 0; j < i - 1; ++j) {
			H[j * n + i] = 0.0;
		}
	}*/

	printfmatrix(H, n, 6, "%11.4le ");

	/*printf("running dgebal_ \n");
	dgebal_(dgebal_job, &n, A, &n, &ilo, &ihi, scale, &info);
	printf("done\n");

	printfmatrix(H, n, 6, "%11.4le ");

	if (info != 0) {
		fprintf(stderr, "error in dgebal_ \n");
		exit(1);
	}*/

	printf("running dhseqr_ \n");
	/*make_identity_matrix(Z, n);*/
	/*memcpy(Z, Q, n * n * sizeof(double));*/
	dhseqr_(dhseqr_job, dhseqr_compz, &n, &ilo, &ihi, H, &n, wr, wi,
		Z, &n, work, &lwork, &info);
	printf("done\n");

	printf("H\n");
	printfmatrix(H, n, 6, "%11.4le ");
	printf("Z\n");
	printfmatrix(Z, n, 6, "%11.4le ");

	if (info != 0) {
		fprintf(stderr, "error in dhseqr_ \n");
		exit(1);
	}

	*nx = 0;
	for (i = 0; i < n; ++i) {
		double w = sqrt((wr[i] + sigma) * (wr[i] + sigma) + wi[i] * wi[i]);

		if (w > 1) {
			select[i] = 1;
/*			++ *nx;*/
		}
	}

	printf("running dtrsen_ \n");
	dtrsen_(dtrsen_job, dtrsen_compq, select, &n, H, &n, Z, &n, 
		wr, wi, nx, &S, &SEP, work, &lwork, iwork, &liwork, &info);
	printf("done\n");

	if (info != 0) {
		fprintf(stderr, "error in dtrsen_ \n");
		exit(1);
	}

	printf("S=%.16lf SEP=%.16lf\n", S, SEP);

	/*M столбцов Z содержат ортонормированный базис*/

	if (Ex) {
		*Ex = malloc(*nx * n * sizeof(double));

		/*	matrix_transpose(Z, n);*/

		for (i = 0; i < *nx; ++i) {
			for (j = 0; j < n; ++j) {
				(*Ex)[j * (*nx) + i] = Z[i * n + j];
			}
		}
	}

	if (E) {
		*E = malloc(n * n * sizeof(double));
		matrix_copy_transpose(*E, Z, n);
	}

	free(tau); free(work); 
	free(wr); free(wi); 
	free(Z); free(select);
	free(scale); free(A);
	/*free(Q);*/
}

void ev_x_subspace_general_lapack3(const double * A1, double **Ex, int n, int *nx)
{
	int i, j;
	int ilo = 1;
	int ihi = n;

	double * A    = malloc(n * n * sizeof(double));
	double * tau  = malloc(n * sizeof(double));
	int lwork     = n * n;
	double * work = malloc(lwork * sizeof(double));
	int info = 0;
	double * H;
/*	double * Q = malloc(n * n * sizeof(double));*/
	double * Z = calloc(n * n, sizeof(double));

	double * wr = malloc(n * sizeof(double));
	double * wi = malloc(n * sizeof(double));

	//char * dgebal_job   = "B";
	char * dgebal_job   = "P";
	double * scale = malloc(n * n * sizeof(double));

	char * dhseqr_job   = "S";
	/*char * dhseqr_compz = "I";*/
	char * dhseqr_compz = "V";

	char * dtrsen_job   = "B";
	char * dtrsen_compq = "V";

	int * select = calloc(n, sizeof(int));

	int liwork  = n * n;
	int * iwork = malloc(liwork * sizeof(int));

	double S, SEP;

	matrix_copy_transpose(A, A1, n);

	printf("condition number=%.16le\n", condition_number(A, n));
	fflush(stdout);

	printfmatrix(A, n, 6, "%11.4le ");

	/*printf("running dgebal_ \n");
	dgebal_(dgebal_job, &n, A, &n, &ilo, &ihi, scale, &info);
	printf("done\n");

	printfmatrix(A, n, 6, "%11.4le ");

	if (info != 0) {
		fprintf(stderr, "error in dgebal_ \n");
		exit(1);
	}*/

	/*приводим к верхнетреугольному виду*/
	printf("running dgehrd_ \n");
	dgehrd_(&n, &ilo, &ihi, A, &n, tau, work, &lwork, &info);
	printf("done\n");

	if (info != 0) {
		fprintf(stderr, "error in dgehrd_ \n");
		exit(1);
	}

	H = A;

/*	memcpy(Q, H, n * n * sizeof(double));*/
	memcpy(Z, H, n * n * sizeof(double));

	printf("running dorghr_ \n");
/*	dorghr_(&n, &ilo, &ihi, Q, &n, tau, work, &lwork, &info);*/
	dorghr_(&n, &ilo, &ihi, Z, &n, tau, work, &lwork, &info);
	printf("done\n");

	if (info != 0) {
		fprintf(stderr, "error in dorghr_ \n");
		exit(1);
	}

	/*for (i = 1; i < n; ++i) {
		for (j = 0; j < i - 1; ++j) {
			H[j * n + i] = 0.0;
		}
	}*/

	printfmatrix(H, n, 6, "%11.4le ");

	/*printf("running dgebal_ \n");
	dgebal_(dgebal_job, &n, A, &n, &ilo, &ihi, scale, &info);
	printf("done\n");

	printfmatrix(H, n, 6, "%11.4le ");

	if (info != 0) {
		fprintf(stderr, "error in dgebal_ \n");
		exit(1);
	}*/

	printf("running dhseqr_ \n");
	/*make_identity_matrix(Z, n);*/
	/*memcpy(Z, Q, n * n * sizeof(double));*/
	dhseqr_(dhseqr_job, dhseqr_compz, &n, &ilo, &ihi, H, &n, wr, wi,
		Z, &n, work, &lwork, &info);
	printf("done\n");

	printf("H\n");
	printfmatrix(H, n, 6, "%11.4le ");
	printf("Z\n");
	printfmatrix(Z, n, 6, "%11.4le ");

	if (info != 0) {
		fprintf(stderr, "error in dhseqr_ \n");
		exit(1);
	}

	*nx = 0;
	for (i = 0; i < n; ++i) {
		//double w = sqrt(wr[i] * wr[i] + wi[i] * wi[i]);
//		if (wr[i] > 0) {
			double r  = exp(wr[i]);
			double vr = cos(wi[i]);
			double vi = sin(wi[i]);

			double w = r * sqrt(vr * vr + vi * vi);

//			printf("i=%d, w=%.16le\n", i, w);

			if (w > 1) {
				select[i] = 1;
			}
//		}
	}

	printf("running dtrsen_ \n");
	dtrsen_(dtrsen_job, dtrsen_compq, select, &n, H, &n, Z, &n, 
		wr, wi, nx, &S, &SEP, work, &lwork, iwork, &liwork, &info);
	printf("done\n");

	if (info != 0) {
		fprintf(stderr, "error in dtrsen_ \n");
		exit(1);
	}

	printf("S=%.16lf SEP=%.16lf\n", S, SEP);

	/*M столбцов Z содержат ортонормированный базис*/

	*Ex = malloc(*nx * n * sizeof(double));

/*	matrix_transpose(Z, n);*/

	for (i = 0; i < *nx; ++i) {
		for (j = 0; j < n; ++j) {
			(*Ex)[j * (*nx) + i] = Z[i * n + j];
		}
	}

	free(tau); free(work); 
	free(wr); free(wi); 
	free(Z); free(select);
	free(scale); free(A);
	/*free(Q);*/
}

static int
check_subspace(const double * S1, const double * S2, int n, int m, double eps)
{
	int i, j;

	for (i = 0; i < m; ++i) {
		const double * v1 = &S1[i * n];
		int flag = 0;
		for (j = 0; j < m; ++j) {
			const double * v2 = &S2[j * n];
			double dst = fabs(fabs(scalar(v1, v2, n)) - 1);
			if (dst < eps) {
				flag = 1;
				break;
			}
		}

		if (!flag) {
			return 0;
		}
	}

	return 1;
}

/*not tested!*/
void ev_x_subspace_iteration(const double * A, double **Ex, int n, int *nx)
{
	int iter   = 5;
	int m      = *nx;
	int nev    = *nx;	
	int i = 0, j = 0, k = 0;
	double * X  = malloc(n * m * sizeof(double));
	double * Z  = malloc(n * m * sizeof(double));
	double * ZT = malloc(n * m * sizeof(double));

	for (i = 0; i < nev; ++i) {
		basis(&X[i * n], n, i);
	}

	while (1) {
		for (i = 0; i < nev; ++i) {
			for (k = 0; k < iter; ++k) {
				matrix_mult_vector(&ZT[i * n], A, &X[i * n], n);
			}
		}

		ortogonalize_mgsch(ZT, n, nev);
		if (check_subspace(X, ZT, n, nev, 1e-5)) {
			memcpy(X, ZT, n * nev * sizeof(double));
			break;
		}
		memcpy(X, ZT, n * nev * sizeof(double));
	}

	*Ex = malloc(n * nev * sizeof(double));
	memcpy(Ex, X, n * nev * sizeof(double));

	free(X); free(Z); free(ZT);
}

/*************************************************************************
Деление комплексных чисел.

procedure DivComplex(a,b,c,d:real; var e,f:real);

e+if = (a+ib)/(c+id)
*************************************************************************/
static void divcomplex(
					   double a,
					   double b,
					   double c,
					   double d,
					   double* e,
					   double* f)
{
	double r;
	double d1;

	if( fabs(c)<fabs(d) )
	{
		r = c/d;
		d1 = d+r*c;
		*e = (a*r+b)/d1;
		*f = (b*r-a)/d1;
	}
	else
	{
		r = d/c;
		d1 = c+r*d;
		*e = (a+b*r)/d1;
		*f = (b-a*r)/d1;
	}
}

/*! вычисляет базис пространства X
shift-invert mode
\param A  - матрица линеаризации
\param Ex - базис X (память выделяется malloc'ом)
\param n  - размерность A
\param nx - предположительная размерность X (может увеличится
            за счет кратных значений)
*/
void ev_x_subspace_general_arnoldi2(double *A1, double **Ex, int n,
								  int *nx)
{
	int i, j, k;
	int info  = 0;          /* статус возврата */
    int  nev  =*nx;/*Number of eigenvalues of OP to be computed. 0 < NEV < N-1.*/
    int ncv    = ((2 * nev + 1) < n)?
		(2 * nev + 1):n; /*  Number of columns of the matrix V. NCV must satisfy the two
                        inequalities 2 <= NCV-NEV and NCV <= N.*/

	double  tol    = 0; //точность 0 - машинная
	int lworkl= 3 * ncv * ncv + 6 * ncv; /* хз что?*/
	int ldv   = n;
	int *c    = (int*)malloc(ncv * sizeof(int)); /* матрица классификации */

	double * resid  = (double*)malloc(n * sizeof(double));
	double * v      = (double*)malloc(ldv * ncv * sizeof(double));
	double * workd  = (double*)malloc(3 * n * sizeof(double));
	double * workev = (double*)malloc(3 * ncv * sizeof(double));
	double * workl  = (double*)malloc(lworkl * sizeof(double));

	double * A      = (double*)malloc(n * n * sizeof(double));

	double *wr   = (double*)malloc(n * sizeof(double));/*с. в.: вещественная часть*/
	double *wi   = (double*)malloc(n * sizeof(double));/*мнимая часть*/
	double *ex   = 0; /* собственные вектора, отвечающие X, Y, если надо то */
	/* BMAT = 'I' -> standard eigenvalue problem A*x = lambda*x*/
    /* BMAT = 'G' -> generalized eigenvalue problem A*x = lambda*B*x*/
	char   bmat[] ="I";
/* 'LM' -> want the NEV eigenvalues of largest magnitude. */
/* 'SM' -> want the NEV eigenvalues of smallest magnitude.*/
/* 'LR' -> want the NEV eigenvalues of largest real part. */
/* 'SR' -> want the NEV eigenvalues of smallest real part.*/
/* 'LI' -> want the NEV eigenvalues of largest imaginary part. */
/* 'SI' -> want the NEV eigenvalues of smallest imaginary part.*/
//	char which[] = "LM";
	char which[] = "LR";
//	char which[] = "LI";

	int  iparam[11], ipntr[14];
    int  ishfts = 1;
	int  maxitr = 400;
	int  mode   = 1;
//	int  mode   = 3;
	int  ido    = 0;

	//сдвиг
	double sigma = 1.0;

	memcpy(A, A1, n * n * sizeof(double));

	for (i = 0; i < n; ++i) {
		A[i * n + i] -= sigma;
	}

	printf("cn=%.16lf\n", condition_number(A, n));

//	inverse_general_matrix(A, A, n);

	//в фортране нумерация массива с 1
	iparam[1-1] = ishfts;
	iparam[3-1] = maxitr;
	iparam[7-1] = mode;

	do {
		dnaupd_ ( &ido, bmat, &n, which, &nev, &tol, resid,
			&ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl,
			&info );

		ERR((check_dnaupd_status(info) >= 0), "error in arpack's dnaupd_ procedure");
		if (ido == -1 || ido == 1) {
			int addr2 = ipntr[2-1]-1;
			int addr1 = ipntr[1-1]-1;
			double *w1 = &workd[addr2]; //ответ
			double *v1 = &workd[addr1]; //начальное условие

			memset(w1, 0, n * sizeof(double));
			for(i = 0; i < n; i++) {
				for(j = 0; j < n; j++) {
					w1[i] += A[i * n + j] * v1[j];
				}
			}
		}
	} while (ido == -1 || ido == 1);

	{
		char c[] = "A";
		int rvec   = 1;
		int *select = malloc(ncv * sizeof(int));
		double sigmar, sigmai;
		int ierr;

		dneupd_ ( &rvec, c, select, wr, wi, v, &ldv,
			&sigmar, &sigmai, workev, bmat, &n, which, &nev, &tol,
			resid, &ncv, v, &ldv, iparam, ipntr, workd, workl,
			&lworkl, &ierr );
		ERR((check_dneupd_status(ierr) >= 0), "error in arpack's dneupd_ procedure\n");
		free(select);
	}

	*nx = 0;
	nev = iparam[4];
	for (i = 0; i < nev; i++) {
		double vr, vi;
		double w;
		vr = wr[i] + sigma;
		vi = wi[i];
		//divcomplex(1.0, 0.0, wr[i], wi[i], &vr, &vi);
		//vr += sigma;
		
		w = sqrt(vr * vr + vi * vi);
		if (w > 1)    {c[i] = 1; (*nx)++;} /* x */
		else /*dist< 1*/ {c[i] = 0;} /* y */
	}

	printf("nx=%d\n", *nx);

	free(wr); free(wi);
	ex = (double*)malloc((*nx) * n * sizeof(double));
	for (j = 0, i = 0; i < nev; i++) {
		if (c[i] == 1) {
			for (k = 0; k < n; k++) {
				ex[k * (*nx) + j] = v[i * ldv + k];
			}
			j++;
		}
	}

	*Ex = ex;
	free(v); free(workd); free(workev); free(workl); free(resid);
	free(c);
}
