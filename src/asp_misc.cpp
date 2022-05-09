/* $Id$ */

/* Copyright (c) 2004, 2005, 2006, 2007 Alexey Ozeritsky
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
 * Мелкие алгоритмы.
 * в том числе матричный ввод вывод, векторый ввод вывод,
 * вычисления норм, загрузка матриц из файлов.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "asp_macros.h"
#include "asp_misc.h"
#include "asp_gauss.h"

namespace asp {

void printcopyright() {
	printf("$Id$\n");
	printf("_____________________________________________________________________\n");
	printf("Copyright (c) 2005 Alexey Ozeritsky\n");
	printf(" All rights reserved.\n\n");
	printf("Redistribution and use in source and binary forms, with or without\n");
	printf("modification, are permitted provided that the following conditions\n");
	printf("are met:\n");
	printf("1. Redistributions of source code must retain the above copyright\n");
	printf("   notice, this list of conditions and the following disclaimer.\n");
	printf("2. Redistributions in binary form must reproduce the above copyright\n");
	printf("   notice, this list of conditions and the following disclaimer in the\n");
	printf("   documentation and/or other materials provided with the distribution.\n");
	printf("3. All advertising materials mentioning features or use of this software\n");
	printf("   must display the following acknowledgement:\n");
	printf("     This product includes software developed by Alexey Ozeritsky.\n");
	printf("4. The name of the author may not be used to endorse or promote products\n");
	printf("   derived from this software without specific prior written permission\n\n");
	printf("THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR\n");
	printf("IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES\n");
	printf("OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.\n");
	printf("IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,\n");
	printf("INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT\n");
	printf("NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,\n");
	printf("DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY\n");
	printf("THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT\n");
	printf("(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF\n");
	printf("THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n");
	printf("_____________________________________________________________________\n\n\n");
}

/**
 * расстояние в стандартной евклидовой метрике
 */
double dist(const double *y1, const double *y2,int n) {
	int i;
	double n2=0;
	for (i = n - 1; i >= 0; --i) {
		n2+=(y1[i]-y2[i])*(y1[i]-y2[i]);
	}
	n2=sqrt(n2);
	return n2;
}

double dist2(const double *y1, const double *y2, int n) {
	return dist(y1, y2, n);
}

/*!длина в норме 1*/
double dist1(const double *y1, const double *y2, int n) {
	int i;
	double n1 = 0.0;
	double f;
	for (i = n - 1; i >= 0; --i) {
		f = fabs(y1[i]-y2[i]);
		if (f > n1) n1 = f;
	}
	return n1;
}

/**
 * норма 1
 */
double norm1(const double *x,int n) {
	int i;
	double n1=0.0;
	double f;
	for (i = n - 1; i >= 0; --i) {
		f = fabs(x[i]);
		if (n1 < f) n1 = f;
	}
	return n1;
}

/**
 * стандартная норма в R^n
 */
double norm2(const double *x,int n) {
	int i;
	double n2=0.0;
	for (i = n - 1; i >= 0; --i)
		n2 += x[i] * x[i];
	n2 = sqrt(n2);
	return n2;
}

void normalize1(double *x, int n) {
	int i = 0;
	double nr = norm1(x, n);
	for (i = 0; i < n; ++i) {
		x[i] /= nr;
	}
}

void normalize2(double *x, int n) {
	int i = 0;
	double nr = norm2(x, n);
	for (i = 0; i < n; ++i) {
		x[i] /= nr;
	}
}

void normilize_(double *x, double a, double b, int n)
{
	int i;
	double max = find_max(x, n);
	double min = find_min(x, n);

	double a1 = min, b1 = max, a2 = a, b2 = b;
	double k  = (b2 - a2) / (b1 - a1);
	double b_ = a2 - k * a1;
	for (i = 0; i < n; ++i) {
		//x[i] = normalize_point(x[i], min, max, a, b);
		x[i] = k * x[i] + b_;
	}
}

double normilize_point(double x, double a1, double b1, double a2, double b2)
{
	double k = (b2 - a2) / (b1 - a1);
	double b = a2 - k * a1;
	return k * x + b;
}

void normilize_point_(double *x, double a1, double b1, double a2, double b2)
{
	double k = (b2 - a2) / (b1 - a1);
	double b = a2 - k * a1;
	*x = k * *x + b;
}

	/**
	 * находит минимум в массиве
	 */
double find_min(const double *x, int n)
{
	int i;
	double min = x[0];
	for (i = 1; i < n; ++i) {
		if (min > x[i]) min = x[i];
	}
	return min;
}

	/**
	 * находит максимум в массиве
	 */
double find_max(const double *x, int n)
{
	int i;
	double max = x[0];
	for (i = 1; i < n; ++i) {
		if (max < x[i]) max = x[i];
	}
	return max;
}

/**
 * скалярное произведение
 */
double scalar(const double *X1, const double *X2, int k) {
	int i;
	double sum = 0.0;
	for (i = k - 1; i >= 0; --i)
		sum += X1[i] * X2[i];
	return sum;
}

	/*!линейное отображение вектора \param h
	в вектор \param h1, вектор разлагается по 
	ортогональному базису e,
	размерность пространства n, размерность отображения nx,
	вектора в базисе лежат по столбцам*/
void linear_reflection(double *h1, const double *h, const double *e, 
					   const double *L, int n, int nx)
{
	double * c = (double*)malloc(nx * sizeof(double));
	double *c_ = (double*)malloc(nx * sizeof(double));
	double *hn = (double*)malloc(n  * sizeof(double));
	int i, j, k;

	/*!находим коэф-ты разложения вектора h по базису e
	 * \f$h = \sum_{i=1}^{n_x} c_i e_i + \sum_{i=n_x+1}^{n}\cdot\f$
	 * \f$c_i = \frac{(h, e_i)}{(e_i, e_i)}\f$*/
	for (i = 0; i < nx; i++) {
		c[i]  = matrix_cols_scalar(h, e, 0, i, 1, nx, n);
		c[i] /= matrix_cols_scalar(e, e, i, i, nx, nx, n);
	}

	/*!находим \f$\hat c = L c\f$*/
	for (i = 0; i < nx; i++) {
		c_[i] = 0.0;
		for (k = 0; k < nx; k++) {
			c_[i] += L[i * nx + k] * c[k];
		}
	}

	memset(hn, 0, n * sizeof(double));
	for (i = 0; i < nx; i++) {
		for (j = 0; j < n; j++) {
			hn[j] += e[j * nx + i] * c_[i];
		}
	}

	memcpy(h1, hn, n * sizeof(double));

	free(c); free(c_); free(hn);
}

/**
 * скалярное произведение столбцов матриц
 * X - nx * n, Y - ny * n
 * nx, ny - число столбцов в матрицах
 */
double matrix_cols_scalar(const double *X, const double *Y, 
						  int i, int j, int nx, int ny, int n) 
{
	int k;
	double sum = 0.0;
	for (k = n - 1; k >= 0; --k) {
		sum += X[k * nx + i] * Y[k * ny + j];
	}
	return sum;
}

	/**
	 * Матричная экспонента
	 * @param Dest - результат экспонента от A.
	 * @param A - матрица
	 * @param n - размерность	 
	 */
void matrix_exp(double * Dest, const double * A, int n) {
	int i, m;
	int size = n * n;
	double * e = 0; //текущий шаг
	double * p = 0; //предыдущий шаг
	double f = 1.0; //факториал
	double norm;

	load_identity_matrix(&e, n);
	load_identity_matrix(&p, n);
	
	m = 1;
	while (1) {
		f *= m;
		//находим очередную степень A ^ n / n!
		matrix_mult(e, e, A, n);
		norm = 0.0;
		//находим sum a^2_ij
		for (i = 0; i < size; i++) {
			norm += e[i] * e[i] / f / f;
		}
		norm = sqrt(norm);
		if (norm < EPS32) break;
		ERR((m < 1000), "error: m > 100");

		for (i = 0; i < size; i++) {
			p[i] += e[i] / f;
		}		
		m ++;		
	}	
	memcpy(Dest, p, n * n * sizeof(double));
	free(e); free(p);
}


	/**
	 * Умножение матриц. Умножение безопасное. 
	 * Можно использовать одни и те же параметры для ответа и данных
	 * @param Dest - результат
	 * @param A
	 * @param B
	 * @param n
	 * @return A*B
	 */
void matrix_mult(double * Dest, const double * S1, const double * S2, int n) 
{
	int i, j, k;
	int offset;
	int need_to_free_A = 0;
	int need_to_free_B = 0;
	double * A = 0;
	double * B = 0;
	double * C = Dest;
	if (Dest == S1) {
		A = (double*)malloc(n * n * sizeof(double)); NOMEM(A);
		memcpy(A, S1, n * n * sizeof(double));
		need_to_free_A = 1;
	} else {
		A = (double*)S1;
	}
	if (Dest == S2) {
		B = (double*)malloc(n * n * sizeof(double)); NOMEM(B);
		memcpy(B, S2, n * n * sizeof(double));
		need_to_free_B = 1;
	} else {
		B = (double*)S2;
	}

	for (i = n - 1; i >= 0; --i) {
		for (j = n - 1; j >= 0; --j) {
			offset = i * n + j;
			C[offset] = 0.0;
			for (k = n - 1; k >= 0; --k) {
				C[offset] += A[i * n + k] * B[k * n + j];
			}
		}
	}
				
	if (need_to_free_A) free(A);
	if (need_to_free_B) free(B);
}

	/**
	 * Умножение матрицы на вектор . Умножение безопасное. 
	 * Можно использовать одни и те же параметры для ответа и данных
	 * @param Dest - результат
	 * @param A
	 * @param v
	 * @param n
	 * @return A * v
	 */
void matrix_mult_vector(double * Dest, const double * A, const double * v, 
						int n) 
{
	int i, k;
	double * C = (double*)malloc(n * sizeof(double));

	for (i = n - 1; i >= 0; --i) {
		C[i] = 0.0;
		for (k = n - 1; k >= 0; --k) {
			C[i] += A[i * n + k] * v[k];
		}
	}

	memcpy(Dest, C, n * sizeof(double));
	free(C);
}

void matrix_mult_scalar(double * Dest, const double * A, double v, int n) 
{
	int size = n * n;
	int i;
	for (i = size - 1; i >= 0; --i) {
		Dest[i] = A[i] * v;
	}
}

void vector_mult_scalar(double * Dest, const double * A, double v, int n)
{
	int i;
	for (i = n - 1; i >= 0; --i) {
		Dest[i] = A[i] * v;
	}
}

	/**
	 * Сумма матриц. 
	 * @param Dest - результат
	 * @param A
	 * @param B
	 * @param n
	 * @return A + B
	 */
void matrix_sum(double * Dest, const double * A, const double * B, int n) {
	int size = n * n;
	int i;
	for (i = size - 1; i >= 0; --i) {
		Dest[i] = A[i] + B[i];
	}
}

void vector_sum(double * Dest, const double * A, const double * B, int n) 
{
	int i;
	for (i = n - 1; i >= 0; --i) {
		Dest[i] = A[i] + B[i];
	}
}

void vector_sum1(double * Dest, const double * A, const double * B, 
				 double k1, double k2, int n) 
{
	int i;
	for (i = n - 1; i >= 0; --i) {
		Dest[i] = k1 * A[i] + k2 * B[i];
	}
}

void vector_sum2(double * Dest, const double * A, const double * B, 
				 double c, int n) 
{
	int i;
	for (i = n - 1; i >= 0; --i) {
		Dest[i] = A[i] + B[i] * c;
	}
}

	/**
	 * Разность векторов. 
	 * @param Dest - результат
	 * @param A
	 * @param B
	 * @param n
	 * @return A - B
	 */
void vector_diff(double * Dest, const double * A, const double * B, int n) {
	int i;
	for (i = n - 1; i >= 0; --i) {
		Dest[i] = A[i] - B[i];
	}
}

/**
 * печать матрицы размерности n,
 * подматрица размера m*m
 */
void  fprintmatrix(FILE*f, const double*A,int n,int m) {
	int i,j;
	for (i = 0; i < n && i < m; i++) {
		for (j = 0; j < n && j < m; j++) {
			fprintf(f, "%.1e ",A[i*n+j]);
		}
		fprintf(f,"\n");
	}
}

void  printmatrix(const double*A,int n,int m) {
	fprintmatrix(stdout,A,n,m);
}

/**
 * форматированная печать матрицы размерности n,
 * подматрица размера m*m
 */
void fprintfmatrix(FILE*f, const double*A,int n,int m, const char * format) {
	int i,j;
	for (i = 0; i < n && i < m; i++) {
		for (j = 0; j < n && j < m; j++) {
			fprintf(f,format,A[i*n+j]);
		}
		fprintf(f,"\n");
	}
}

void printfmatrix(const double*A,int n,int m, const char * format) {
	fprintfmatrix(stdout,A,n,m,format);
}

void _fprintfmatrix(const char *fname, const double*A, int n, int m, const char* format)
{
	FILE *f = fopen(fname, "w"); IOERR(f);
	fprintfmatrix(f, A, n, m, format);
	fclose(f);
}

/**
 * форматированная печать матрицы размерности n,
 * с форматированием отступов
 * подматрица размера m*m
 */
void fprintfvmatrix(FILE*f, const double*A,int n,int m, const char * format) {
	int i,j;
	int len = strlen(format) + 10;
	const char *format_negative = format;
	char *format_positive = (char*)malloc(len*sizeof(char));
	snprintf(format_positive, len - 1, " %s",format);
	for (i = 0; i < n && i < m; i++) {
		for (j = 0; j < n && j < m; j++) {
			if (A[i * n + j] < 0) {
				fprintf(f,format_negative,A[i*n+j]);
			} else {
				fprintf(f,format_positive,A[i*n+j]);
			}
		}
		fprintf(f,"\n");
	}
	free(format_positive);
}

void _fprintfvmatrix(const char *fname, const double*A, int n, int m, const char* format)
{
	FILE *f = fopen(fname, "w"); IOERR(f);
	fprintfvmatrix(f, A, n, m, format);
	fclose(f);
}

void printfvmatrix(const double*A, int n,int m, const char * format) {
	fprintfvmatrix(stdout,A,n,m,format);
}

/*для прямоугольных матриц*/
void printfwmatrix         (const double*A, int n, int k, int m, const char* format) {
	fprintfwmatrix(stdout,A,n,k,m,format);
}

void fprintfwmatrix(FILE*f, const double*A, int n, int k, int m, const char* format) {
	int i,j;
	int len = strlen(format);
	char *local_format = (char*)malloc((len + 10)*sizeof(char));
	for (j = 0, i = 0; i < len; ++i) {
		if (i > 0
			&& format[i - 1] == '%' 
			&& format[i]     == '.')
		{
			local_format[j++] = '1';
			local_format[j++] = '0';
			local_format[j++] = '.';
		} else {
			local_format[j++] = format[i];
		}
		local_format[j] = 0;
	}

	for (i = 0; i < n && i < m; i++) {
		for (j = 0; j < k && j < m; j++) {
			fprintf(f,local_format,A[i*k+j]);
		}
		fprintf(f,"\n");
	}
	free(local_format);
}

void _fprintfwmatrix(const char *fname, const double*A, int n, int k, int m, const char* format)
{
	FILE * f = fopen(fname, "w"); IOERR(f);
	fprintfwmatrix(f, A, n, k, m, format);
	fclose(f);
}

/**
 * печать вектора размера n
 * подвектора размера m
 */
void fprintvector(FILE*f, const double*A,int n,int m) {
	int i;
	for (i = 0; i < n && i < m; i++) {
		fprintf(f,"%.1e ",A[i]);
	}
}

void printvector(const double*A,int n,int m) {
	fprintvector(stdout,A,n,m);
}

void _fprintfvector(const char *fname, const double*A, int n, int m, const char* format)
{
	FILE * f = fopen(fname, "w");
	if (!f) return;
	fprintfvector(f, A, n, m, format);
	fclose(f);
}

/**
 * форматированная печать вектора размера n
 * подвектора размера m
 */
void fprintfvector(FILE*f, const double*A,int n,int m, const char * format) {
	int i;
	for (i = 0; i < n && i < m; i++) {
		fprintf(f,format,A[i]);
	}
}

void printfvector(const double*A,int n,int m, const char * format) {
	fprintfvector(stdout,A,n,m,format);
}

/**
 * форматированная печать вектора размера n
 * печать с отступами для положительных
 * подвектора размера m
 */
void fprintfvvector(FILE*f, const double*A,int n,int m, const char * format) {
	int i;
	int len = strlen(format) + 10;
	const char *format_negative = format;
	char *format_positive = (char*)malloc(len*sizeof(char));
	snprintf(format_positive, len - 1, " %s", format);
	for (i = 0; i < n && i < m; i++) {
		if(A[i]<0) {
			fprintf(f,format_negative,A[i]);
		} else {
			fprintf(f,format_positive,A[i]);
		}
	}
	free(format_positive);
}

void printfvvector(const double*A, int n,int m, const char * format) {
	fprintfvvector(stdout,A,n,m,format);
}

/*целочисленный ввод/вывод*/
/*неформатированный вывод*/
void printimatrix(const int *A, int n, int m) {
	fprintimatrix(stdout, A, n, m);
}

void printivector(const int *A, int n, int m) {
	fprintivector(stdout, A, n, m);
}

void fprintimatrix(FILE*f, const int *A, int n, int m) {
	int i,j;
	for (i = 0; i < n && i < m; i++) {
		for (j = 0; j < n && j < m; j++) {
			fprintf(f, "%d ",A[i*n+j]);
		}
		fprintf(f,"\n");
	}
}
	
void fprintivector(FILE*f, const int*A, int n, int m) {
	int i;
	for (i = 0; i < n && i < m; i++) {
		fprintf(f,"%d ",A[i]);
	}
}

/**
 * целочисленное возведение в степень
 *
 * @param x
 * @param n
 * @return
 */
double ipow(double x, int n) {
    double otv = 1.;
	int i;
    for (i = 0; i < n; i++)
        otv *= x;
    return otv;
}

int _ipow(int x, int n) {
    int otv = 1;
	int i;
    for (i = 0; i < n; i++)
        otv *= x;
    return otv;
}

void matrix2file(const double**M, int N1, int N2, const char* fname) {
	int i,j;
	FILE *f = fopen(fname,"w");
	IOERR(f);                
    for (i = 0; i < N1; i++) {
        for (j = 0; j < N2; j++) {
            fprintf(f,"%.3e ",M[i][j]);
        }
        fprintf(f,"\n");
    }
	fclose(f);
}

void _matrix2file(const double *M, int N1, int N2, const char *fname) {
	int i,j;
	FILE *f = fopen(fname,"w");
	IOERR(f);                
    for (i = 0; i < N1; i++) {
        for (j = 0; j < N2; j++) {
            fprintf(f,"%.3e ",M[i * N2 + j]);
        }
        fprintf(f,"\n");
    }
	fclose(f);
}

/**
 * невязка
 *
 * @param M1 реальный ответ
 * @param M2 получившийся
 * @return
 */
double matrix_diff(const double** M1, const double** M2, int N1, int N2) {
	int i,j;
    double max = 0.0;
    double max1;

    for (i = 0; i < N1; i++) {
        for (j = 0; j < N2; j++) {
            max1 = fabs(M1[i][j] - M2[i][j]);
            if (max < max1) max = max1;
        }
    }
    return max;
}

double _matrix_diff(const double* M1, const double* M2, int N1, int N2) {
	int i,j;
    double max = 0.0;
    double max1;

    for (i = 0; i < N1; i++) {
        for (j = 0; j < N2; j++) {
            max1 = fabs(M1[i * N2 + j] - M2[i * N2 + j]);
            if (max < max1) max = max1;
        }
    }
    return max;
}

double _matrix_inside_diff(const double *A, const double *B, int N1, int N2) 
{
	int i,j;
    double max = 0.0;
    double max1;

    for (i = 2; i < N1 - 2; i++) {
        for (j = 2; j < N2 - 2; j++) {
            max1 = fabs(A[i * N2 + j] - B[i * N2 + j]);
            if (max < max1) max = max1;
        }
    }
    return max;
}

/*целочисленный минимум и максимум двух чисел*/
int  mini(int n,int m) {
	if(n<m) return n;
	else return m;
}

int  maxi(int n,int m) {
	if(n<m) return m;
	else return n;
}
/*вещественные минимум и максимум двух чисел*/
double maxd(double n,double m) {
	if(n<m) return m;
	else return n;
}

double mind(double n,double m) {
	if(n<m) return n;
	else return m;
}

/*для трех чисел*/
double max3d(double x, double y, double z) {
	if(x>y&&x>z)
		return x;
	else if(y>x&&y>z)
		return y;
	else
		return z;
}

double min3d(double x, double y, double z) {
	if(x<y&&x<z)
		return x;
	else if(y<x&&y<z)
		return y;
	else
		return z;
}

int max3i(int x, int y, int z) {
	if(x>y&&x>z)
		return x;
	else if(y>x&&y>z)
		return y;
	else
		return z;
}

int min3i(int x, int y, int z) {
	if(x<y&&x<z)
		return x;
	else if(y<x&&y<z)
		return y;
	else
		return z;
}

/*транспонирование*/
void matrix_transpose(double *A, int n) {
	int i, j;
	double t;
	for (i = 1; i < n; i++) {
		for (j = 0; j < i; j++) {
			t = A[i * n + j];
			A[i * n + j] = A[j * n + i];
			A[j * n + i] = t;
		}
	}
}

/*транспонирование с копированием*/
void matrix_copy_transpose(double *Dest, const double * Source, int n) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n ; j++) {
			Dest[j * n + i] = Source[i * n + j];
		}
	}
}

void make_identity_matrix(double *X, int n) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) X[i * n + j] = 1.0;
			else X[i * n + j] = 0.0;
		}
	}
}

void load_identity_matrix(double **X, int n) {
	*X = (double *) malloc(n * n * sizeof(double));
	make_identity_matrix(*X, n);
}

/* возвращает три средние диагонали матрицы */
void extract_tdiags_from_matrix(double *D, double *M, double *U, 
								const double*A, int n) 
{
	int i;
	M[0] = A[0];
	for (i = 1; i < n; ++i) {
		if (M != 0) M[i] = A[i * n + i];
		if (D != 0) D[i - 1] = A[i * n + i - 1];
		if (U != 0) U[i - 1] = A[(i - 1)* n + i];
	}
}

	/**
	 * загрузка матрицы из файла формата
	 * n размерность
	 * n * n чисел формата double
	 */
void load_dim_matrix_from_txtfile(double **A, int *n, const char *filename) {
	int i;
	FILE * f = fopen(filename, "r");
	IOERR(f);
	ERR((fscanf(f,"%d",n) == 1), "bad file format");
	*A = (double*)malloc(*n**n*sizeof(double));
	for (i = 0; i < *n**n; ++i) {
		ERR((fscanf(f,"%lf",&(*A)[i]) == 1), "bad file format");
	}
	fclose(f);
}

//размер буфера для сохранения
#define SB_MIN_SIZE 512
void save_matrix_to_binfile(const double *A, int n, int m, 
							int endian,
							const char * separator,
							const char *filename)
{
	FILE * f = fopen(filename, "wb");
	uchar buf[SB_MIN_SIZE * sizeof(double)];
	int i, j, k, l = 0;
	int sz  = sizeof(double);
	int sep = strlen(separator);

	IOERR(f);

	for (i = 0; i < n; ++i) {
		for (j = 0; j < m; ++j) {
			double d = A[i * m + j];
			uchar *s = (uchar*)&d;
			switch (endian) {
			case 1:  //little
#ifdef LITTLE_ENDIAN
				for (k = 0; k < sz; ++k) {
					buf[l + k] = s[k];
				}
#else //BIG_ENDIAN
				for (k = sz - 1; k >= 0; --k) {
					buf[l + sz - k - 1] = s[k];
				}
#endif
				break;
			default: //big
#ifdef LITTLE_ENDIAN
				for (k = sz - 1; k >= 0; --k) {
					buf[l + sz - k - 1] = s[k];
				}
#else //BIG_ENDIAN
				for (k = 0; k < sz; ++k) {
					buf[l + k] = s[k];
				}
#endif
				break;
			}
			l += sz;
			if (l >= SB_MIN_SIZE * sz) {
				fwrite(buf, 1, SB_MIN_SIZE * sz, f);
				l = 0;
			}
		}
		fwrite(buf, 1, l, f);
		fwrite(separator, 1, sep, f);
		l = 0;
	}

	if (l < SB_MIN_SIZE * sz) {
		fwrite(buf, 1, l, f);
		fwrite(separator, 1, sep, f);
	}

	fclose(f);
}

#define LB_MIN_SIZE 512
void load_matrix_from_binfile(double **A, int *n, int *m, 
							  int endian,
							  const char * separator,							  
							  const char *filename)
{
	FILE * f   = fopen(filename, "rb");
	char * str = 0;
	uchar s[sizeof(double)];
	double * M   = 0;
	double * vec = 0;
	double *nvec = 0;
	int len, i = 0, k, l;
	int sz = sizeof(double);

	IOERR(f);


	str = _fget_long_string(f, separator, &len);
	if (!str) {
		fclose(f);
		return;
	}

	*m = len / sz;
	if (*m == 0) {
		fclose(f);
		return;
	}

	vec = (double*)malloc(*m * sz);

	do {
		if (len < *m * sz)
			continue;
		for (l = 0; l < *m * sz; l += sz)
		{
			switch(endian) {
			case 1:  //little
#ifdef LITTLE_ENDIAN
				for (k = 0; k < sz; ++k) {
					s[k] = str[l + k];
				}
#else //BIG_ENDIAN
				for (k = sz - 1; k >= 0; --k) {
					s[k] = str[l + sz - k - 1];
				}
#endif
				break;
			default: //big
#ifdef LITTLE_ENDIAN
				for (k = sz - 1; k >= 0; --k) {
					s[k] = str[l + sz - k - 1];
				}
#else //BIG_ENDIAN
				for (k = 0; k < sz; ++k) {
					s[k] = str[l + k];
				}
#endif
				break;
			}
			memcpy(&vec[l / sz], s, sz);
		}
		free(str); str = 0;
		nvec = (double*)realloc(M, (i + 1) * *m * sz);
		if (nvec) {
			M = nvec;
		} else {
			break;
		}
		memcpy(&M[i * *m], vec, *m * sz);
		str = _fget_long_string(f, separator, &len);

		++i;
	} while (str && !feof(f));

	*A = M;
	*n = i;
	free(vec);
	if (str) free(str);
	fclose(f);
}

#define LT_MIN_SIZE 128
char * _fget_long_string(FILE *f, const char * separator, int *len)
{
	char * buf = (char*)malloc(LT_MIN_SIZE);
	char * str = (char*)malloc(LT_MIN_SIZE);
	char *nstr = 0;
	int c = 0, i = 0, j = 0;
	int count;
	int sep = strlen(separator);

	NOMEM(buf); NOMEM(str);

	str[0] = 0;
	i      = 0;
	while (1)
	{
		c = fread(buf, 1, LT_MIN_SIZE, f);
		if (c <= sep && i == 0) {
			free(str); free(buf);
			return 0;
		}
		count = 0;
		while (count < (c - sep)) {
			int flag = 1;
			for (j = 0; j < sep; j++) {
				if (buf[count + j] != separator[j]) {
					flag = 0;
					break;
				}
			}
			if (flag)
				break;
			++count;
		}

		memcpy(&str[i * LT_MIN_SIZE], buf, c);
		if (count < (c - sep) || (feof(f) && count == (c - sep))) 
		{
			fseek(f, count - c + sep, SEEK_CUR);
			str[i * LT_MIN_SIZE + count] = 0;
			if (len)
				*len = i * LT_MIN_SIZE + count;
			break;
		}
		nstr = (char*)realloc(str, (i + 2) * LT_MIN_SIZE);
		if (nstr) {
			str = nstr;
		} else {
			str[i * LT_MIN_SIZE - 1] = 0;
			if (len)
				*len = 0;
			break;
		}
		++i;
	}
	free(buf);
	return str;
}

char * fget_long_string(FILE *f)
{
	return _fget_long_string(f, "\n", 0);
}

void noise_data(double *x, double percent, int n)
{
	static int init = 0;
	double max_, min_;
	int i;

	if (!init) {
		srand((int)time(0));
		init = 1;
	}

	max_ = find_max(x, n);
	min_ = find_min(x, n);

	if (max_ == min_) {
		max_ = 1.0;
		min_ = 0.0;
	}

	for (i = 0; i < n; ++i) {
		if ((double)rand() / (double)RAND_MAX > 0.5) {
			x[i] = x[i] + 0.01 * percent * fabs(max_ - min_) *
				(2.0 * ((double)rand() / (double)RAND_MAX - 1.0));
		}
	}
}

void load_matrix_from_txtfile(double **A, int *n, int *m, const char *filename)
{
	double * vec = (double*)malloc(LT_MIN_SIZE * sizeof(double));
	double * M   = 0;
	double *nvec = 0;
	double a;
	char * str;
	int pos = 0, i = 0;
	int size, len;
	char * token;
	const char * sep = " \t\r\n";

	FILE * f = fopen(filename, "rb");
	printf("open %s\n", filename);
	IOERR(f);

	/*число столбцов матрицы равно числу чисел первой строки*/
	str  = fget_long_string(f);
	if (str == 0) {
		*A = 0; *n = 0; *m = 0;
		free(vec);
		fclose(f);
		return;
	}

	len  = strlen(str);
	size = LT_MIN_SIZE;
	i    = 0;
	pos  = 0;

	token = strtok(str, sep);
	while (token) {
		if (sscanf(token, "%lf", &a) == 1) {
			if (i >= size) {
				size = 2 * size;
				nvec = (double*)realloc(vec, size * sizeof(double));
				if (nvec) {
					vec = nvec;
				} else {
					break;
				}
			}
			vec[i++] = a;
		}
		token = strtok(0, sep);
	}

	*m = i;
	free(str);

	M = (double*)malloc(*m * sizeof(double));
	memcpy(M, vec, *m * sizeof(double));

	*n = 1;
	while (1) {
		str  = fget_long_string(f);
		if (str == 0)
			break;
		len  = strlen(str);
		i    = 0;
		pos  = 0;
		token= strtok(str, sep);
		while (token && i < *m) {
			if (sscanf(token, "%lf", &a) == 1) {
				vec[i++] = a;
			}
			token = strtok(0, sep);
		}
		free(str);

		if (i < *m)
			continue;

		nvec = (double*)realloc(M, *m * (*n + 1) * sizeof(double));		
		if (nvec) {
			M = nvec;
			memcpy(&M[*n * *m], vec, *m * sizeof(double));
		} else {
			break;
		}
		++(*n);
		if (feof(f))
			break;
	}

	*A = M;
	free(vec);
	fclose(f);
}

	/**
	 * загрузка матрицы из функции
	 */
void load_matrix_from_func(double **A, int n, double (*load_matrix_func)(int i, int j)) {
	int j, k;
	*A = (double*)malloc(n * n * sizeof(double));
	for (k = 0; k < n; ++k) {
		for (j = 0; j < n; ++j) {
			(*A)[k* n + j] = load_matrix_func(k, j);
		}
	}
}

	/**
	 * загрузка из функции или из файла
	 * приоритет по файлу
	 * если оба указателя ноль, то получаем сообщение об ошибке
	 */
void load_dim_matrix_from_txtfile_or_func(double **A, int *n, 
										  const char *filename, 
										  double (*load_matrix_func)(int i, int j)) 
{	
	if (filename != 0) {
		load_dim_matrix_from_txtfile(A, n, filename);
		return;
	}
	if (load_matrix_func != 0) {
		load_matrix_from_func(A, *n, load_matrix_func);
		return;
	}
	ERR(0, "filename or load_matrix_func must be > 0");
}

void gramm_matrix(double * g, const double *e1, const double *e2, 
				  int m, int n) 
{
	int i, j;
	for (i = 0; i < m; ++i) {
		for (j = 0; j < m; ++j) {
			g[i * m + j] = matrix_cols_scalar(e1, e2, i, j, m, m, n);
		}
	}
}

	/* транспонирование с копированием для прямоугольных матриц.
	 * @param Dest   - ответ размерности m * n
	 * @param Source - исходная матрица размерности n * m
	 */
void matrix_copy_transposew(double * Dest, const double * Source , 
							int n, int m) 
{
	double * A = Dest;
	const double * B = Source;
	int i, j;
	for (i = 0; i < n; ++i) {
		for (j = 0; j < m; ++j) {
			A[j * n + i] = B[i * m + j];
		}
	}
}

	/* транспонирование прямоугольных матриц.
	 * @param A   - матрица размерности m * n
	 */
void matrix_transposew(double * A, int n, int m) {
	//!\todo не рационально!
	double * temp = (double*)malloc(n * m * sizeof(double));
	matrix_copy_transposew(temp, A, n, m);
	memcpy(A, temp, n * m * sizeof(double));
	free(temp);
}

void random_initialize(double *a, int n) 
{
	static int init = 1;
	int i;
	if (!init) {
		srand((time_t)time(0));
		init = 1;
	}

	for (i = n - 1; i >= 0; --i) {
		a[i] = (double)rand() / RAND_MAX - 0.5;
	}
}

void basis(double *vec, int n, int i)
{
	memset(vec, 0, n * sizeof(double));
	vec[i] = 1;
}
}

