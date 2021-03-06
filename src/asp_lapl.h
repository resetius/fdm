/*$Id$*/

/* Copyright (c) 2005, 2006, 2014 Alexey Ozeritsky
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
 * 3. The name of the author may not be used to endorse or promote products
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
 *  �������� ������� � ��������������
 */

#ifndef _ASP_LAPL_H
#define _ASP_LAPL_H

#include "asp_macros.h"

/*!
   ���� �� ��������������
 */
struct SqSteps {
	/*! ����� ����� �� �������� ��������������	*/
	int n_x;
	int n_y;
	double lx;
	double ly;
	double d_x;  //!<��� �� x
	double d_y;  //!<��� �� y
	double d_x2; //!<��� �� x � ��������
	double d_y2; //!<��� �� y � ��������

	double d_x_1; //!<1 ������ �� ��� �� x
	double d_y_1; //!<1 ������ �� ��� �� y
	double d_x2_1;//!<1 ������ �� ��� �� x � ��������
	double d_y2_1;//!<1 ������ �� ��� �� y � ��������

	SqSteps(double l_x, double l_y, int nx, int ny)
		: n_x(nx), n_y(ny), lx(l_x), ly(l_y)
	{
        d_x = lx / n_x;
        d_y = ly / n_y;
		
		d_x_1 = 1.0 / d_x;
		d_y_1 = 1.0 / d_y;

        d_x2 = d_x * d_x;
        d_y2 = d_y * d_y;

		d_x2_1 = 1.0 / d_x2;
		d_y2_1 = 1.0 / d_y2;
	}
};

/*!
   �������� ������� � ��������������
   ��� ��������� ������������ ��������+����� �� ��� Y
   ���������� �� �������->�� �������� ������������ ��� X
   ������ ����������� ������� ������� ��������� ����!

   �������� ������� � ���� A[x_i * (n_y + 1) + y_i]
   ��� ������� ������������ n_x, n_y �������� ����� �������� 
   ��������� lx � ly, �� ���� ����� �� ���� ������, ��� ������
   for (x_i)
     for (y_j)
	    print(A[x_i, y_j])
	������� ����� �����
	(0, 0), (0, 1) ...  ->y
	(1, 0), (1, 1) ...
	|
	x
 */
class FDM_API Laplacian2D {
	class Private;
	Private * d;

public:
	Laplacian2D(double l_x, double l_y, int n_x, int n_y);
	~Laplacian2D();

	void lapl(double * Dest, const double * M);
	double lapl(const double * M, int i, int j);
	/*������� ������� - ����*/
	void lapl_1(double * Dest, const double * Source, 
		double mult = 1.0, double diag = 0.0);
};

/*!���������� �������� �������*/
class FDM_API Laplacian {
	double d_x;
	double d_x2_1;
	double l_x;

	int n_x;
	int n; // ����������� ������
	int type;

public:
	enum {
		ZERO_COND = 0, //!<������� ������� �������
		PERIODIC  = 1  //!<������������� ������� �������
	};

	// n_x ����� ��������
	// ����������� � ������������� ������ n = n_x
	//             � ���������������      n = n_x+1 
	Laplacian(double l_x, int n_x, int type = ZERO_COND);

	void lapl(double * Dest, const double * M);
	double lapl(const double * M, int i);
};

extern "C" {
	void FDM_API fdm_lapl1d(
		double * dst, const double * src,
		double * l_x,
		int * n_x, int type);

	void FDM_API fdm_lapl2d(
		double * dst, const double * src,
		double * l_x, double * l_y,
		int * n_x, int * n_y);
}

#endif //_ASP_LAPL_H
