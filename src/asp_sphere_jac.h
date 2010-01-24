#ifndef _SPHERE_JAC_H
#define _SPHERE_JAC_H
/*$Id$*/

/* Copyright (c) 2005, 2006 Alexey Ozeritsky
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
 *  ����������� �������
 */

#include "asp_sphere_lapl.h"

class SJacobian {
	class Private;
	Private * d;

public:
	/*!
	   �����������
	   \param n_phi - ����� ����� �� ������
	   \param n_la  - ����� ����� �� �������
	   \param full  - ������������ ������ ����� ��� ���������?
	*/
	SJacobian(int n_phi, int n_la, bool full = false);
	~SJacobian();

	/*!
	    u_x v_y - v_x u_y
	 */
	double J1(const double *u, const double *v, int i, int j);
	void J1(double *dest, const double *u, const double *v);

	/*!
	    ����������� �������� � J1 ��� ������������� ����� ���������
		\param k - ����� (������� � 1) ��������� 
		           ������������ �������� ���������,
		           �������������� ������ �������� �����������
	 */
	double J1T(const double *u, const double *v, int i, int j, int k);
	void J1T(double *dest, const double *u, const double *v, int k);

	/*!
	    (u_y v)_x - (u_x v)_y
	 */
	double J2(const double *u, const double *v, int i, int j);
	void J2(double *dest, const double *u, const double *v);

	/*!
	    ����������� �������� � J2 ��� ������������� ����� ���������
		\param k - ����� (������� � 1) ��������� 
		           ������������ �������� ���������,
		           �������������� ������ �������� �����������
	 */
	double J2T(const double *u, const double *v, int i, int j, int k);
	void J2T(double *dest, const double *u, const double *v, int k);

	/*!
	    (v_x u)_y - (v_y u)_x
	 */
	double J3(const double *u, const double *v, int i, int j);
	void J3(double *dest, const double *u, const double *v);

	/*!
	    ����������� �������� � J3 ��� ������������� ����� ���������
		\param k - ����� (������� � 1) ��������� 
		           ������������ �������� ���������,
		           �������������� ������ �������� �����������
	 */
	double J3T(const double *u, const double *v, int i, int j, int k);
	void J3T(double *dest, const double *u, const double *v, int k);

	/*!����� �������*/
	double J(const double *u, const double *v, int i, int j);
	void J(double *dest, const double *u, const double *v);

	/*!
	   ����������� � J, ���������� �� ������� � �������
	   ��������� ���������
     */
	void JT(double *dest, const double *u, const double *v);
	double JT(const double *u, const double *v, int i, int j);

	/*!��������� ������������ � ����������� �����������*/
	double scalar(const double *u, const double *v);

	void JV2(double * dest, const double * u, const double * b);
	void JV2T(double * dest, const double * u, const double * b);
};

#endif
