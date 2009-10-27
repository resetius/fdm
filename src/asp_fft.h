#ifndef _ASP_FFT_H
#define _ASP_FFT_H
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
 * ������� �������������� �����
 *
 */

//������������ ���������� fftw?
//warning: ���������� ��� GPL ���������!
//��� ������������ ���������� ���� ������ ���� define !
//#define _FFTW

	enum FFT_type {
		FFT_SIN      = 1, //!<�������� ��������������
		FFT_COS      = 2, //!<���������� ��������������
		FFT_PERIODIC = 3  //!<������������� (�������� �� ��� � ��� �� ���) FFT_SIN & FFT_COS
	};

	typedef struct _fft_internal {
		int type; //!<��� ��������������
		int N;    //!<����������� ������
		int sz;   //!<������ ������ � sin/cos
		int n;    //!<������� ������
		int k;    //!<����-� ���������� ������ cos/sin (����� ��� PERIODIC)
		double * ffCOS;
		double * ffSIN;
	} fft;

	//!�������������� ���������
	fft * FFT_init(int type, int N);
	void FFT_free(fft *);

/*~~~ �������� ������� � ��������� ����������� � �������� ~~~~~ */
/*~~~ sin((2 * k - 1) * M_PI * j / (double)(idx * 2)); ~~~~~~~~ */
/*~~~ cos((2 * k - 1) * M_PI * j / (double)(idx * 2)); ~~~~~~~~ */
	/*!������� �������������� ����� ������������� �������.
	   �� ����� ����� ������� �������� �������.
	   fk->f(i)
       ���������-��������, �������� 180-181
	  \param S  - �����
	  \param s  - ��������� �������
	  \param dx - ��������� ����� ������
	  \param N  - ����� �����
	  \param n  - log2(N)
	*/
	void pFFT(double *S, double *s, double dx, int N, int n);
      /** �������� �������������:
        dx = d_la * sqrt(M_1_PI)  // ��� * ������ �� 2/_�����_�������(�� ����� 2pi)
        pFFT_1(s, S, dx, n_la, fft);
        ������ �������������:
        dx = d_la * sqrt(M_1_PI) // ������ �� 2/_�����_�������(�� ����� 2pi)
        pFFT(S, s, dx, n_la, fft);
      */

	/*!������� �������������� ����� ������������� �������.
	   �� ��������� ������� ������� ����� �����.
	   f(i)->fk
       ���������-��������, �������� 180-181, ������� 65-66
	  \param S  - �����
	  \param s  - ��������� �������
	  \param dx - ��������� ����� ������
	  \param N  - ����� �����
	  \param n  - log2(N)
	*/
	void pFFT_1(double *S, double *s1, double dx, int N, int n);

	/*! ������� �������� ��������������.
	   ���������-��������, �������� 180
	 */
	void sFFT(double *S, double *s, double dx, int N, int n);
	/*!��������� �������� ��������������*/
	void sFT(double *S, double *s, double dx, int N);

	/*! ������� ���������� ��������������.
	   ���������-��������, �������� 176, ������� 46-47
	 */
	void cFFT(double *S, double *s, double dx, int N, int n);
	/*!��������� ���������� ��������������*/
	void cFT(double *S, double *s, double dx, int N);

/*~~~ �������� ������� � ��������� ���� ~~~~~ */
/*������ sin((2 * k - 1) * M_PI * j / (double)(idx * 2));
  ���������� ������
  ffSIN[(2 * k - 1) * vm * 2 * n_la + j];
  ������ cos((2 * k - 1) * M_PI * j / (double)(idx * 2));
  ���������� ������
  ffCOS[(2 * k - 1) * vm * 2 * n_la + j];
  ���, �������� ������� ��� � ������ ����� �� �����:
  (� ������� ������ ����� ����� ���������� ����������
  ��� sin, cos)

			ffCOS = new double[n_la * n_la];
			ffSIN = new double[n_la * n_la];

			for (int m = 0; m < n_la; m++) {
				for (int j = 0; j < n_la; j++) {
					ffCOS[m * n_la + j] =
						cos(m * M_PI * j / (double)n_la);
					ffSIN[m * n_la + j] =
						sin(m * M_PI * j / (double)n_la);
				}
			}
*/
	/*!������� �������������� ����� ������������� �������.
	   �� ����� ����� ������� �������� �������.
	   fk->f(i)
       ���������-��������, �������� 180-181
	  \param S  - �����
	  \param s  - ��������� �������
	  \param dx - ��������� ����� ������
	  \param p  - ���������, ��������������������� FFT_init
	*/
	void pFFT_2(double *S, double *s, double dx, fft * p);

      /** �������� �������������:
        dx = d_la * sqrt(M_1_PI)  // ��� * ������ �� 2/_�����_�������(�� ����� 2pi)
        pFFT_1(s, S, dx, n_la, fft);
        ������ �������������:
        dx = d_la * sqrt(M_1_PI) // ������ �� 2/_�����_�������(�� ����� 2pi)
        pFFT(S, s, dx, n_la, fft);
      */

	/*!������� �������������� ����� ������������� �������.
	   �� ��������� ������� ������� ����� �����.
	   f(i)->fk
       ���������-��������, �������� 180-181, ������� 65-66
	  \param S  - �����
	  \param s  - ��������� �������
	  \param dx - ��������� ����� ������
	  \param p  - ���������, ��������������������� FFT_init
	*/
	void pFFT_2_1(double *S, const double *s1, double dx, fft * p);

	/*! ������� �������� ��������������.
	   ���������-��������, �������� 180
	 */
	void sFFT_2(double *S, double *s, double dx, fft * p);

	/*! ������� ���������� ��������������.
	    ���������-��������, �������� 176, ������� 46-47
	 */
	void cFFT_2(double *S, double *s, double dx, fft * p);

	/*!������� �������������� �����.
	   ���������-��������, �������� 170, ������� 30-31
	  \param S  - �����
	  \param s  - ��������� �������
	  \param dx - ��������� ����� ������
	      d_y * sqrt(2 / l_y) ��� �������������� �������� ������� � ���� �����
		� sqrt(2 / l_y) ��� ���������� ������� �� ���� �����
	  \param Vm - ����������� ������� � ����
	      Vm(m, j) = sqrt(2. / ly) * sin(M_PI * m * j / n_y)
		  ��� l_y - ����� �������, n_y - ����� �����
	  \param N  - ����� �����
	  \param n  - log2(N)
	*/
	void FFT(double *S, const double *s, double *Vm, double dx, int N, int n);

#endif //_ASP_FFT_H
