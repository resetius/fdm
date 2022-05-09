#ifndef _SPHERE_DATA
#define _SPHERE_DATA

/*$Id$*/

/* Copyright (c) 2006 Alexey Ozeritsky
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
 * Параметры сферической сетки -- шаги, число точек и тп
 */

#include <vector>

struct SData: public SSteps {
	SLaplacian * lapl; //!<сферический оператор Лапласа
	SJacobian  * jac;  //!<сферический якобиан

	double tau;
	double tau_1;

	double forward_mult;
	double forward_diag;
	double backward_mult;
	double backward_diag;

	std::vector < double > LA;  //!<координаты по lambda
	std::vector < double > PHI;  //!<координаты по phi

	SData(int n_phi, int n_la, bool full)
		: SSteps(n_phi, n_la, full),
		lapl(0), LA(0), PHI(0)
	{
	}

	virtual ~SData()
	{
		delete lapl;	
		delete jac;
	}

	void init()
	{
		lapl = new SLaplacian(n_phi, n_la, full);
		jac  = new SJacobian(n_phi, n_la, full);

		LA.resize(n_la); 
		PHI.resize(n_phi);
		for (int i = 0; i < n_la; i++) {
			LA[i] = (double)i * d_la;
		}

		double phi;
		for (int i = 0; i < n_phi; i++) {
			if (full)
				phi =(d_phi - M_PI) * 0.5 + (double)i * d_phi;
			else
				phi = (double)i * d_phi;
			PHI[i] = phi;
		}
	}
};

#endif //_SPHERE_DATA
