/*  $Id$  */

/* Copyright (c) 2000, 2001, 2002, 2003, 2004, 2005 Alexey Ozeritsky
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


/** Методы гаусса с произвольной точностью.
 *	
 */


#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
//#include "macros.h"
#include "asp_gauss_mp.h"
using namespace std;



void  gauss_mp(int n,mpf_class* A,mpf_class *B)
{
	int i,j=0,k=0;
	mpf_class a,b;
	for(j=0;j<n;j++)
	{	

		/*a=A[j*n+j];
		  if(a==0)
		  {
		  findmain(&r,j,n,A);
		  swop(r,j,n,A);
		  tmp=B[r];B[r]=B[j];B[j]=tmp;
		  }*/	
		

		a=A[j*n+j];
		for(i=j+1;i<min1(n,j+4);i++)
		{
	
			b=A[(i)*n+j];
			for(k=j;k<n;k++) A[(i)*n+k]-=A[(j)*n+k]*(b/a);
			B[i]-=B[j]*(b/a);
		
		}

	}
    
	for(j=n-1;j>=0;j--)
	{	
		a=A[(j)*n+j];
		for(i=j-1;i>=max1_i(j-3,0);i--)
		{
		
			b=A[(i)*n+j];
			/*for(k=n-1;k>=j;k--) A[(i)*n+k]-=A[(j)*n+k]*(b/a);*/
			B[i]-=B[j]*(b/a);
	
		}
		
	}
	for(i=0;i<n;i++)
		B[i]*=1/(A[i*n+i]);
		
	/*printvector(B,n,7);		*/
	
}


void  tgauss_mp(int n,mpf_class*A1,mpf_class *A2,mpf_class *A3,mpf_class *B)
{
	int i,j=0;
	mpf_class a;
/*    printvector(A1,n-1,7);
    printvector(A2,n,7);
    printvector(A3,n-1,7);
    printvector(B,n);*/
	for(j=0;j<n-1;j++)
	{	
		a=A1[j]/A2[j];
		A1[j]=0;
		A2[j+1]-=A3[j]*a;
		B[j+1]-=B[j]*a;
	}
    
	for(j=n-1;j>0;j--)
	{	
		a=A3[j-1]/A2[j];
		B[j-1]-=B[j]*a;
	}
    
	for(i=0;i<n;i++)
		B[i]*=1/(A2[i]);
    
/*    printvector(B,n);*/
}


int  min1(int n,int m)
{
	if(n<m) return n;
	else return m;
}

int  max1_i(int n,int m)
{
	if(n<m) return m;
	else return n;
}

mpf_class  max1_d(mpf_class n,mpf_class m)
{
	if(n<m) return m;
	else return n;
}

/*int min1(int n,int m)
  {
  if(n<m) return n;
  else return m;
  }

  int max1(int n,int m)
  {
  if(n<m) return m;
  else return n;
  }*/



void  printmatrix(mpf_class*A,int n,int m)
{
	int i,j,k;
	k=min1(n,m);
	for(i=0;i<k;i++)
	{
		for(j=0;j<k;j++)
		{
			cout<<A[i*n+j]<<" ";
		}
		printf("\n");
	}
}

void  printvector(mpf_class*A,int n,int m)
{
	int i;/*,k;*/
/*    k=min1(n,m);*/
	for(i=0;i<n&&i<m;i++)
	{
		cout<<A[i];
	}
}


void  findmain(int*r,int k,int n,mpf_class*A)
{
	int j;
	mpf_class max=A[k*n+k];
	mpf_class c;
	(*r)=k;
	for(j=k;j<n;j++)
	{
		c=abs(A[j*n+k]);
		if(max<c)
		{
			(*r)=j;
			/*(*s)=k;*/
			max=c;
		}	
	}
}

void  swop(int r,int k,int n,mpf_class*A)
{
	int i;
	mpf_class temp;
	for(i=k;i<n;i++)
	{
		temp=A[k*n+i];A[k*n+i]=A[r*n+i];A[r*n+i]=temp;
	}
}	

void  swop2(int r,int k,int n,mpf_class*A)
{
	int i;
	mpf_class temp;
	if(r!=k)
		for(i=0;i<1;i++)
		{
			temp=A[k*n+i];A[k*n+i]=A[r*n+i];A[r*n+i]=temp;
		}
}	

