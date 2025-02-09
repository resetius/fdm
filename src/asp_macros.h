#ifndef _ASP_MACROS_H
#define _ASP_MACROS_H
/* Copyright (c) 2005, 2006, 2022 Alexey Ozeritsky
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
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
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
 * Макросы
 */

#include <float.h>
#ifdef WIN32
#include <sstream>
#endif

#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif
#ifndef M_E
#define M_E        2.71828182845904523536
#endif
#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923
#endif
#ifndef M_PI2
#define M_PI2 6.28318530717958648
#endif

//#ifdef WIN32
//#if defined(FDM_LIBRARY_EXPORT)
//#define FDM_API   __declspec( dllexport ) 
//#else
//#define FDM_API   __declspec( dllimport ) 
//#endif
//#else
#define FDM_API
//#endif

#ifdef _WIN32
#define isNaN _isnan
#define isInf __noop//_isinf
#define snprintf _snprintf
#else
#define isNaN isnan
#define isInf isinf
#endif

#if defined(_WIN32) && defined(_DEBUG)
#define _PRESSKEY
#endif

#ifdef _PRESSKEY
#ifndef __cplusplus
#define PRESSKEY printf("press enter\n"); fflush(stdout); getchar();
#else
#define PRESSKEY ::printf("press enter\n"); fflush(stdout); getchar();
#endif
#else
#define PRESSKEY
#endif

#define sign(x) (x > 0)? 1 : -1

#ifdef _DEBUG
#ifdef __cplusplus
#define NOMEM(a) if(!a) { throw BadArgument("NOMEM: ", __FILE__, __LINE__); }
#else
#define NOMEM(a) if(!a) { fprintf(stderr,"NOMEM: %s:%d\n",__FILE__,__LINE__); \
	PRESSKEY; exit(100); }
#endif
#ifdef __cplusplus
#define MEMSIZE(a) if(!(a>0)) { \
	std::ostringstream str; \
	str << "Cannot allocate " << a; \
	throw BadArgument(str.str().c_str(), __FILE__, __LINE__); \
	}
#else
#define MEMSIZE(a) if(!(a>0)) { fprintf(stderr, "Cannot allocate %d bytes: %s:%d\n",a,__FILE__,__LINE__); \
    PRESSKEY; exit(101); }
#endif
#ifdef __cplusplus
#define ERR(a,b) if(!a) { \
	std::ostringstream str; \
	str << "error: " << b << " :"; \
	throw BadArgument(str.str().c_str(), __FILE__, __LINE__); \
	}
#else
#define ERR(a,b) if(!a) { fprintf(stderr,"error:%s: %s:%d\n",b,__FILE__,__LINE__); \
    PRESSKEY; exit(300); }
#endif
#define WARN(a,b) if(a) { fprintf(stderr,"warning:%s: %s:%d\n",b,__FILE__,__LINE__); }
#else
#define NOMEM(a)
#define MEMSIZE(a)
#define ERR(a,b)
#define WARN(a,b)
#endif

#ifdef __cplusplus
#define IOERR(a) if(!a) { throw NotFound("", __FILE__, __LINE__); }
#else
#define IOERR(a) if(!a) { fprintf(stderr,"IOERR: %s:%d\n",__FILE__,__LINE__); perror(""); \
	PRESSKEY; exit(200); }
#endif

#ifdef __cplusplus
#define NOTIMPL { throw NotImplemented(__FUNCTION__,__FILE__,__LINE__); }
#else
#define NOTIMPL { fprintf(stderr,"%s NOT Implemented: %s:%d\n",__FUNCTION__,__FILE__,__LINE__); \
	PRESSKEY; exit(400); }
#endif

#ifdef _DEBUG
//i, j - границы
//k - переменная
#define _DEBUG_ASSERT_RANGE(i, j, k) \
	if (!( ( i ) <= ( k ) && ( k ) <= ( j ) )) \
       throw IndexOutOfRange(( i ), ( j ), ( k ));
#else
#define _DEBUG_ASSERT_RANGE(i, j, k)
#endif

typedef unsigned int  uint;
typedef unsigned int  uint32;
typedef unsigned long ulong;
typedef unsigned char uchar;

#undef BIG_ENDIAN
#undef LITTLE_ENDIAN

#if defined(__BIGEND__)
#define BIG_ENDIAN
#else //if defined(__LITTLEEND__)
#define LITTLE_ENDIAN
#endif

#define EPS32 1e-15
#define EPS16 1e-7

#ifdef __cplusplus
#include "asp_excs.h"
#endif

#endif //_ASP_MACROS_H
