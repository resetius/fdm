#pragma once

#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAVE_FFTW3
#include <fftw3.h>
#endif

namespace fdm {

template<typename T>
class FFTTable {
public:
    int N;
    std::vector<T> ffCOS;
    std::vector<T> ffSIN;
    std::vector<T> ffiCOS;

    std::vector<T> ffEXPr;
    std::vector<T> ffEXPi;

    FFTTable(int N): N(N)
    {
        init();
    }

    T iCOS(int k, int l) const {
        return ffiCOS[(l+1)*N+(k-1)];
    }

private:
    void init();
};

/*
  FFT algorithms for real numbers from classic book
  A. A. Samarskii, E. S. Nikolaev, "Numerical Methods For Grid
  Equations" (Birkhauser Verlag, 1989)
  original in russian:
  Самарский А. А., Николаев Е. С.
  "Методы решения сеточных уравнений", М: Наука, 1978.
 */
template<typename T>
class FFT {
    const FFTTable<T>& t;

    int N;
    int n; // N = 2^n

    std::vector<T> b;
    std::vector<T> bn;

    // for fft3d
    std::vector<T> in;
    std::vector<T> out;

public:
    FFT(const FFTTable<T>& table, int N)
        : t(table)
        , N(N)
        , b(N)
        , bn(N)
        , in(N)
        , out(N)
    {
        init();
    }

    /*!быстрое преобразование Фурье периодической функции.
      по коэфф Фурье находим значения функции.
      fk->f(i)
      Самарский-Николаев, страница 180-181
      \param S  - ответ
      \param s  - начальное условие
      \param dx - множитель перед суммой
	*/
    void pFFT(T *S, T* s, T dx);

    // new optimized version
    void sFFT(T* S, T* s, T dx);
    void cFFT(T* S, T* s, T dx);
    void pFFT_1(T* S, T* s, T dx);

    void pFFT_3d(T* S, T* s, T dx);
    void pFFT_1_3d(T* S, T* s, T dx);

    void cpFFT(T* S, T* s, T dx);

private:
    void init();

    void sFFT(T* S, T* s, T dx, int N, int n);
    void cFFT(T* S, T* s, T dx, int N, int n);
};

// fftw3 wrapper
#ifdef HAVE_FFTW3
template<typename T>
struct FFT_fftw3_plan {};

template<>
struct FFT_fftw3_plan<double> {
    fftw_complex* r2c_out;
    double* r2c_in;
    fftw_plan r2c_plan;

    double* c2r_out;
    fftw_complex* c2r_in;
    fftw_plan c2r_plan;

    double* dst1_in;
    double* dst1_out;
    fftw_plan dst1_plan;

    double* dct1_in;
    double* dct1_out;
    fftw_plan dct1_plan;

    FFT_fftw3_plan(int N)
        : r2c_out(fftw_alloc_complex(N/2+1))
        , r2c_in(fftw_alloc_real(N))
        , r2c_plan(fftw_plan_dft_r2c_1d(
            N, r2c_in, r2c_out, FFTW_ESTIMATE
        ))
        , c2r_out(fftw_alloc_real(N))
        , c2r_in(fftw_alloc_complex(N/2+1))
        , c2r_plan(fftw_plan_dft_c2r_1d(
            N, c2r_in, c2r_out, FFTW_ESTIMATE
        ))
        , dst1_in(fftw_alloc_real(N-1))
        , dst1_out(fftw_alloc_real(N-1))
        , dst1_plan(fftw_plan_r2r_1d(
            N-1, dst1_in, dst1_out, FFTW_RODFT00, FFTW_ESTIMATE
        ))
        , dct1_in(fftw_alloc_real(N+1))
        , dct1_out(fftw_alloc_real(N+1))
        , dct1_plan(fftw_plan_r2r_1d(
            N+1, dct1_in, dct1_out, FFTW_REDFT00, FFTW_ESTIMATE
        ))
    { }

    ~FFT_fftw3_plan() {
        fftw_destroy_plan(r2c_plan);
        fftw_free(r2c_out);
        fftw_free(r2c_in);

        fftw_destroy_plan(c2r_plan);
        fftw_free(c2r_out);
        fftw_free(c2r_in);

        fftw_destroy_plan(dst1_plan);
        fftw_free(dst1_out);
        fftw_free(dst1_in);

        fftw_destroy_plan(dct1_plan);
        fftw_free(dct1_out);
        fftw_free(dct1_in);
    }

    FFT_fftw3_plan(const FFT_fftw3_plan&) = delete;
    FFT_fftw3_plan(FFT_fftw3_plan&&) = default;

    void c2r_execute() {
        fftw_execute(c2r_plan);
    }

    void r2c_execute() {
        fftw_execute(r2c_plan);
    }

    void dst1_execute() {
        fftw_execute(dst1_plan);
    }

    void dct1_execute() {
        fftw_execute(dct1_plan);
    }
};

template<>
struct FFT_fftw3_plan<float> {
    fftwf_complex* r2c_out;
    float* r2c_in;
    fftwf_plan r2c_plan;

    float* c2r_out;
    fftwf_complex* c2r_in;
    fftwf_plan c2r_plan;

    float* dst1_in;
    float* dst1_out;
    fftwf_plan dst1_plan;

    float* dct1_in;
    float* dct1_out;
    fftwf_plan dct1_plan;

    FFT_fftw3_plan(int N)
        : r2c_out(fftwf_alloc_complex(N/2+1))
        , r2c_in(fftwf_alloc_real(N))
        , r2c_plan(fftwf_plan_dft_r2c_1d(
            N, r2c_in, r2c_out, FFTW_ESTIMATE
        ))
        , c2r_out(fftwf_alloc_real(N))
        , c2r_in(fftwf_alloc_complex(N/2+1))
        , c2r_plan(fftwf_plan_dft_c2r_1d(
            N, c2r_in, c2r_out, FFTW_ESTIMATE
        ))
        , dst1_in(fftwf_alloc_real(N-1))
        , dst1_out(fftwf_alloc_real(N-1))
        , dst1_plan(fftwf_plan_r2r_1d(
            N-1, dst1_in, dst1_out, FFTW_RODFT00, FFTW_ESTIMATE
        ))
        , dct1_in(fftwf_alloc_real(N+1))
        , dct1_out(fftwf_alloc_real(N+1))
        , dct1_plan(fftwf_plan_r2r_1d(
            N+1, dct1_in, dct1_out, FFTW_REDFT00, FFTW_ESTIMATE
        ))
    { }

    ~FFT_fftw3_plan() {
        fftwf_destroy_plan(r2c_plan);
        fftwf_free(r2c_out);
        fftwf_free(r2c_in);

        fftwf_destroy_plan(c2r_plan);
        fftwf_free(c2r_out);
        fftwf_free(c2r_in);

        fftwf_destroy_plan(dst1_plan);
        fftwf_free(dst1_out);
        fftwf_free(dst1_in);

        fftwf_destroy_plan(dct1_plan);
        fftwf_free(dct1_out);
        fftwf_free(dct1_in);
    }

    FFT_fftw3_plan(const FFT_fftw3_plan&) = delete;
    FFT_fftw3_plan(FFT_fftw3_plan&&) = default;

    void c2r_execute() {
        fftwf_execute(c2r_plan);
    }

    void r2c_execute() {
        fftwf_execute(r2c_plan);
    }

    void dst1_execute() {
        fftwf_execute(dst1_plan);
    }

    void dct1_execute() {
        fftwf_execute(dct1_plan);
    }
};

template<typename T>
class FFT_fftw3 {
    int N;
    FFT_fftw3_plan<T> plan;

public:
    FFT_fftw3(int N)
        : N(N)
        , plan(N)
    { }

    void pFFT_1(T *S, T *s1, T dx);
    void pFFT(T *S, T* s, T dx);
    void sFFT(T *S, T *s, T dx);
    void cFFT(T *S, T *s, T dx);
};
#endif

// don't use
template<typename T>
class FFT_old {
    const FFTTable<T>& t;

    int N;
    int n; // N = 2^n

public:
    FFT_old(const FFTTable<T>& table, int N)
        : t(table)
        , N(N)
    {
        init();
    }

	/*!быстрое преобразование Фурье периодической функции.
      по значениям функции находим коэфф Фурье.
      f(i)->fk
      Самарский-Николаев, страница 180-181, формулы 65-66
      \param S  - ответ
      \param s  - начальное условие
      \param dx - множитель перед суммой
	*/
	void pFFT_1(T *S, T *s1, T dx);

	/*! быстрое косинусное преобразование.
	   Самарский-Николаев, страница 176, формулы 46-47
       fftw: REDFT00
       S и s: массивы размера N+1, 0-indexing
	 */
    void cFFT(T *S, T *s, T dx);
	/*! быстрое синусное преобразование.
	   Самарский-Николаев, страница 180
       fftw: RODFT00
       S и s: массивы размера N-1, 1-indexing
	 */
    void sFFT(T *S, T *s, T dx);

private:
    void init();

    void sFFT(T *S, T *s, T dx, int N, int n,int nr);
    void cFFT(T *S, T *s, T dx, int N, int n,int nr);
};

// don't use
// experimental, don not use!
// these omp functions were created to help me debug and implement
// ideas for GPU (GLSL) version of fft
// they don't speedup anything on CPU because of high cost of threads creating
// and synchronization but the same approach for GPU works perfectly
template<typename T>
class FFT_debug_omp {
    const FFTTable<T>& t;

    int N;
    int n; // N = 2^n

public:
    FFT_debug_omp(const FFTTable<T>& table, int N)
        : t(table)
        , N(N)
    {
        init();
    }

    void sFFT(T* S, T* s, T dx);
    void cFFT(T* S, T* s, T dx);
    void pFFT_1(T *S, T* s, T dx);
    void pFFT(T *S, T* s, T dx);

private:
    void init();
};

template<typename T, typename U>
class FFTOmpSafe
{
    std::vector<U> instances;

    static int thread_count() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

    static int thread_id() {
#ifdef _OPENMP
        return omp_get_thread_num();
#else
        return 0;
#endif
    }

public:
    template <typename... Args>
    FFTOmpSafe(Args&&... args)
    {
        instances.reserve(thread_count());
        for (int i = 0; i < thread_count(); i++) {
            instances.emplace_back(std::forward<Args>(args)...);
        }
    }

    void pFFT_1(T *S, T* s, T dx) {
        instances[thread_id()].pFFT_1(S, s, dx);
    }

    void pFFT(T *S, T* s, T dx) {
        instances[thread_id()].pFFT(S, s, dx);
    }

    void sFFT(T* S, T* s, T dx) {
        instances[thread_id()].sFFT(S, s, dx);
    }

    void cFFT(T* S, T* s, T dx) {
        instances[thread_id()].cFFT(S, s, dx);
    }
};

} // namespace fdm

