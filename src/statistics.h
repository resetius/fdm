#ifndef STATISTICS_H
#define STATISTICS_H

#include <vector>

template < typename T >
class ExpectedValue
{
	typedef std::vector < T > array_t;
	int n;
	int i;
	array_t cur;

public:
	ExpectedValue(int n): n(n), i(0), cur(n) {}

	// M = sum ai / n
	// M0 = ai
	// Mi = ((n-1) sum a(i-1) / (n+1) + ai)/n = ((n-1)M(i-1) + ai)/n
	void accumulate(const array_t & value)
	{
		i += 1;
		for (int i0 = 0; i0 < n; ++i0) {
			cur[i0] = ((i - 1) * cur[i0] + value[i0]) / (double) i;
		}
	}

	array_t current()
	{
		return cur;
	}
};

template < typename T >
class Variance
{
	typedef std::vector < T > array_t;
	int n;
	ExpectedValue < T > m2x;
	ExpectedValue < T > mx2;

public:
	Variance(int n): n(n), m2x(n), mx2(n) {}

	void accumulate(const array_t & value)
	{
		array_t tmp(n);
		for (int i0 = 0; i0 < n; ++i0) {
			tmp[i0] = value[i0] * value[i0];
		}
		m2x.accumulate(value);
		mx2.accumulate(tmp);
	}

	array_t current()
	{
		array_t m2x_cur = m2x.current();
		array_t mx2_cur = mx2.current();
		array_t cur(n);
		for (int i0 = 0; i0 < n; ++i0) {
			cur[i0] = mx2_cur[i0] - m2x_cur[i0] * m2x_cur[i0];
		}
		return cur;
	}

	array_t m_current()
	{
		return m2x.current();
	}
};

#endif /* STATISTICS_H */
