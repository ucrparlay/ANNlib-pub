#ifndef __DIST_HPP__
#define __DIST_HPP__

#include <type_traits>
#include <cstdint>
#include "parse_points.hpp"
#include "NSGDist.h"
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

// Single-pass float dot product. efanna2e's DistanceInnerProduct caps at AVX
// (256-bit, 8 floats) with separate mul+add; on a machine with AVX512 that is
// half the width UNIFY uses (its space_ip.h has an _mm512 path). For
// high-dimensional angular data (cohere 768, openai 1536) the large-range
// post-filter beam is distance-bound, so this width gap is the main reason our
// KQPS trails UNIFY there. This kernel uses AVX512 + FMA (16-wide, 4
// accumulators to hide FMA latency), then AVX2+FMA, then a scalar fallback.
static inline float cosnorm_dot_f32(const float *a, const float *b, uint32_t n)
{
#if defined(__AVX512F__)
	__m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
	uint32_t i = 0;
	for (; i + 64 <= n; i += 64) {
		s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i),    _mm512_loadu_ps(b+i),    s0);
		s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16), s1);
		s2 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+32), _mm512_loadu_ps(b+i+32), s2);
		s3 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+48), _mm512_loadu_ps(b+i+48), s3);
	}
	for (; i + 16 <= n; i += 16)
		s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i), _mm512_loadu_ps(b+i), s0);
	float dot = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0,s1), _mm512_add_ps(s2,s3)));
	for (; i < n; ++i) dot += a[i]*b[i];
	return dot;
#elif defined(__AVX2__) && defined(__FMA__)
	__m256 s0 = _mm256_setzero_ps(), s1 = s0;
	uint32_t i = 0;
	for (; i + 16 <= n; i += 16) {
		s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),   _mm256_loadu_ps(b+i),   s0);
		s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8), _mm256_loadu_ps(b+i+8), s1);
	}
	for (; i + 8 <= n; i += 8)
		s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), s0);
	__m256 s = _mm256_add_ps(s0, s1);
	__m128 lo = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
	lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
	float dot = _mm_cvtss_f32(lo);
	for (; i < n; ++i) dot += a[i]*b[i];
	return dot;
#else
	float dot = 0;
	for (uint32_t i = 0; i < n; ++i) dot += a[i]*b[i];
	return dot;
#endif
}

template<typename T>
class descr_ang
{
	using promoted_type = std::conditional_t<std::is_integral_v<T>&&sizeof(T)<=4,
		std::conditional_t<sizeof(T)==4, int64_t, int32_t>,
		float
	>;
public:
	typedef T type_elem;
	typedef point<T> type_point;
	typedef typename type_point::coord_t type_coord;
	static float distance(const type_coord &uc, const type_coord &vc, uint32_t dim)
	{
		promoted_type dot=0, nu=0, nv=0;
		for(uint32_t i=0; i<dim; ++i)
		{
			nu += promoted_type(uc[i])*uc[i];
			nv += promoted_type(vc[i])*vc[i];
			dot += promoted_type(uc[i])*vc[i];
		}
		return 1-dot/(sqrtf(nu)*sqrtf(nv));
	}

	static auto get_id(const type_point &u)
	{
		return u.id;
	}
};

// Cosine distance for coordinates that are ALREADY unit-normalized: then
// cos(u,v) == dot(u,v), so the distance is simply 1 - dot.
//
// This exists because descr_ang is the bottleneck on high-dimensional angular
// datasets: it runs three multiply-accumulate chains per element (dot plus both
// norms, and the query's own norm is recomputed for every candidate) and then a
// division by two sqrtf's -- all as a plain scalar loop, which gcc cannot
// auto-vectorize without -ffast-math because float reduction is not associative.
// Here we do one pass through efanna2e's AVX dot kernel instead, which is what
// UNIFY does (its benchmark pre-normalizes the vectors and uses a SIMD inner
// product). Only valid when the stored coordinates were normalized --
// slotted_vamana does that at load time when constructed with normalize=true.
template<typename T>
class descr_cosine_norm
{
public:
	typedef T type_elem;
	typedef point<T> type_point;
	typedef typename type_point::coord_t type_coord;
	static float distance(const type_coord &uc, const type_coord &vc, uint32_t dim)
	{
		if constexpr(std::is_same_v<T,float>)
		{
			return 1.0f - cosnorm_dot_f32(uc, vc, dim);
		}
		else
		{
			float dot = 0;
			for(uint32_t i=0; i<dim; ++i)
				dot += float(uc[i])*float(vc[i]);
			return 1.0f-dot;
		}
	}

	static auto get_id(const type_point &u)
	{
		return u.id;
	}
};

template<typename T>
class descr_ndot
{
	using promoted_type = std::conditional_t<std::is_integral_v<T>&&sizeof(T)<=4,
		std::conditional_t<sizeof(T)==4, int64_t, int32_t>,
		float
	>;
public:
	typedef T type_elem;
	typedef point<T> type_point;
	typedef typename type_point::coord_t type_coord;
	static float distance(const type_coord &uc, const type_coord &vc, uint32_t dim)
	{
		promoted_type dot=0;
		for(uint32_t i=0; i<dim; ++i)
			dot += promoted_type(uc[i])*vc[i];
		return -float(dot);
	}

	static auto get_id(const type_point &u)
	{
		return u.id;
	}
};

template<typename T>
class descr_l2
{
	using promoted_type = std::conditional_t<std::is_integral_v<T>&&sizeof(T)<=4,
		std::conditional_t<sizeof(T)==4, int64_t, int32_t>,
		float
	>;
public:
	typedef T type_elem;
	typedef point<T> type_point;
	typedef typename type_point::coord_t type_coord;
	static float distance(const type_coord &uc, const type_coord &vc, uint32_t dim)
	{
		if constexpr(std::is_integral_v<T>)
		{
			promoted_type sum = 0;
			for(uint32_t i=0; i<dim; ++i)
			{
				const auto d = promoted_type(uc[i])-vc[i];
				sum += d*d;
			}
			return sum;
		}
		else
		{
			efanna2e::DistanceL2 distfunc;
			return distfunc.compare(uc, vc, dim);
		}
	}

	static auto get_id(const type_point &u)
	{
		return u.id;
	}
};

#endif // __DIST_HPP__
