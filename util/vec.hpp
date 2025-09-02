#ifndef _ANN_UTIL_VEC_HPP
#define _ANN_UTIL_VEC_HPP

#include <cassert>
#include <cstdint>
#include <iterator>
#include <utility>
#include <type_traits>
#include "custom/custom.hpp"
#include "util.hpp"
#include "shared.hpp"

namespace ANN::util{

template<typename C>
class vec
{
	using coord_t = C;
	coord_t coord;
public:
	using elem_t = typename coord_t::value_type;

	vec() = default;
	vec(const vec&) = default;
	vec(vec &&) noexcept = default;

	template<typename A, typename ...Args, typename=std::enable_if_t<
		!std::is_same_v<std::decay_t<A>, inner_t> &&
		!std::is_same_v<std::decay_t<A>, vec>
	>>
	vec(A &&a, Args &&...args) :
		coord(std::forward<A>(a), std::forward<Args>(args)...){
	}

	template<typename U>
	vec(inner_t, const U &src, uint32_t dim) : coord(dim){
		for(uint32_t i=0; i<dim; ++i)
			coord[i] = src[i];
	}

	vec& operator=(const vec&) = default;
	vec& operator=(vec &&) noexcept = default;

	const elem_t* data() const{
		return std::data(coord);
	}
	auto size() const{
		return coord.size();
	}
	decltype(auto) operator[](size_t idx) const{
		return coord[idx];
	}

	vec& operator+=(const vec &v){
		assert(size()==v.size());
		for(size_t i=0; i<size(); ++i)
			coord[i] += v[i];
		return *this;
	}

	vec& operator-=(const vec &v){
		assert(size()==v.size());
		for(size_t i=0; i<size(); ++i)
			coord[i] -= v[i];
		return *this;
	}

	template<typename T>
	vec& operator*=(const T &val){
		for(auto &e : coord) e *= val;
		return *this;
	}

	template<typename T>
	vec& operator/=(const T &val){
		for(auto &e : coord) e /= val;
		return *this;
	}

	friend vec operator-(const vec &v){
		auto res = v;
		for(auto &e : res.coord) e = -e;
		return res;
	}
};

template<typename C>
inline vec<C> operator+(const vec<C> &lhs, const vec<C> &rhs){
	auto res = lhs;
	res += rhs;
	return res;
}

template<typename C>
inline vec<C> operator-(const vec<C> &lhs, const vec<C> &rhs){
	auto res = lhs;
	res -= rhs;
	return res;
}

template<typename C, typename T>
inline vec<C> operator*(const vec<C> &v, const T &val){
	auto res = v;
	res *= val;
	return res;
}
template<typename C, typename T>
inline vec<C> operator*(const T &val, const vec<C> &v){
	return v*val;
}

template<typename C, typename T>
inline vec<C> operator/(const vec<C> &v, const T &val){
	auto res = v;
	res /= val;
	return res;
}


template<typename T>
class shared_vector
{
	using cm = ANN::custom<typename ANN::lookup_custom_tag<T>::type>;
	using alloc_t = shared_allocator<typename cm::alloc<T>>;
	using ptr_t = typename alloc_t::pointer;
	ptr_t storage;
	size_t len, cap;
	alloc_t alloc;
	static const constexpr float over_resv = 1.6;

	template<typename V>
	void push_back_impl(V &&value){
		if(len<cap){
			new(end()) T(std::forward<V>(value));
			len++;
			return;
		}
		size_t cap_new = (len+1)*over_resv;
		ptr_t s_new = alloc.allocate(cap_new);
		T *p = s_new.operator->();
		for(auto it=begin(); it!=end(); ++it, ++p)
			new(p) T(*it);
		new(p) T(std::forward<V>(value));
		len++;
		if(storage!=nullptr)
			alloc.deallocate(storage, cap);
		storage = std::move(s_new);
		cap = cap_new;
	}
public:
	typedef ptr_t pointer;
	typedef T value_type;
	typedef T* iterator;
	typedef const T* const_iterator;
	typedef T& reference;
	typedef const T& const_reference;
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;

	shared_vector() : len(0), cap(0){
	}
	template<typename Iter>
	shared_vector(Iter start, Iter last) : 
		len(std::distance(start, last)), cap(len*over_resv)
	{
		if(len==0) return;

		storage = alloc.allocate(cap);
		T *p = storage.operator->();
		for(; start!=last; ++start, ++p)
			new(p) T(*start);
	}
	shared_vector(shared_vector&&) noexcept = default;
	shared_vector(const shared_vector&) = default;
	shared_vector& operator=(shared_vector&&) noexcept = default;
	shared_vector& operator=(const shared_vector&) = default;

	T* data() noexcept{
		return storage.operator->();
	}
	const T* data() const noexcept{
		return storage.operator->();
	}
	size_type size() const noexcept{
		return len;
	}
	size_type capacity() const noexcept{
		return cap;
	}
	iterator begin() noexcept{
		return data();
	}
	const_iterator begin() const noexcept{
		return data();
	}
	iterator end() noexcept{
		return data()+len;
	}
	const_iterator end() const noexcept{
		return data()+len;
	}
	reference operator[](size_type pos){
		return data()[pos];
	}
	const_reference operator[](size_type pos) const{
		return data()[pos];
	}
	void push_back(const T &value){
		push_back_impl(value);
	}
	void push_back(T &&value){
		push_back_impl(std::move(value));
	}
	void clear(){
		if(storage!=nullptr)
			alloc.deallocate(storage, cap);
		len = cap = 0;
	}

	~shared_vector(){
		clear();
	}
};

} // namespace ANN::util

#endif // _ANN_UTIL_VEC_HPP
