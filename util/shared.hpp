#ifndef _ANN_UTIL_SHARED_HPP
#define _ANN_UTIL_SHARED_HPP

#include <utility>
#include <memory>
#include <type_traits>
#include "util/util.hpp"

namespace ANN::util{

// Wrap a (fancy) pointer type UPtr to meet the requirements of A::pointer
// where A is an Allocator following the C++ named requirements
template<class UPtr>
class ptr_adapter : 
	public UPtr,
	public ANN::util::enable_rand_iter<
		ptr_adapter<UPtr>, size_t, ANN::util::empty>
{
	using base_ptr = UPtr;
	using base_iter = ANN::util::enable_rand_iter<
		ptr_adapter<UPtr>, size_t, ANN::util::empty
	>;
	using T = typename UPtr::element_type;
	using inner_t = ANN::util::inner_t;

public:
	typedef T value_type;
	typedef T* pointer;
	typedef T& reference;

	ptr_adapter() : base_ptr(), base_iter(0){
	}
	ptr_adapter(const ptr_adapter&) = default;
	ptr_adapter(ptr_adapter&&) = default;
	template<typename ...Args>
	ptr_adapter(inner_t, size_t offset, Args &&...args) :
		base_ptr(std::forward<Args>(args)...),
		base_iter(offset){
	}
	template<typename A, typename ...Args, typename=std::enable_if_t<!std::is_same_v<
		std::remove_cv_t<std::remove_reference_t<A>>,
		ptr_adapter<typename std::pointer_traits<UPtr>::template rebind<std::remove_cv_t<T>>>
	>>>
	ptr_adapter(A &&a, Args &&...args) : 
		ptr_adapter(inner_t{}, size_t(0), std::forward<A>(a), std::forward<Args>(args)...){
	}
	ptr_adapter& operator=(const ptr_adapter&) = default;
	ptr_adapter& operator=(ptr_adapter&&) = default;
	/*
	ptr_adapter(UPtr p) : 
		base_ptr(std::move(p)),
		base_iter(0){
	}
	*/

	pointer operator->() const{
		return base_ptr::operator->()+base_iter::get_pos();
	}
	reference operator*() const{
		return *operator->();
	}
	operator T*() const{
		return operator->();
	}

	// TODO: use the default comparison in C++20
	// bool operator==(const ptr_adapter &rhs) const = default;
	bool operator==(const ptr_adapter &rhs) const{
		return static_cast<const base_ptr&>(*this) ==
			static_cast<const base_ptr&>(rhs) &&
				static_cast<const base_iter&>(*this) ==
			static_cast<const base_iter&>(rhs);
	}
	bool operator!=(const ptr_adapter &rhs) const{
		return !operator==(rhs);
	}
	bool operator==(const nullptr_t &null) const{
		return base_iter::get_pos()==0 &&
			static_cast<const base_ptr&>(*this)==null;
	}
	bool operator!=(const nullptr_t &null) const{
		return !operator==(null);
	}

	template<typename U=T, class CUPtr=std::enable_if_t<
		!std::is_const_v<U>,
		typename std::pointer_traits<UPtr>::template rebind<const U>
	>>
	operator ptr_adapter<CUPtr>() const{
		return {inner_t{}, base_iter::get_pos(), static_cast<const base_ptr&>(*this)};
	}
};


template<class Alloc>
class shared_allocator : public Alloc{
	using at = std::allocator_traits<Alloc>;

	static_assert(std::is_pointer_v<typename at::pointer>);
	using sz_t = typename at::size_type;
	using val_t = typename at::value_type;

	Alloc& base(){
		return static_cast<Alloc&>(*this);
	}
	const Alloc& base() const{
		return static_cast<const Alloc&>(*this);
	}

public:
	using pointer = ptr_adapter<std::shared_ptr<val_t>>;
	using const_pointer = ptr_adapter<std::shared_ptr<const val_t>>;
	using Alloc::Alloc;

	template<typename ...Args>
	pointer allocate(sz_t n, Args ...args){
		return pointer(
			Alloc::allocate(n, std::forward<Args>(args)...),
			[=,this](auto *p){this->Alloc::deallocate(p,n);},
			base()
		);
	}

	void deallocate(pointer &p, sz_t /*n*/){
		p.reset();
	}
	void deallocate(pointer &&/*p*/, sz_t /*n*/){
		// we do nothing on p as it is destroyed after the call
	}

	template<typename U>
	struct rebind{
		using other = shared_allocator<typename at::template rebind_alloc<U>>;
	};

	bool operator==(const shared_allocator &rhs) const{
		return base()==rhs.base();
	}
	bool operator!=(const shared_allocator &rhs) const{
		return base()!=rhs.base();
	}
};

} // namespace ANN::util

#endif // _ANN_UTIL_SHARED_HPP
