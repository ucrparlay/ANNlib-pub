#ifndef _ANN_UTIL_REF_HPP
#define _ANN_UTIL_REF_HPP

#include <utility>
#include <functional>
#include <type_traits>
#include <ranges>
#include "util/intrin.hpp"

namespace ANN{
namespace util{

template<typename RT>
class ref_view :
	public std::ranges::ref_view<std::remove_reference_t<RT>>
{
	using _b = std::ranges::ref_view<std::remove_reference_t<RT>>;
	// static const constexpr bool is_rval_ref = std::is_rvalue_reference_v<RT>;
	using base_t = std::remove_cvref_t<RT>;
	static const constexpr bool is_const = 
		std::is_const_v<std::remove_reference_t<RT>>;

	static base_t empty;

public:
	ref_view(): _b(empty){
	}
	ref_view(RT ref): _b(ref){
		static_assert(std::is_reference_v<RT>);
	}

	template<typename C>
	C to() && requires(std::is_rvalue_reference_v<RT>){
		return util::to<C>(static_cast<RT>(this->base()));
	}
	template<typename C>
	C to(){
		return util::to<C>(this->base());
	}
	template<typename C>
	C to() const{
		return util::to<C>(std::as_const(this->base()));
	}

	template<typename U>
	ref_view& operator=(U &&value){
		this->base() = std::forward<U>(value);
		return *this;
	}
};

template<typename T>
ref_view(T&&) -> ref_view<T&&>;

template<typename RT>
ref_view<RT>::base_t ref_view<RT>::empty{};


} // namespace util
} // namespace ANN

#endif // _ANN_UTIL_REF_HPP
