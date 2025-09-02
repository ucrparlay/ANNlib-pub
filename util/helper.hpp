#ifndef _ANN_UTIL_HELPER_HPP
#define _ANN_UTIL_HELPER_HPP

#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <ranges>
#include "custom/custom.hpp"
#include "util/intrin.hpp"
#include "util/seq.hpp"

namespace ANN::util{

template<class Seq, class L=lookup_custom_tag<>>
auto pack_index(Seq &&seq)
{
	using cm = custom<typename L::type>;
	// return cm::pack_index(std::forward<Seq>(seq));
	// TODO: fix
	return cm::pack_index(std::forward<Seq>(seq));
}

template<class Seq, class L=lookup_custom_tag<>>
auto flatten(Seq &&seq)
{
	using cm = custom<typename L::type>;
	return cm::flatten(std::forward<Seq>(seq));
}

template<class Seq, class L=lookup_custom_tag<>>
auto group_by_key(Seq &&seq)
{
	using cm = custom<typename L::type>;
	return cm::group_by_key(std::forward<Seq>(seq));
}

template<typename Seq, class F, class L=lookup_custom_tag<>>
auto filter(Seq &&seq, F &&f)
{
	using cm = custom<typename L::type>;
	return cm::filter(std::forward<Seq>(seq), std::forward<F>(f));
}

template<class SR, typename S1, typename S2>
SR zip(S1 &&s1, S2 &&s2)
{
	size_t n = std::min<size_t>(s1.size(), s2.size());
	auto gen = delayed_seq(n, [&](size_t i){
		return typename SR::value_type{
			std::is_rvalue_reference_v<S1&&>? std::move(s1[i]): s1[i],
			std::is_rvalue_reference_v<S2&&>? std::move(s2[i]): s2[i]
		};
	});
	return SR(gen.begin(), gen.end());
}

template<class C, class F>
C init(size_t n, F f)
{
	using ret_t = std::invoke_result_t<F, size_t>;
	using value_t = std::ranges::range_value_t<C>;
	static_assert(std::is_convertible_v<ret_t, value_t>);
	return to<C>(delayed_seq(n, f));
}

} // namespace ANN::util

#endif // _ANN_UTIL_HELPER_HPP
