/*
	Modified from parlaylib/examples/hash_map.h (struct hash_map) and 
	parlaylib/include/parlay/utilities.h (parlay::hash)
*/

#ifndef _ANN_UTIL_HASHMAP_HPP
#define _ANN_UTIL_HASHMAP_HPP

#include <cstdint>
#include <functional>
#include <utility>
#include <algorithm>
#include <atomic>
#include <optional>
#include <limits>
#include <type_traits>
#include <ranges>
#include "custom/custom.hpp"
#include "util/intrin.hpp"

namespace ANN::util{

namespace detail{

template<typename T, typename=void>
struct hash : public std::hash<T>{};

template <typename T>
struct hash<T, std::enable_if_t<std::is_integral_v<T>>>{
	size_t operator()(const T &p) const{
		return p * size_t(0xbf58476d1ce4e5b9);
	}
};

}

template<
	typename K, typename V,
	typename Hash = detail::hash<K>,
	typename Equal = std::equal_to<>
>
class hash_map
{
private:
	using idx_t = size_t;

	using cm = custom<typename lookup_custom_tag<hash_map>::type>;
	template<typename T>
	using seq = typename cm::seq<T>;

public:
	using key_type = K;
	using value_type = V;
	using kv_type = std::pair<const K,V>;
	using reference = value_type&;
	using const_reference = const value_type&;
	using pointer = std::optional<reference>;
	using const_pointer = std::optional<const_reference>;
	using kvref = kv_type&;
	using const_kvref = const kv_type&;
	using erase_return_type = std::optional<V>;
	using pair_type = std::pair<K,V>;

private:
	enum class state : char {
		empty, full, locked, tomb
	};
	struct entry{
		union{
			kv_type data;
			char uninitialized[1];
		};
		std::atomic<state> status;
		/*
		entry() requires (!std::is_arithmetic_v<key_type>) :
			status(state::empty){
		}
		entry() requires (std::is_arithmetic_v<key_type>) :
			data(std::numeric_limits<key_type>::max(), {}),
			status(state::empty){
		}
		*/
		entry() : status(state::empty){
		}
		~entry(){
			assert(status!=state::locked);
			if(status!=state::empty)
				data.~kv_type();
		}
	};

	idx_t m;
	Hash hash;
	Equal equal;
	seq<entry> H;

	idx_t first_index(const key_type &k) const noexcept{
		return (hash(k)+idx_t(this)) % m;
	}
	idx_t next_index(idx_t h) const noexcept{
		return (h+1==m)? 0: h+1;
	}

	// use explicit this deduction in C++23 to simplify the code
	seq<const entry*> entry_pointers() const
	{
		auto pts = H | std::views::transform(
			[](const entry &e){return &e;}
		);
		auto p_full = [](const entry *pe){
			return pe->status==state::full;
		};
		return cm::filter(pts, p_full);
	}
	seq<entry*> entry_pointers()
	{
		auto pts = H | std::views::transform(
			[](entry &e){return &e;}
		);
		auto p_full = [](const entry *pe){
			return pe->status==state::full;
		};
		return cm::filter(pts, p_full);
	}

public:
	hash_map(size_t size, Hash hash={}, Equal equal={}) :
		m(200 + static_cast<idx_t>(2.5*size)),
		hash(std::move(hash)), equal(std::move(equal)),
		H(seq<entry>(m))
	{
	}

	// Safe to set `strong`=false if the only concurrent operation
	// is `insert` and there is no possibly duplicated keys
	bool insert(const key_type &k, V v, bool strong=true)
	{
		state trans = strong? state::locked: state::full;
		idx_t i = first_index(k);
		size_t cnt_slots = 0;
		while(true)
		{
			state status = H[i].status;
			if(status==state::full || status==state::tomb)
			{
				if(equal(H[i].data.first,k))
				{
					if(status==state::full)
					{
						if(strong) return false;
					}

					if(H[i].status.compare_exchange_strong(status,trans))
					{
						H[i].data.second = std::move(v);
						if(strong) H[i].status = state::full;
						return true;
					}
				}
				else
				{
					assert(++cnt_slots<std::min<size_t>(500,m));
					i = next_index(i);
				}
			}
			else if(status==state::empty)
			{
				if(H[i].status.compare_exchange_strong(status,trans))
				{
					new(&H[i].data) kv_type(k, std::move(v));
					if(strong) H[i].status = state::full;
					return true;
				}
			}
		}
	}

	const_pointer find(const key_type& k) const noexcept
	{
		idx_t i = first_index(k);
		while(true)
		{
			if(H[i].status!=state::full)
				return {};

			if(equal(H[i].data.first,k))
				return H[i].data.second;

			i = next_index(i);
		}
	}
	pointer find(const key_type& k) noexcept
	{
		auto res = std::as_const(*this).find(k);
		return res? const_cast<V&>(*res): std::nullopt;
	}

	erase_return_type erase(const key_type& k)
	{
		idx_t i = first_index(k);
		while(true)
		{
			if(H[i].status==state::empty || H[i].status==state::locked)
				return {};

			if(equal(H[i].data.first,k))
			{
				state old = state::full;
				if(H[i].status==state::full &&
					H[i].status.compare_exchange_strong(old,state::tomb))
				{
					return std::move(H[i].data.second);
				}
				return {};
			}
			i = next_index(i);
		}
	}

	auto keys() const& //-> sequence of const key_type&
	{
		std::ranges::owning_view pts(entry_pointers());
		auto to_key = std::views::transform(
			[](const entry *pe) -> const key_type&{
				return pe->data.first;
		});
		return std::move(pts) | to_key;
	}
	seq<key_type> keys() &&
	{
		auto keys = entry_pointers() | 
			std::views::transform([](entry *pe) -> key_type&&{
				return std::move(pe->data.first);
			});
		return util::to<seq<key_type>>(keys);
	}

	// use explicit this deduction in C++23 to simplify the code
	auto values() const& //-> sequence of const value_type&
	{
		std::ranges::owning_view pts(entry_pointers());
		auto to_value = std::views::transform(
			[](const entry *pe) -> const value_type&{
				return pe->data.second;
		});
		return std::move(pts) | to_value;
	}
	auto values() & //-> sequence of value_type&
	{
		std::ranges::owning_view pts(entry_pointers());
		auto to_value = std::views::transform(
			[](entry *pe) -> value_type&{
				return pe->data.second;
		});
		return std::move(pts) | to_value;
	}
	seq<value_type> values() &&
	{
		auto values = entry_pointers() | 
			std::views::transform([](entry *pe) -> value_type&&{
				return std::move(pe->data.second);
			});
		return util::to<seq<value_type>>(values);
	}

	auto kvs() const& //->sequence of const_kvref
	{
		std::ranges::owning_view pts(entry_pointers());
		auto to_kv = std::views::transform(
			[](const entry *pe) -> const_kvref{
				return *pe;
		});
		return std::move(pts) | to_kv;
	}
	auto kvs() & //->sequence of kvref
	{
		std::ranges::owning_view pts(entry_pointers());
		auto to_kv = std::views::transform(
			[](entry *pe) -> kvref{
				return *pe;
		});
		return std::move(pts) | to_kv;
	}
	seq<pair_type> kvs() &&
	{
		auto kvs = entry_pointers() | 
			std::views::transform([](entry *pe) -> kv_type&&{
				return std::move(pe->data);
			});
		return util::to<seq<pair_type>>(kvs);
	}

	size_t size() const
	{
		return cm::reduce(H | std::views::transform([](const auto &e){
			return size_t(e.status==state::full);
		}));
	}
};

} // namespace ANN::util

#endif // _ANN_UTIL_HASHMAP_HPP
