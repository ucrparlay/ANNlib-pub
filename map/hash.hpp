#ifndef _ANN_MAP_HASH_HPP
#define _ANN_MAP_HASH_HPP

#include <iterator>
#include <utility>
#include <type_traits>
#include <optional>
#include <unordered_set>
#include <unordered_map>
#include "map.hpp"

namespace ANN::map{

namespace detail{
template<typename Pid, typename Nid, class Hash, class KeyEqual, typename=void, typename=void>
class hash{
public:
	template<typename Iter>
	void insert(Iter begin, Iter end){
		(void)begin, (void)end;
	}
	template<class Ctr>
	void insert(Ctr &&c){
		if constexpr(std::is_rvalue_reference_v<Ctr&&>)
		{
			insert(
				std::make_move_iterator(c.begin()),
				std::make_move_iterator(c.end())
			);
		}
		else insert(c.begin(), c.end());
	}

	Nid insert(const Pid &pid){
		return Nid(pid);
	}
	Nid insert(Pid &&pid){
		return Nid(std::move(pid));
	}

	Pid get_pid(Nid nid) const{
		static_assert(std::is_convertible_v<Nid,Pid>);
		return Pid(nid);
	}
	Nid get_nid(const Pid &pid) const{
		static_assert(std::is_convertible_v<Pid,Nid>);
		return Nid(pid);
	}

	std::optional<Nid> find_nid(const Pid &pid) const{
		return {get_nid(pid)};
	}
};

template<typename Pid, typename Nid, class Hash, class KeyEqual, typename Comm=std::void_t<std::common_type<Pid,Nid>>,
typename=std::enable_if_t<std::is_convertible_v<Pid,Comm>&&std::is_convertible_v<Nid,Comm>>>
class hash<Pid,Nid,Hash,KeyEqual>{
public:
	template<typename Iter>
	void insert(Iter begin, Iter end){
		(void)begin, (void)end;
	}
	template<class Ctr>
	void insert(Ctr &&c){
		if constexpr(std::is_rvalue_reference_v<Ctr&&>)
		{
			insert(
				std::make_move_iterator(c.begin()),
				std::make_move_iterator(c.end())
			);
		}
		else insert(c.begin(), c.end());
	}

	Nid insert(const Pid &pid){
		return Nid(pid);
	}
	Nid insert(Pid &&pid){
		return Nid(std::move(pid));
	}

	Pid get_pid(Nid nid) const{
		static_assert(std::is_convertible_v<Nid,Pid>);
		return Pid(nid);
	}
	Nid get_nid(const Pid &pid) const{
		static_assert(std::is_convertible_v<Pid,Nid>);
		return Nid(pid);
	}

	std::optional<Nid> find_nid(const Pid &pid) const{
		return {get_nid(pid)};
	}
};

} // namespace detail

template<typename Pid, typename Nid, class Hash=std::hash<Pid>, class KeyEqual=std::equal_to<Pid>>
using hash = detail::hash<Pid, Nid, Hash, KeyEqual>;

} // namespace ANN::map

#endif // _ANN_MAP_HASH_HPP
