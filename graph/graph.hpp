#ifndef _ANN_GRAPH_HPP
#define _ANN_GRAPH_HPP

#include <cstdint>
#include <fstream>
#include <utility>
#include <iterator>
#include <algorithm>
#include <type_traits>
#include <memory>
#include <ranges>
#include "ANN.hpp"
#include "custom/custom.hpp"
#include "util/intrin.hpp"
#include "util/seq.hpp"
#include "custom/custom.hpp"
#include "util/hash_map.hpp"

namespace ANN{
namespace graph{

class base
{
	struct node;
public:
	using nid_t = uint32_t;
	struct node_ptr;
	struct node_cptr;
	// TODO: update interfaces
	node_ptr get_node(nid_t);
	node_cptr get_node(nid_t) const;
	template<class Seq>
	Seq get_edges(node_cptr) const;
	template<class Seq>
	Seq get_edges(nid_t) const;
	template<class Seq>
	void set_edges(nid_t, Seq&&);
	void add_node(nid_t);
	template<class Seq>
	void add_nodes(Seq&&);
	size_t num_nodes() const;
	void remove_node(nid_t);
	template<typename Iter>
	void remove_nodes(Iter, Iter);
};

template<class>
class shim;

enum class file_format{
	BUILTIN
};

namespace detail{

template<class T>
const static bool constexpr is_graph = std::is_base_of_v<base,T>;
/*
template<class T>
concept Graph = is_graph<T>
*/

template<typename T, class L=lookup_custom_tag<>>
struct seq_default_impl : custom<typename L::type>::seq<T>{
	using _base = typename custom<typename L::type>::template seq<T>;
	using _base::_base;
	using _base::operator=;

	seq_default_impl(const _base &b) : _base(b){}
	seq_default_impl(_base &&b) : _base(std::move(b)){}
};

template<typename T>
using seq_default = seq_default_impl<T>;

template<typename Key, typename T>
using map_seq = seq_default_impl<T>;

template<typename Key, typename T>
struct map_default : std::unordered_map<Key,T>{
	using _base = std::unordered_map<Key,T>;
	using _base::unordered_map;
	using _base::operator=;
	using _base::operator[];

	const T& operator[](const Key &key) const{
		auto it = _base::find(key);
		// TODO: match `Key' type
		// assert(it!=_base::end() || (debug_output("key=%u\n",key),false));
		assert(it!=_base::end());
		return it->second;
	}
};

} // namespace detail

template<class G, class R, class F, class E>
G load(R &reader, const F &vtx, const E &edge)
{
	using cm = custom<typename lookup_custom_tag<G>::type>;
	using nid_t = typename G::nid_t;
	//using cnt_t = uint32_t;
	using util::delayed_seq;
	using std::views::transform;

	uint64_t meta = reader();
	assert((meta&0x0000ff)==0x05);
	assert((meta&0x00ff00)>>8==0x00);
	//assert((meta&0xff0000)>>16==sizeof(cnt_t));
	size_t cnt_t_size = (meta&0xff0000)>>16;
	assert(cnt_t_size==4 || cnt_t_size==8);

	auto do_read = [&](auto type_v){
		using cnt_t = decltype(type_v);

		cnt_t num_nodes = reader();
		typename cm::seq<nid_t> ids = reader(num_nodes);
		typename cm::seq<cnt_t> offset = reader(num_nodes+1);
		// TODO: use view-like return value to prevent dup mem
		auto sizes = delayed_seq(num_nodes, [&](size_t i){
				return offset[i+1] - offset[i];
		});
		typename cm::seq<nid_t> els = reader(offset.back());

		G g;
		g.add_nodes(ids | transform([&](nid_t u){
			return std::pair(u, vtx(u));}
		));
		auto es = delayed_seq(num_nodes, [&](size_t i){
			return std::pair(ids[i], delayed_seq(sizes[i], [&,i](size_t j){
				return edge(ids[i], els[j+offset[i]]);
			}));
		});
		g.set_edges(es);

		return g;
	};

	return cnt_t_size==4? do_read(uint32_t{}): do_read(uint64_t{});
}

template<class L=lookup_custom_tag<>, class G, class W>
void save(const G &g, W &writer)
{
	static_assert(detail::is_graph<G>);
	using cm = custom<typename L::type>;
	using nid_t = typename G::nid_t;
	using cnt_t = uint64_t;
	using cptr_t = typename G::node_cptr;
	using std::views::transform;

	auto num_nodes = g.num_nodes();
	assert(std::in_range<cnt_t>(num_nodes));

	util::hash_map<nid_t, cptr_t> collection(num_nodes);
	g.for_each([&](auto p){
		collection.insert(p.get_id(), p);
	});
	auto nps = std::move(collection).values();

	// sizeof(cnt_t)=8, default format, version 5
	writer(uint64_t(0x08'00'05));

	writer(cnt_t(num_nodes));

	auto ids = nps | transform(
		[&](const auto &p){return p.get_id();
	});
	writer(ids);

	using ea_t = decltype(g.get_edges(nid_t()));
	auto els = util::to<typename cm::seq<ea_t>>(
		nps | transform([&](const auto &p){
			return g.get_edges(p);
	}));
	auto [offset,tot] = cm::scan(els | 
		transform([&](const auto &el){
			return cnt_t(el.size());
	}));
	offset.push_back(tot);
	writer(offset);

	writer(els | transform([&](const auto &el){
		return el | transform([](const auto &e){
			return e.u;
		});
	}));
/*
	const size_t BUFFER_SIZE = cm::num_workers()*1024*1024;
	auto buffer = std::make_unique<nid_t[]>(BUFFER_SIZE);
	size_t i = 0;
	while(i+1<offset.size())
	{
		size_t j = std::prev(std::upper_bound(
			offset.begin(), offset.end(), 
			offset[i]+BUFFER_SIZE
		)) - offset.begin();

		if(i==j)
		{
			size_t c = 0;
			util::iter_each(g.get_edges(nps[i]), [&](const auto &conn){
				buffer[c++] = conn.u;
				if(c==BUFFER_SIZE)
					writer(buffer.get(), BUFFER_SIZE);
			});
			writer(buffer.get(), c);
			j++;
		}
		else
		{
			cm::parallel_for(i, j, [&](size_t k){
				size_t c = offset[k] - offset[i];
				util::iter_each(g.get_edges(nps[k]), [&](const auto &conn){
					buffer[c++] = conn.u;
				});
			});
			writer(buffer.get(), (offset[j]-offset[i]));
		}

		i = j;
	}
*/
}

template<class G>
G load(const std::string &name, file_format type)
{
	static_assert(detail::is_graph<G>);
}

template<class To, class From>
To cast(From &&g)
{
	static_assert(detail::is_graph<From>&&detail::is_graph<To>);
}

} // namespace graph
} // namespace ANN

#endif // _ANN_GRAPH_HPP
