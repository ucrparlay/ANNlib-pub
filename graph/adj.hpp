#ifndef _ANN_GRAPH_ADJVEC_HPP
#define _ANN_GRAPH_ADJVEC_HPP

#include <unordered_map>
#include <cassert>
#include <iterator>
#include <utility>
#include <type_traits>
#include "graph.hpp"
#include "custom/custom.hpp"
#include "util/util.hpp"
#include "util/intrin.hpp"
#include "util/seq.hpp"
#include "util/debug.hpp"
#include "util/ref.hpp"
using ANN::util::debug_output;

namespace ANN::graph{

template<
	class Nid, class Ext, class Edge,
	template<typename,typename> class TCtrV, 
	template<typename> class TCtrE>
class adj_base : public base
{
public:
	using nid_t = Nid;

protected:
	using cm = custom<typename lookup_custom_tag<Nid>::type>;
	using edgelist = TCtrE<Edge>;
	struct node_t : Ext{
		node_t() = default;
		node_t(const Ext &ext) : Ext(ext){
		}
		node_t(Ext &&ext) : Ext(std::move(ext)){
		}
		using Ext::operator=;

		edgelist neighbors;
	};
	using nodelist = TCtrV<nid_t,node_t>;

private:
	template<bool IsConst>
	struct ptr_base{
		using ptr_t = const Ext*;
		using ref_t = const Ext&;

		ref_t operator*() const{
			return *raw;
		}
		ptr_t operator->() const{
			return raw;
		}

		nid_t get_id() const{
			return id;
		}

	protected:
		using data_t = std::conditional_t<IsConst, const node_t*, node_t*>;
		nid_t id;
		data_t raw;
		ptr_base(nid_t id, data_t raw) : id(id), raw(raw){
		}

		friend class adj_base;
	};

public:
	struct node_ptr : ptr_base<false>{
		using ptr_base<false>::ptr_base;
	};

	struct node_cptr : ptr_base<true>{
		using ptr_base<true>::ptr_base;
		node_cptr(const node_ptr &other) :
			ptr_base<true>(other.id, other.raw){
		}
	};

protected:
	static node_ptr gen_node_ptr(nid_t nid, node_t *p){
		return node_ptr(nid, p);
	}
	static node_cptr gen_node_cptr(nid_t nid, const node_t *p){
		return node_cptr(nid, p);
	}

	template<class T>
	node_ptr set_node_impl(nid_t nid, T &&ext){
		node_ptr p = get_node(nid);
		*(p.raw) = std::forward<T>(ext);
		return p;
	}

	const edgelist& get_edges_impl(node_cptr p, util::dummy<edgelist>) const{
		// return std::ranges::subrange(p.raw->neighbors);
		return p.raw->neighbors;
	}
	template<class Seq>
	const Seq get_edges_impl(node_cptr p, util::dummy<Seq>) const{
		assert(0); // TODO: remove assert(0)
		const auto &origin = get_edges_impl(p, util::dummy<edgelist>{});
		return Seq(origin.begin(), origin.end());
	}

	edgelist pop_edges_impl(node_ptr p, util::dummy<edgelist>){
		edgelist &nbhs = p.raw->neighbors;
		edgelist edges = std::move(nbhs);
		nbhs.clear();
		return edges;
	}
	template<class Seq>
	Seq pop_edges_impl(node_ptr p, util::dummy<Seq>){
		assert(0);
		edgelist &nbhs = p.raw->neighbors;
		// TODO: use `Seq(std::from_range, nbhs)` in C++23
		Seq edges(
			std::make_move_iterator(nbhs.begin()),
			std::make_move_iterator(nbhs.end())
		);
		nbhs.clear();
		return edges;
	}

public:
	node_ptr get_node(nid_t nid){
		return {nid, &nodes[nid]};
	}
	node_cptr get_node(nid_t nid) const{
		return {nid, &nodes[nid]};
	}

	node_ptr set_node(nid_t nid, const Ext &ext){
		return set_node_impl(nid, ext);
	}
	node_ptr set_node(nid_t nid, Ext &&ext){
		return set_node_impl(nid, std::move(ext));
	}
	/*
	template<typename Seq=edgelist>
	decltype(auto) get_edges(node_cptr p) const{
		return get_edges_impl(p, util::dummy<Seq>{});
	}
	template<class Seq=edgelist>
	decltype(auto) get_edges(nid_t nid) const{
		return get_edges<Seq>(get_node(nid));
	}
	*/
	auto get_edges(node_cptr p) const{
		return util::ref_view(std::move(p.raw->neighbors));
	}
	auto get_edges(nid_t nid) const{
		return get_edges(get_node(nid));
	}
	auto get_edges(node_ptr p){
		return util::ref_view(std::move(p.raw->neighbors));
	}
	auto get_edges(nid_t nid){
		return get_edges(get_node(nid));
	}

	template<typename Seq=edgelist>
	Seq pop_edges(node_ptr p){
		return pop_edges_impl(p, util::dummy<Seq>{});
	}
	template<class Seq=edgelist>
	Seq pop_edges(nid_t nid){
		return pop_edges<Seq>(get_node(nid));
	}

	template<class Seq>
	void set_edges(node_ptr p, Seq&& es){
		if constexpr(requires{p.raw->neighbors=std::forward<Seq>(es);}){
			p.raw->neighbors = std::forward<Seq>(es);
		}
		else{
			assert(0);
			p.raw->neighbors = edgelist(es.begin(),es.end());
		}
	}
	template<class Iter>
	void set_edges(Iter begin, Iter end){
		// TODO: forward to `set_edges(Seq&&)` using `subrange` in C++20
		const auto n = std::distance(begin, end);
		cm::parallel_for(0, n, [&](size_t i){
			auto &&[nid,es] = *(begin+i);
			get_node(nid).raw->neighbors = 
				util::to<edgelist>(std::forward<decltype(es)>(es));
		});
	}
	template<class Seq>
	void set_edges(Seq&& ps){
		if constexpr(std::is_rvalue_reference_v<Seq&&>)
		{
			// TODO: use `util::for_each`
			set_edges(
				std::make_move_iterator(ps.begin()),
				std::make_move_iterator(ps.end())
			);
		}
		else set_edges(ps.begin(), ps.end());
	}

	template<class Seq>
	void set_edges(nid_t nid, Seq&& es){
		set_edges(get_node(nid), std::forward<Seq>(es));
	}

	size_t num_nodes() const{
		return nodes.size();
	}
	bool empty() const{
		return nodes.empty();
	}

protected:
	nodelist nodes;
};


template<
	class Nid, class Ext, class Edge=Nid,
	template<typename,typename> class TMapV=detail::map_seq,
	template<typename> class TSeqE=detail::seq_default>
class adj_seq : public adj_base<Nid,Ext,Edge,TMapV,TSeqE>
{
	static_assert(std::is_default_constructible_v<Ext>);

	using _base = adj_base<Nid,Ext,Edge,TMapV,TSeqE>;

	using typename _base::cm;
	using _base::nodes;
	using _base::set_node_impl;

public:
	using typename _base::nid_t;
	using typename _base::node_ptr;
	using _base::gen_node_ptr;
	using _base::gen_node_cptr;
	// using _base::get_node;

private:
	template<class T>
	node_ptr add_node_impl(nid_t nid, T &&ext){
		if(nodes.size()<nid+1)
			nodes.resize(nid+1);
		return set_node_impl(nid, std::forward<T>(ext));
	}

public:
	auto get_node(nid_t nid){
		assert(nid<nodes.size());
		return _base::get_node(nid);
	}
	auto get_node(nid_t nid) const{
		assert(nid<nodes.size());
		return _base::get_node(nid);
	}
	node_ptr add_node(nid_t nid, const Ext &ext){
		return add_node_impl(nid, ext);
	}
	node_ptr add_node(nid_t nid, Ext &&ext){
		return add_node_impl(nid, std::move(ext));
	}
	template<class Iter>
	void add_nodes(Iter begin, Iter end){
		// TODO: forward to `add_nodes(Seq&&)` using `subrange` in C++20
		const auto n = std::distance(begin, end);
		auto nids = util::delayed_seq(n, [&](size_t i){
			return (nid_t)std::get<0>(*(begin+i));
		});
		nid_t nid_max = *cm::max_element(nids.begin(), nids.end());
		if(nodes.size()<nid_max+1)
			nodes.resize(nid_max+1);

		cm::parallel_for(0, n, [&](size_t i){
			// TODO: avoid using nodes
			nodes[nids[i]] = std::get<1>(*(begin+i));
		});
	}
	template<class Seq>
	void add_nodes(Seq&& ns){
		if constexpr(std::is_rvalue_reference_v<Seq&&>)
		{
			// TODO: use `util::for_each`
			add_nodes(
				std::make_move_iterator(ns.begin()),
				std::make_move_iterator(ns.end())
			);
		}
		else add_nodes(ns.begin(), ns.end());
	}

	template<class Iter>
	void remove_nodes(Iter begin, Iter end){
		const auto n = std::distance(begin, end);
		cm::parallel_for(0, n, [&](size_t i){
			nodes[*(begin+i)] = typename decltype(nodes)::value_type{};
		});
	}

	template<class F>
	void iter_each(F &&f) const{
		for(size_t i=0, n=nodes.size(); i<n; ++i)
			f(gen_node_cptr(i, &nodes[i]));
	}
	template<class F>
	void iter_each(F &&f){
		for(size_t i=0, n=nodes.size(); i<n; ++i)
			f(gen_node_ptr(i, &nodes[i]));
	}
	// TODO: eliminate redundant code by deducing 'this' in C++23
	template<class F>
	void for_each(F &&f) const{
		cm::parallel_for(0, nodes.size(),
			[&,it=nodes.begin()](size_t i){
				f(gen_node_cptr(i, &it[i]));
			}
		);
	}
	template<class F>
	void for_each(F &&f){
		cm::parallel_for(0, nodes.size(),
			[&,it=nodes.begin()](size_t i){
				f(gen_node_ptr(i, &it[i]));
			}
		);
	}
};

template<
	class Nid, class Ext, class Edge=Nid,
	template<typename,typename> class TMapV=detail::map_default,
	template<typename> class TSeqE=detail::seq_default>
class adj_map : public adj_base<Nid,Ext,Edge,TMapV,TSeqE>
{
	using _base = adj_base<Nid,Ext,Edge,TMapV,TSeqE>;
	using typename _base::cm;
	using typename _base::node_t;
	using typename _base::nodelist;
	using _base::nodes;
	using _base::set_node_impl;

public:
	using typename _base::nid_t;
	using typename _base::node_ptr;
	using _base::gen_node_ptr;
	using _base::gen_node_cptr;
	using _base::get_node;
	using _base::num_nodes;

private:
	template<class T>
	node_ptr add_node_impl(nid_t nid, T &&ext){
		return set_node_impl(nid, std::forward<T>(ext));
	}

public:
	node_ptr add_node(nid_t nid, const Ext &ext){
		return add_node_impl(nid, ext);
	}
	node_ptr add_node(nid_t nid, Ext &&ext){
		return add_node_impl(nid, std::move(ext));
	}
	template<class Iter>
	void add_nodes(Iter begin, Iter end){
		// TODO: forward to `add_nodes(Seq&&)` using `subrange` in C++20
		nodes.insert(begin, end);
	}
	template<class Seq>
	void add_nodes(Seq&& ns){
		if constexpr(std::is_rvalue_reference_v<Seq&&>)
		{
			// TODO: use `util::for_each`
			add_nodes(
				std::make_move_iterator(ns.begin()),
				std::make_move_iterator(ns.end())
			);
		}
		else add_nodes(ns.begin(), ns.end());
	}

	template<class Iter>
	void remove_nodes(Iter begin, Iter end){
		for(auto it=begin; it!=end; ++it)
			nodes.erase(*it);
	}

	// TODO: eliminate redundant code by deducing 'this' in C++23
	template<class F>
	void iter_each(F &&f) const{
		util::iter_each(nodes, [&](const auto &p){
			f(gen_node_cptr(p.first, &p.second));
		});
	}
	template<class F>
	void iter_each(F &&f){
		util::iter_each(nodes, [&](auto &p){
			f(gen_node_ptr(p.first, &p.second));
		});
	}
	template<class F>
	void for_each(F &&f) const{
		/*
		util::for_each(nodes, [&](const auto &p){
			f(gen_node_cptr(p.first, &p.second));
		});
		*/
		typename cm::seq<typename nodelist::const_iterator> iters;
		iters.reserve(num_nodes());
		for(auto it=nodes.cbegin(); it!=nodes.cend(); ++it)
			iters.push_back(it);
		util::for_each(iters, [&](const auto &it){
			f(gen_node_cptr(it->first, &it->second));
		});
	}
	template<class F>
	void for_each(F &&f){
		/*
		util::for_each(nodes, [&](auto &p){
			f(gen_node_ptr(p.first, &p.second));
		});
		*/
		typename cm::seq<typename nodelist::iterator> iters;
		iters.reserve(num_nodes());
		for(auto it=nodes.begin(); it!=nodes.end(); ++it)
			iters.push_back(it);
		util::for_each(iters, [&](const auto &it){
			f(gen_node_ptr(it->first, &it->second));
		});
	}
};

} // namespace ANN::graph

#endif // _ANN_GRAPH_ADJVEC_HPP
