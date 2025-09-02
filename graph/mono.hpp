#ifndef _ANN_GRAPH_MONO_HPP
#define _ANN_GRAPH_MONO_HPP

#include <cassert>
#include <algorithm>
#include <utility>
#include <type_traits>
#include <ranges>
#include "graph.hpp"
#include "custom/custom.hpp"
#include "util/util.hpp"
#include "util/intrin.hpp"

namespace ANN{
namespace graph{

template<
	class Nid, class Ext, class Edge,
	template<typename> class TCtrE=detail::seq_default
>
class mono : public base
{
public:
	using nid_t = Nid;
	using deg_t = uint32_t;
	static_assert(sizeof(Ext)>=sizeof(deg_t));

protected:
	using cm = custom<typename lookup_custom_tag<Nid>::type>;
	using edgelist = TCtrE<Edge>;

	template<typename T>
	using seq = typename cm::seq<T>;

	template<class PExt>
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
		using data_t = PExt;
		nid_t id;
		data_t raw;
		ptr_base(nid_t id, data_t raw) : id(id), raw(raw){
		}

		friend class mono;
	};

public:
	struct node_ptr : ptr_base<Ext*>{
		using ptr_base<Ext*>::ptr_base;
	};

	struct node_cptr : ptr_base<const Ext*>{
		using ptr_base<const Ext*>::ptr_base;
		node_cptr(const node_ptr &other) :
			ptr_base<const Ext*>(other.id, other.raw){
		}
	};

	template<typename Iter>
	class agent_t : public std::ranges::subrange<Iter>
	{
		using _b = std::ranges::subrange<Iter>;
		static const constexpr bool is_const = 
			std::is_const_v<std::remove_pointer_t<Iter>>;

		deg_t deg_limit;

	public:
		static const constexpr bool is_agent = true;

		agent_t(): _b(), deg_limit(0){
		}
		agent_t(Iter begin, Iter end, uint32_t deg_limit):
			_b(begin,end), deg_limit(deg_limit){
		}
		/*
		template<typename C>
		C to() const{
			return util::to<C>(static_cast<const _b&>(*this));
		}
		*/
		template<typename U>
		agent_t& operator=(U &&es) requires(!is_const){
			auto *pe = this->data();
			deg_t &deg = *(deg_t*)&pe[-1];
			deg = std::ranges::size(es);
			assert(deg<=deg_limit);

			for(deg_t i=0; i<deg; ++i)
				pe[i] = util::forward_like<U>(es[i]);

			return *this;
		}
	};

protected:
	template<class T>
	node_ptr add_node_impl(nid_t u, T &&ext){
		if(nodes.size()<u+1)
			nodes.resize(u+1);

		nodes[u] = std::forward<T>(ext);
		return get_node(u);
	}

public:
	void reset(uint32_t cnt_nodes, uint32_t cnt_deg){
		nodes.resize(cnt_nodes);
		edges.resize(cnt_nodes*(1+cnt_deg));
		deg_limit = cnt_deg;
	}

	node_ptr get_node(nid_t u){
		assert(u<nodes.size());
		return {u, &nodes[u]};
	}
	node_cptr get_node(nid_t u) const{
		assert(u<nodes.size());
		return {u, &nodes[u]};
	}

	agent_t<const Edge*> get_edges(nid_t u) const{
		const auto *pe = &edges[u*(1+deg_limit)];
		deg_t deg = *(deg_t*)&pe[0];
		return {&pe[1], &pe[1+deg], deg_limit};
	}
	agent_t<const Edge*> get_edges(node_cptr p) const{
		return get_edges(p.get_id());
	}
	agent_t<Edge*> get_edges(nid_t u){
		auto *pe = &edges[u*(1+deg_limit)];
		deg_t deg = *(deg_t*)&pe[0];
		return {&pe[1], &pe[1+deg], deg_limit};
	}
	agent_t<Edge*> get_edges(node_cptr p){
		return get_edges(p.get_id());
	}

	template<class Seq>
	void set_edges(nid_t u, Seq &&es){
		assert(es.size()<=deg_limit);

		auto *pe = &edges[u*(1+deg_limit)];
		deg_t &deg = *(deg_t*)&pe[0];
		deg = std::size(es);

		for(deg_t i=0; i<deg; ++i)
			pe[1+i] = util::forward_like<Seq>(es[i]);
	}
	template<class Seq>
	void set_edges(node_ptr p, Seq&& es){
		set_edges(p.get_id(), std::forward<Seq>(es));
	}
	template<class Seq>
	void set_edges(Seq &&ps){
		if constexpr(requires{ps[0].second.is_agent;})
		{
			(void)ps;
			return;
		}
		else
		{
			const auto n = std::size(ps);
			cm::parallel_for(0, n, [&](size_t i){
				auto &&[u,es] = util::forward_like<Seq>(ps[i]);
				set_edges(u, std::forward<decltype(es)>(es));
			});
		}
	}

	node_ptr add_node(nid_t u, const Ext &ext){
		actual_num++;
		return add_node_impl(u, ext);
	}
	node_ptr add_node(nid_t u, Ext &&ext){
		actual_num++;
		return add_node_impl(u, std::move(ext));
	}
	template<class Seq>
	void add_nodes(Seq &&ns){
		const auto n = std::size(ns);
		actual_num += n;
		/*auto nids = ns | std::views::transform([](const auto &p){
			return std::get<0>(p);
		});*/
		auto nids = util::delayed_seq(n, [&](size_t i){
			return std::get<0>(ns[i]);
		});

		nid_t nid_max = *cm::max_element(nids.begin(), nids.end());
		if(nodes.size()<nid_max+1)
			nodes.resize(nid_max+1);

		cm::parallel_for(0, n, [&](size_t i){
			nodes[nids[i]] = util::forward_like<Seq>(std::get<1>(ns[i]));
		});
	}

	// void remove_node(nid_t);
	template<class Seq>
	void remove_nodes(const Seq &nids){
		const auto n = std::size(nids);
		actual_num -= n;
		cm::parallel_for(0, n, [&](size_t i){
			nodes[nids[i]] = Ext();
		});
	}

	size_t num_nodes() const{
		return actual_num;
	}
	bool empty() const{
		return actual_num==0;
	}

	template<class F>
	void iter_each(F &&f) const{
		for(size_t i=0; i<nodes.size(); ++i)
			f(node_cptr(i, &nodes[i]));
	}
	template<class F>
	void iter_each(F &&f){
		for(size_t i=0; i<nodes.size(); ++i)
			f(node_ptr(i, &nodes[i]));
	}
	// TODO: eliminate redundant code by deducing 'this' in C++23
	template<class F>
	void for_each(F &&f) const{
		cm::parallel_for(0, nodes.size(),
			[&](size_t i){
				f(node_cptr(i, &nodes[i]));
			}
		);
	}
	template<class F>
	void for_each(F &&f){
		cm::parallel_for(0, nodes.size(),
			[&](size_t i){
				f(node_ptr(i, &nodes[i]));
			}
		);
	}

protected:
	uint32_t actual_num = 0;
	seq<Ext> nodes;
	edgelist edges;
	uint32_t deg_limit;
};

} // namespace graph
} // namesapce ANN

#endif // _ANN_GRAPH_MONO_HPP