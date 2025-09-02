#ifndef _ANN_GRAPH_CHRONO_HPP
#define _ANN_GRAPH_CHRONO_HPP

#include <cstddef>
#include <cassert>
#include <algorithm>
#include <utility>
#include <list>
#include <memory>
#include <type_traits>
#include <ranges>
#include "graph.hpp"
#include "custom/custom.hpp"
#include "util/util.hpp"
#include "util/ref.hpp"
#include "util/seq.hpp"
#include "util/vec.hpp"

#include <cstdio>

namespace ANN::graph{

namespace detail{

template<
	class Nid, class Ext, class Edge,
	template<typename> class TCtrE
>
class chrono_list_base : public base
{
public:
	using nid_t = Nid;

protected:
	using epoch_t = uint32_t;
	using cm = custom<typename lookup_custom_tag<Nid>::type>;
	//using edgelist = typename cm::seq<Edge>;
	using edgelist = TCtrE<Edge>;

	struct version_t{
		epoch_t epoch;
		edgelist edges;
	};

	struct node_t : Ext{
		node_t() = default;
		node_t(const Ext &ext) : Ext(ext){
		}
		node_t(Ext &&ext) : Ext(std::move(ext)){
		}
		using Ext::operator=;

		std::list<version_t, typename cm::alloc<version_t>> neighbors;
	};
	using nodelist = typename cm::seq<node_t>;

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

		friend class chrono_list_base;
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
	const version_t* get_edges_impl(node_cptr p) const{
		for(const version_t &ver : p.raw->neighbors)
		{
			if(ver.epoch<=epoch)
				return &ver;
		}
		return nullptr;
		// static const edgelist empty;
		// return &empty;
	}

	template<class T>
	node_ptr add_node_impl(nid_t nid, T &&ext){
		if(!nodes)
			nodes = std::make_shared<nodelist>();
		if(nodes->size()<nid+1)
			nodes->resize(nid+1);
		node_ptr p = get_node(nid);
		*(p.raw) = std::forward<T>(ext);
		p.raw->neighbors.push_front({epoch,{}});
		return p;
	}

public:
	node_ptr get_node(nid_t nid){
		assert(nid<nodes->size());
		return {nid, &(*nodes)[nid]};
	}
	node_cptr get_node(nid_t nid) const{
		assert(nid<nodes->size());
		return const_cast<chrono_list_base*>(this)->get_node(nid);
	}

	auto get_edges(node_cptr p) const{
		const version_t *ptr = get_edges_impl(p);
		return util::ref_view(ptr->edges);
	}
	auto get_edges(nid_t nid) const{
		return get_edges(get_node(nid));
	}
	auto get_edges(node_ptr p){
		version_t *ptr = const_cast<version_t*>(get_edges_impl(p));
		// make sure we have the ownership of the returned version
		if(ptr->epoch!=epoch)
		{
			p.raw->neighbors.emplace_front(epoch, ptr->edges);
			ptr = &p.raw->neighbors.front();
		}
		return util::ref_view(std::move(ptr->edges));
	}
	auto get_edges(nid_t nid){
		return get_edges(get_node(nid));
	}

	template<class Seq>
	void set_edges(Seq &&ps){
		const auto n = std::ranges::size(ps);
		cm::parallel_for(0, n, [&](size_t i){
			auto &&[nid,es] = util::forward_like<Seq>(ps[i]);

			auto &nbhs = (*nodes)[nid].neighbors;
			auto it = std::find_if(nbhs.begin(), nbhs.end(), [&](const auto &ver){
				return ver.epoch==epoch;
			});

			if(it==nbhs.end())
			{
				nbhs.emplace_front(
					epoch,
					util::to<edgelist>(std::forward<decltype(es)>(es))
				);
			}
			else
			{
				it->edges = util::to<edgelist>(std::forward<decltype(es)>(es));
			}
		});
	}

	node_ptr add_node(nid_t nid, const Ext &ext){
		return add_node_impl(nid, ext);
	}
	node_ptr add_node(nid_t nid, Ext &&ext){
		return add_node_impl(nid, std::move(ext));
	}
	template<class Seq>
	void add_nodes(Seq &&ns){
		const auto n = std::ranges::size(ns);
		/*auto nids = ns | std::views::transform([](const auto &p){
			return std::get<0>(p);
		});*/
		auto nids = util::delayed_seq(n, [&](size_t i){
			return std::get<0>(ns[i]);
		});

		nid_t nid_max = *cm::max_element(nids.begin(), nids.end());
		if(!nodes)
			nodes = std::make_shared<nodelist>();
		if(nodes->size()<nid_max+1)
			nodes->resize(nid_max+1);

		cm::parallel_for(0, n, [&](size_t i){
			(*nodes)[nids[i]] = util::forward_like<Seq>(std::get<1>(ns[i]));
		});
	}

	// void remove_node(nid_t);
	template<class Seq>
	void remove_nodes(const Seq &nids){
		const auto n = std::size(nids);
		cm::parallel_for(0, n, [&](size_t i){
			(*nodes)[nids[i]] = typename decltype(*nodes)::value_type{};
		});
	}

	size_t num_nodes() const{
		return nodes? nodes->size(): 0;
	}
	bool empty() const{
		return !nodes || nodes->empty();
	}

	template<class F>
	void iter_each(F &&f) const{
		if(!nodes) return;
		for(size_t i=0, n=nodes->size(); i<n; ++i)
			f(node_cptr(i, &(*nodes)[i]));
	}
	template<class F>
	void iter_each(F &&f){
		if(!nodes) return;
		for(size_t i=0, n=nodes->size(); i<n; ++i)
			f(node_ptr(i, &(*nodes)[i]));
	}
	// TODO: eliminate redundant code by deducing 'this' in C++23
	template<class F>
	void for_each(F &&f) const{
		if(!nodes) return;
		cm::parallel_for(0, nodes->size(),
			[&,it=nodes->begin()](size_t i){
				f(node_cptr(i, &it[i]));
			}
		);
	}
	template<class F>
	void for_each(F &&f){
		if(!nodes) return;
		cm::parallel_for(0, nodes->size(),
			[&,it=nodes->begin()](size_t i){
				f(node_ptr(i, &it[i]));
			}
		);
	}

	chrono_list_base() noexcept = default;
	chrono_list_base(chrono_list_base &&other) noexcept = default;
	chrono_list_base(const chrono_list_base &other) :
		epoch(other.epoch+1), nodes(other.nodes){
	}
	chrono_list_base& operator=(chrono_list_base &&other) noexcept = default;
	chrono_list_base& operator=(const chrono_list_base &other){
		epoch = other.epoch+1;
		nodes = other.nodes;
		return *this;
	}

	void print_node(nid_t u) const{
		printf("Current epoch: [%u]\n", epoch);
		printf("nbhs of %u:\n", u);
		for(const auto &[ep,el] : (*nodes)[u].neighbors)
		{
			printf("epoch [%u]{p:%p sz:%lu}:\t", ep, el.data(), el.size());
			for(const auto &e : el)
				printf(" %u", e.u);
			putchar('\n');
		}
	}

protected:
	epoch_t epoch = 0;
	std::shared_ptr<nodelist> nodes;
};

} // namespace detail

template<
	class Nid, class Ext, class Edge,
	template<typename> class TCtrE=detail::seq_default
>
class chrono_list : public detail::chrono_list_base<Nid,Ext,Edge,TCtrE>
{
	using detail::chrono_list_base<Nid,Ext,Edge,TCtrE>::chrono_list_base;
};

template<class Nid, class Ext, class Edge>
class chrono_list<Nid,Ext,Edge,util::shared_vector> : 
	public detail::chrono_list_base<Nid,Ext,Edge,util::shared_vector>
{
	using _base = detail::chrono_list_base<Nid,Ext,Edge,util::shared_vector>;

	using typename _base::cm;
	using typename _base::edgelist;
	using typename _base::version_t;

public:
	using const_edge_agent = util::ref_view<const edgelist&>;

	class edge_agent{
	public:
		typedef Edge* pointer;
		// typedef typename edgelist::value_type value_type;
		typedef Edge value_type;
		typedef Edge* iterator;
		typedef const Edge* const_iterator;
		typedef Edge& reference;
		typedef const Edge& const_reference;
		typedef size_t size_type;
		typedef ptrdiff_t difference_type;

	private:
		typename cm::seq<value_type> el_new;
		edgelist el_old;
		Edge *raw;
		size_t len;

		void merge(){
			if(!raw || raw==el_old.data()) return;

			thread_local static std::unordered_set<Edge> curr;
			curr.clear();
			for(const Edge &c : el_old)
				curr.insert(c);

			size_t len_r = 0;
			for(const Edge &c : el_new)
				if(curr.find(c)==curr.end())
					len_r++;

			if(len_r+el_old.size()<=el_old.capacity())
			{
				for(const Edge &c : el_new)
					if(curr.find(c)==curr.end())
						el_old.push_back(c);
			}
			else el_old = ANN::util::to<edgelist>(std::move(el_new));

			raw = el_old.data();
			len = el_old.size();
		}

	public:
		edge_agent() : raw(nullptr), len(0){
		}
		edge_agent(const edgelist &el) :
			el_old(el), raw(el_old.data()), len(el_old.size()){
		}
		edge_agent(const edge_agent &other) = default;
		edge_agent(edge_agent &&other) noexcept = default;

		const_reference operator[](size_t i) const{
			return data()[i];
		}
		const_iterator begin() const{
			return raw;
		}
		const_iterator end() const{
			return raw+len;
		}
		const Edge* data() const{
			return raw;
		}
		size_type size() const{
			return len;
		}

		template<typename C>
		C to() &&{
			merge();
			return util::to<C>(std::move(el_old));
		}
		template<typename C>
		C to() &{
			merge();
			return util::to<C>(el_old);
		}


		edge_agent& operator=(const edge_agent&) = default;
		edge_agent& operator=(edge_agent&&) noexcept = default;

		template<class R>
		edge_agent& operator=(R &&r){
			el_new = util::to<decltype(el_new)>(std::forward<R>(r));
			raw = el_new.data();
			len = el_new.size();
			return *this;
		}
	};

	using typename _base::nid_t;
	using typename _base::node_cptr;
	using typename _base::node_ptr;
	using _base::chrono_list_base;
	using _base::get_edges_impl;
	using _base::get_node;

	const_edge_agent get_edges(node_cptr p) const{
		const version_t *ptr = get_edges_impl(p);
		return {ptr->edges};
	}
	const_edge_agent get_edges(nid_t nid) const{
		return get_edges(get_node(nid));
	}
	edge_agent get_edges(node_ptr p){
		const version_t *ptr = get_edges_impl(p);
		return {ptr->edges};
	}
	edge_agent get_edges(nid_t nid){
		return get_edges(get_node(nid));
	}

	void print_node(nid_t u) const{
		printf("Current epoch: [%u]\n", this->epoch);
		printf("nbhs of %u:\n", u);
		for(const auto &[ep,el] : (*this->nodes)[u].neighbors)
		{
			printf("epoch [%u]{p:%p sz:%lu cap:%lu}:\t",
				ep, el.data(), el.size(), el.capacity());
			for(const auto &e : el)
				printf(" %u", e.u);
			putchar('\n');
		}
	}
};

} // namespace ANN::graph

#endif // _ANN_GRAPH_CHRONO_HPP
