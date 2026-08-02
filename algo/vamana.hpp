#ifndef _ANN_ALGO_VAMANA_HPP
#define _ANN_ALGO_VAMANA_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <memory>
#include <functional>
#include <ranges>
#include <fstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <iterator>
#include <type_traits>
#include <typeinfo>
#include <limits>
#include <thread>
#include "algo/algo.hpp"
#include "graph/graph.hpp"
#include "map/direct.hpp"
#include "util/intrin.hpp"
#include "util/debug.hpp"
#include "util/helper.hpp"
#include "util/vec.hpp"
#include "custom/custom.hpp"
#include "io/reader.hpp"
#include "io/writer.hpp"
#include <atomic>

namespace ANN::vamana_details{

template<typename Nid>
struct edge : util::conn<Nid>{
	constexpr bool operator<(const edge &rhs) const{
		return this->u<rhs.u;
	}
	constexpr bool operator>(const edge &rhs) const{
		return this->u>rhs.u;
	}
	constexpr bool operator==(const edge &rhs) const{
		return this->u==rhs.u;
	}
	constexpr bool operator!=(const edge &rhs) const{
		return this->u!=rhs.u;
	}

#ifdef SUPPORT_DELETION
	mutable uint32_t livestamp = 0;
#endif
};

} // namespace ANN::vamana_details

template<typename Nid>
struct std::hash<ANN::vamana_details::edge<Nid>>{
	size_t operator()(const ANN::vamana_details::edge<Nid> &e) const noexcept{
		return std::hash<decltype(e.u)>{}(e.u);
	}
};

namespace ANN{

template<class Desc>
class vamana
{
	protected:
	using cm = custom<typename lookup_custom_tag<Desc>::type>;

	template<typename T>
	using seq = typename cm::seq<T>;

	typedef uint32_t nid_t;
	using point_t = typename Desc::point_t;
	using pid_t = typename point_t::id_t;
	using coord_t = typename point_t::coord_t;
	using md_t = util::vec<seq<float>>;
	using dist_t = typename Desc::dist_t; // TODO: elaborate
	using conn = util::conn<nid_t>;
	using edge = vamana_details::edge<nid_t>;
	using search_control = algo::search_control;
	using prune_control = algo::prune_control;

public:
	struct result_t{
		dist_t dist;
		pid_t pid;
	};
	/*
		Construction parameters
		dim: 		vector dimension
		R: 			max degree
		L:			beam size during the construction
		alpha:		parameter of the heuristic pruning
		batch_base: growth rate of the batch size
	*/
	vamana(uint32_t dim, uint32_t R=50, uint32_t L=75, float alpha=1.0);
	/*
		Construct from the saved model
		getter(i) returns the actual data (convertible to type T) of the vector with id i
	*/
	template<typename F>
	vamana(const std::string &idx_path, const F &coord);

	/*
		Insert the vectors from [begin, end).
		std::iterator_trait<Iter>::value_type ought to be convertible to T
	*/
	template<typename Iter>
	void insert(Iter begin, Iter end, float batch_base=2, bool use_approx_hash=true, bool use_workset=true);
	template<typename Iter>
	void insert_with_timestamp_filtered(Iter begin, Iter end, float batch_base=2, bool use_approx_hash=true, bool use_workset=true);
	/*
		Insert the vectors from [begin, end) with timestamp filtering during beam search.
		During neighbor search, only considers neighbors that are alive at the given timestamp.
	*/
	template<typename Iter, class PointSeq>
	void insert_with_timestamp(Iter begin, Iter end, const PointSeq &ps, float batch_base=2, bool use_approx_hash=true, bool use_workset=true);

#ifdef SUPPORT_DELETION
	template<typename Iter>
	void erase(Iter begin, Iter end);
	void erase_bottom();
#endif

	void save(const std::string &idx_path) const;

	nid_t get_entry_nid() const { return ep; }
	pid_t get_entry_pid() const { return id_map.get_pid(ep); }

#ifdef SUPPORT_DELETION
	void consolidate_full(); // traverse all alive nodes, repair dead edges via 2-hop expansion

	// Rebuild from scratch: extract alive nodes, reset internal state, re-insert (see vamana_reinsert_extension.hpp)
	void consolidate_rebuild(float batch_base = 2);

	// HNT-inspired: consolidate using pre-computed backup candidates (see vamana_hnt_extension.hpp)
	template<class BackupMap>
	void consolidate_with_backups(BackupMap &backups);

	// Reinsert isolated: find alive nodes with all-dead neighbors and re-run beam search (see vamana_reinsert_extension.hpp)
	void consolidate_reinsert_isolated(uint32_t ef_reinsert = 200);

	// Hybrid: 2-hop expansion for partially-dead nodes; brute-force scan fallback for fully isolated nodes (see vamana_reinsert_extension.hpp)
	void consolidate_hybrid(uint32_t ef_reinsert = 200);

	void check_after_consolidate(){
		using ea_t = std::remove_cvref_t<decltype(g.get_edges(nid_t()))>;
		std::atomic<size_t> cnt{0}, valid{0}, invalid_edge{0};
		g.for_each([&](auto p){
			cnt.fetch_add(1, std::memory_order_relaxed);
			nid_t u = p.get_id();
			if(!id_map.contain_nid(u)) return;
			valid.fetch_add(1, std::memory_order_relaxed);
			auto agent = g.get_edges(p);
			if(agent.size()==0) return;
			for(const edge &ve : agent){
				if(!id_map.contain_nid(ve.u)){
					invalid_edge.fetch_add(1, std::memory_order_relaxed);
				}
			}
		});
		printf("number of vertices: %lu/%lu\n",
			valid.load(std::memory_order_relaxed),
			cnt.load(std::memory_order_relaxed));
		printf("number of invalid edges: %lu\n",
			invalid_edge.load(std::memory_order_relaxed));

	}
	void consolidate(){
		using ea_t = std::remove_cvref_t<decltype(g.get_edges(nid_t()))>;
		seq<seq<std::pair<nid_t,ea_t>>> ps(cm::num_workers());
		std::unordered_set<nid_t> del_cache(del_list.begin(), del_list.end());
		printf("size of del_list: %lu\n", del_list.size());

		g.for_each([&](auto p){
			nid_t u = p.get_id();
			if(!id_map.contain_nid(u)) return;

			auto agent = g.get_edges(p);
			if(agent.size()==0) return;

			const coord_t &c = p->get_coord();
			std::unordered_set<nid_t> hash;
			seq<conn> cand;
			bool is_changed = false;
			for(const edge &ve : agent)
			{
				if(del_cache.contains(ve.u))
				{
					is_changed = true;
					auto agent_v = g.get_edges(ve.u);
					auto edge_2hop = agent_v | 
						std::views::filter([&](const edge &e){
							return !del_cache.contains(e.u) && !hash.contains(e.u) && id_map.contain_nid(e.u);
						});
					for(const edge &e : edge_2hop)
					{
						dist_t d = Desc::distance(g.get_node(e.u)->get_coord(), c, dim);
						assert(id_map.contain_nid(e.u));
						cand.push_back({d, e.u});
						hash.insert(e.u);
					}
				}
				else if(!hash.contains(ve.u))
				{
					if(!id_map.contain_nid(ve.u)){
						is_changed = true;
					}else{
						assert(id_map.contain_nid(ve.u));
						hash.insert(ve.u);
						cand.push_back(ve);
					}
				}
			}

			if(is_changed)
			{
				prune_control pctrl{
					.alpha = 1,
					.alpha_min = alpha,
					.alpha_exp = 1/1.2
				};
				seq<conn> conns = algo::occlude_list(
				// seq<conn> conns = algo::prune_heuristic(
					std::move(cand), get_deg_bound(),
					gen_f_nbhs(), gen_f_dist(u), pctrl
				);
				agent = edge_cast(std::move(conns));
				ps[cm::worker_id()].push_back({u, std::move(agent)});
			}
		});
		auto ps_flatten = cm::flatten(std::move(ps));

		auto sizes_cand = ps_flatten |
			std::views::transform([](const auto &s){return s.second.size();});

		fprintf(stderr, "#nodes to update: %lu\n", ps_flatten.size());
		fprintf(stderr, "#cand per node max: %lu\n", cm::reduce(
			sizes_cand, size_t(0), [](size_t x, size_t y){return std::max(x, y);}
		));
		fprintf(stderr, "#cand total: %lu\n", cm::reduce(sizes_cand));

		g.set_edges(std::move(ps_flatten));
		//relic.clear();
		erase_bottom();
	}
#endif

	template<class Seq=seq<result_t>>
	Seq search(
		const coord_t &cq, uint32_t k, uint32_t ef, const search_control &ctrl={}
	) const;

	// search with timestamp filtering (using lifespan sequence)
	template<class Seq=seq<result_t>, class LifespanSeq>
	Seq search_with_timestamp(
		const coord_t &cq, uint32_t k, uint32_t ef, int32_t timestamp,
		const LifespanSeq &lifespans,
		const search_control &ctrl={}
	) const requires requires(const LifespanSeq &ls, size_t i) { ls[i].insert_time; };

	// search from a given start node with timestamp filtering (using lifespan sequence)
	template<class Seq=seq<result_t>, class LifespanSeq>
	Seq search_with_timestamp_from(
		const coord_t &cq, nid_t start_nid, uint32_t k, uint32_t ef, int32_t timestamp,
		const LifespanSeq &lifespans,
		const search_control &ctrl={}
	) const;

	// search with timestamp range filtering (using external timestamp map)
	// (definition in vamana_range.hpp)
	template<class Seq=seq<result_t>, class TimestampMap>
	Seq search_with_timestamp_range(
		const coord_t &cq, uint32_t k, uint32_t ef,
		int32_t range_begin, int32_t range_end,
		const TimestampMap &timestamps,
		const search_control &ctrl={}
	) const;


private:
#ifdef SUPPORT_DELETION
	seq<edge> edge_cast(const seq<conn> &cs){
		// TODO: use range in C++20
		auto es = util::delayed_seq(cs.size(), [&](size_t i){
			return edge{cs[i], deltick};
		});
		return util::to<seq<edge>>(es);
	}
	seq<conn> conn_cast(const seq<edge> &es){
		return util::to<seq<conn>>(es);
	}
#else
	static seq<edge>&& edge_cast(seq<conn> &&cs){
		return reinterpret_cast<seq<edge>&&>(std::move(cs));
	}
	static const seq<edge>& edge_cast(const seq<conn> &cs){
		return reinterpret_cast<const seq<edge>&>(cs);
	}
	static seq<conn>&& conn_cast(seq<edge> &&es){
		return reinterpret_cast<seq<conn>&&>(std::move(es));
	}
	static const seq<conn>& conn_cast(const seq<edge> &es){
		return reinterpret_cast<const seq<conn>&>(es);
	}
#endif

	struct node_t{
		coord_t coord;

		coord_t& get_coord(){
			return coord;
		}
		const coord_t& get_coord() const{
			return coord;
		}
	};

	using graph_t = typename Desc::graph_t<nid_t,node_t,edge>;

	graph_t g;
	map::direct<pid_t,nid_t> id_map;
	uint32_t deltick = 1;

	nid_t ep; // entry point
	md_t medoid;
	uint32_t dim;
	uint32_t R;
	uint32_t L;
	float alpha;
	
	size_t cnt_eval = 0;
	size_t cnt_visited = 0;
	size_t cnt_traversed = 0;

	//std::unordered_map<nid_t,seq<edge>> relic;
	seq<nid_t> del_list;

	template<typename Iter>
	void insert_batch_impl(Iter begin, Iter end, bool use_approx_hash, bool use_workset);

	template<typename Iter, class PointSeq>
	void insert_batch_impl(Iter begin, Iter end, const PointSeq &ps, bool use_approx_hash, bool use_workset);
	
	uint32_t get_deg_bound() const{
		return R;
	}

	auto gen_f_dist(const coord_t &c) const{

		class dist_evaluator{
			std::reference_wrapper<const graph_t> g;
			std::reference_wrapper<const coord_t> c;
			uint32_t dim;
		public:
			dist_evaluator(const graph_t &g, const coord_t &c, uint32_t dim):
				g(g), c(c), dim(dim){
			}
			dist_t operator()(nid_t v) const{
				return Desc::distance(c.get(), g.get().get_node(v)->get_coord(), dim);
			}
			dist_t operator()(nid_t u, nid_t v) const{
				return Desc::distance(
					g.get().get_node(u)->get_coord(),
					g.get().get_node(v)->get_coord(),
					dim
				);
			}
			void prefetch(nid_t v) const{
				const coord_t &vc = g.get().get_node(v)->get_coord();
				const char *p = reinterpret_cast<const char*>(&vc[0]);
				const size_t bytes = size_t(dim) * sizeof(vc[0]);
				for(size_t off = 0; off < bytes; off += 64)
					__builtin_prefetch(p + off, 0, 3);
			}
		};

		return dist_evaluator(g, c, dim);
	}
	auto gen_f_dist(nid_t u) const{
		return gen_f_dist(g.get_node(u)->get_coord());
	}

	// FLATCOORD_EXPERIMENT, query-only: node_t::coord is redundant when points
	// come from a fixed-stride contiguous buffer and nid==pid (true for our
	// .fbin-loaded, no-reorder scenario -- map::direct's identity hash on an
	// integral pid_t guarantees nid==pid regardless of insertion order, and
	// parse_points.hpp's load_from_bin sets coord(i) = fp+header+i*stride on a
	// PERSISTENT mmap). Calibrated once per beamSearch call from two already-
	// stored pointers (nodes 0,1 are guaranteed to exist post-build), then used
	// arithmetically instead of indirecting through nodes[v] for every
	// candidate. Deliberately a SEPARATE method from gen_f_dist rather than a
	// modification of it: gen_f_dist(nid_t) is also called from inside
	// insert_batch_impl mid-build, where nodes 0/1 may not exist yet -- scoping
	// this to only the query call site avoids that hazard entirely.
	auto gen_f_dist_flat(const coord_t &c) const{

		class dist_evaluator_flat{
			std::reference_wrapper<const coord_t> c;
			uint32_t dim;
			coord_t base;
			ptrdiff_t stride;
			coord_t addr(nid_t v) const{
				return base + ptrdiff_t(v)*stride;
			}
		public:
			dist_evaluator_flat(const graph_t &g, const coord_t &c, uint32_t dim):
				c(c), dim(dim){
				coord_t c0 = g.get_node(nid_t(0))->get_coord();
				coord_t c1 = g.get_node(nid_t(1))->get_coord();
				base = c0;
				stride = c1-c0;
				coord_t c2 = g.get_node(nid_t(2))->get_coord();
				if(base+2*stride != c2){
					std::fprintf(stderr, "FLATCOORD_EXPERIMENT: assumption violated (c2 mismatch)\n");
					std::abort();
				}
			}
			dist_t operator()(nid_t v) const{
				return Desc::distance(c.get(), addr(v), dim);
			}
			dist_t operator()(nid_t u, nid_t v) const{
				return Desc::distance(addr(u), addr(v), dim);
			}
			void prefetch(nid_t v) const{
				const char *p = reinterpret_cast<const char*>(addr(v));
				const size_t bytes = std::min<size_t>(256, size_t(dim) * sizeof(*base));
				for(size_t off = 0; off < bytes; off += 64)
					__builtin_prefetch(p + off, 0, 3);
			}
		};

		return dist_evaluator_flat(g, c, dim);
	}

	auto gen_f_nbhs() const{
		return [&](nid_t u){
			#ifdef SUPPORT_DELETION
			auto f = std::views::filter([&](const edge &e){
				auto &ls = e.livestamp;
				if(ls==0) return false;
				if(ls==deltick) return true;
				ls = id_map.contain_nid(e.u)? deltick: 0;
				return ls!=0;
			});
			#endif
			
			auto t = std::views::transform([&](const edge &e){return e.u;});

			if constexpr(std::is_reference_v<decltype(g.get_edges(u))>)
			#ifdef SUPPORT_DELETION
				return std::ranges::ref_view(g.get_edges(u)) | f | t;
			#else
				return std::ranges::ref_view(g.get_edges(u)) | t;
			#endif
			else
			#ifdef SUPPORT_DELETION
				return std::ranges::owning_view(g.get_edges(u)) | f | t;
			#else
				return std::ranges::owning_view(g.get_edges(u)) | t;
			#endif
		};
	}

	// TODO: solve the type of append
	auto refine_edge(nid_t u, seq<conn> &&append){
		auto agent = g.get_edges(u);
		// TODO: consider moving the filter into conn_cast
		// TODO: properly pass the move semantics
		auto edges = util::to<seq<edge>>(
			std::move(agent) |
			std::views::filter([&](const edge &e){
				return id_map.contain_nid(e.u);
			})
		);
		edges.insert(edges.end(),
			std::make_move_iterator(append.begin()),
			std::make_move_iterator(append.end())
		);

		prune_control pctrl{
			.alpha = 1,
			.alpha_min = alpha,
			.alpha_exp = 1/1.2
		};
		seq<conn> conns = algo::occlude_list(
		// seq<conn> conns = algo::prune_heuristic(
			conn_cast(std::move(edges)), get_deg_bound(),
			gen_f_nbhs(), gen_f_dist(u), pctrl
		);
		return edge_cast(std::move(conns));
	}

	template<class Op>
	auto calc_degs(Op op) const{
		seq<size_t> degs(cm::num_workers(), 0);
		g.iter_each([&](auto p){
			auto &deg = degs[cm::worker_id()];
			deg = op(deg, num_edges(p.get_id()));
		});
		return cm::reduce(degs, size_t(0), op);
	}

public:
	size_t num_nodes() const{
		return g.num_nodes();
	}

	size_t num_edges(nid_t u) const{
		// return g.get_edges(u).size();
		// return std::ranges::distance(gen_f_nbhs()(u));
		auto edges = gen_f_nbhs()(u);
		return std::distance(edges.begin(), edges.end());
	}
	size_t num_edges() const{
		return calc_degs(std::plus<>{});
	}

	size_t max_deg() const{
		return calc_degs([](size_t x, size_t y){
			return std::max(x, y);
		});
	}

	void print_stat() const
	{
		printf("cnt_eval: %lu\n", cnt_eval);
		printf("cnt_visited: %lu\n", cnt_visited);
		printf("cnt_traversed: %lu\n", cnt_traversed);

		printf("id_map remains: %lu\n", id_map.size());

		puts("#vertices       edges  avg. deg  max deg");
		size_t cnt_vertex = num_nodes();
		size_t cnt_degree = num_edges();
		printf("%10lu %12lu %8.2f %8lu\n", 
			cnt_vertex, cnt_degree, float(cnt_degree)/cnt_vertex, max_deg()
		);

		printf("ep: [%u]\n", ep);
		printf("ep's nbhs:");
		for(nid_t u : gen_f_nbhs()(ep))
			printf(" %u", u);
		putchar('\n');

		auto t = std::views::transform([&](const edge &e){return e.u;});
		printf("ep's all nbhs:");
		for(nid_t u : g.get_edges(ep)|t)
			printf(" %u", u);
		putchar('\n');

		constexpr const auto support_rc = 
			requires(){g.for_each_raw([](...){});};
		if constexpr(support_rc)
		{
			auto collect_refs = [&](){
				using Map = std::unordered_map<uint64_t,uint64_t>;
				seq<Map> ref_cnt(cm::num_workers());
				g.for_each_raw([&](const auto &e, auto *ptr, auto r){
					(void)r;
					auto &counter = ref_cnt[cm::worker_id()];
					// uint64_t id = (uintptr_t(ptr)<<32) | std::get<0>(e);
					uint64_t id = uintptr_t(ptr);
					const auto &[key,val] = e;
					if constexpr(requires{ val.get_edges_raw().data(); })
					{
						const auto *ptr_el = val.get_edges_raw().data();
						const auto size_el = val.get_edges().size();
						counter[id] = (uintptr_t(ptr_el)<<16) | size_el;
					}
				});
				Map all_cnts;
				for(auto &counter : ref_cnt)
					all_cnts.merge(std::move(counter));
				return all_cnts;
			};

			auto sum_edges = [](const auto &m){
			return std::transform_reduce(
					m.begin(), m.end(),
					size_t(0),
					std::plus<size_t>{},
					// [](uint64_t eid){return size_t(eid&0xffff);}
					[](std::pair<uint64_t,size_t> p){return p.second;}
				);
			};

			// static variables are ad-hoc solution
			// TODO: implement it in a more elegent way
			static std::unordered_set<uint64_t> cnts_vtx;
			static std::unordered_map<uint64_t,size_t> cnts_edge;

			auto refs = collect_refs();
			// TODO: use std::transform_view in C++20
			for(auto [key,val] : refs)
			{
				cnts_vtx.insert(key);
				uint64_t ptr_el = val>>16;
				size_t size_el = val&0xffff;
				auto &old_size = cnts_edge[ptr_el];
				if(old_size<size_el)
					old_size = size_el;
			}
			printf("cnt vtx refs: %lu\n", cnts_vtx.size());
			printf("cnt edge refs: %lu\n", sum_edges(cnts_edge));
		}
	}

	void print_node(nid_t u) const{
		if constexpr(requires{g.print_node(u);})
			g.print_node(u);
	}

	auto get_nbhs(nid_t nid) const{
		return g.get_edges(nid);
	}
};

template<class Desc>
vamana<Desc>::vamana(uint32_t dim, uint32_t R, uint32_t L, float alpha) :
	dim(dim), R(R), L(L), alpha(alpha)
{
}

template<class Desc>
template<typename F>
vamana<Desc>::vamana(const std::string &idx_path, const F &coord)
{
	using namespace std::views;

	io::mmap_reader r(idx_path);
	assert((int)r()==0x04);
	assert((std::string)r(8)=="vamanaRG");
	assert((size_t)r()==
		(typeid(Desc).hash_code()^typeid(graph_t).hash_code()));

	dim = r();
	R = r();
	L = r();
	alpha = r();
	printf("dim: %u, R: %u, L: %u, alpha: %.2f\n", dim, R, L, alpha);

	ep = r();
	printf("ep: %u\n", ep);

	// TODO use view-like return value to prevent dup mem
	seq<std::pair<pid_t,nid_t>> mapping = r((size_t)r());
	id_map.insert(mapping | transform([](const auto &p){return p.first;}));

	g = graph::load<graph_t>(r, 
		[&](nid_t u){return node_t{coord(id_map.get_pid(u))};},
		[&](nid_t u, nid_t v){
			auto d = Desc::distance(
				coord(id_map.get_pid(u)),
				coord(id_map.get_pid(v)),
				dim
			);
		#ifdef SUPPORT_DELETION
			return edge{{d,v}, deltick};
		#else
			return edge{d,v};
		#endif
		}
	);

	seq<md_t> sum_coords(cm::num_workers(), md_t(dim));
	g.for_each([&](auto p){
		auto &c = sum_coords[cm::worker_id()];
		c += md_t(util::inner_t{}, p->get_coord(), dim);
	});
	medoid = cm::reduce(sum_coords, md_t(dim))/g.num_nodes();
}

template<class Desc>
template<typename Iter>
void vamana<Desc>::insert(Iter begin, Iter end, float batch_base, bool use_approx_hash, bool use_workset)
{
	// static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type, point_t>);
	static_assert(std::is_base_of_v<
		std::random_access_iterator_tag, typename std::iterator_traits<Iter>::iterator_category
	>);

	const size_t n = std::distance(begin, end);
	if(n==0) return;

	// std::random_device rd;
	auto perm = cm::random_permutation(n/*, rd()*/);
	auto rand_seq = util::delayed_seq(n, [&](size_t i) -> decltype(auto){
		// return *(begin+perm[i]); // CHECK: restore before release
		return *(begin+i);
	});

	size_t cnt_skip = 0;
	if(g.empty())
	{
		auto init = rand_seq.begin();
		ep = id_map.insert(init->get_id());
		medoid = md_t(util::inner_t{}, init->get_coord(), dim);
		g.add_node(ep, node_t{init->get_coord()});
		cnt_skip = 1;
	}

	size_t batch_begin=0, batch_end=cnt_skip, size_limit=std::max<size_t>(n*0.02,20000);
	float progress = 0.0;
	while(batch_end<n)
	{
		batch_begin = batch_end;
		batch_end = std::min({n, (size_t)std::ceil(batch_begin*batch_base)+1, batch_begin+size_limit});

		util::debug_output("Batch insertion: [%u, %u)\n", batch_begin, batch_end);
		insert_batch_impl(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end, use_approx_hash, use_workset);
		// insert(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end, false);

		if(batch_end>n*(progress+0.05))
		{
			progress = float(batch_end)/n;
			fprintf(stderr, "Built: %3.2f%%\n", progress*100);
			// fprintf(stderr, "# visited: %lu\n", cm::reduce(per_visited));
			// fprintf(stderr, "# eval: %lu\n", cm::reduce(per_eval));
			// fprintf(stderr, "size of C: %lu\n", cm::reduce(per_size_C));
			// per_visited.clear();
			// per_eval.clear();
			// per_size_C.clear();
		}
	}

	// fprintf(stderr, "# visited: %lu\n", cm::reduce(per_visited));
	// fprintf(stderr, "# eval: %lu\n", cm::reduce(per_eval));
	// fprintf(stderr, "size of C: %lu\n", cm::reduce(per_size_C));
	// per_visited.clear();
	// per_eval.clear();
	// per_size_C.clear();
}

template<class Desc>
template<typename Iter, class PointSeq>
void vamana<Desc>::insert_with_timestamp(Iter begin, Iter end, const PointSeq &ps, float batch_base, bool use_approx_hash, bool use_workset)
{
	static_assert(std::is_base_of_v<
		std::random_access_iterator_tag, typename std::iterator_traits<Iter>::iterator_category
	>);

	const size_t n = std::distance(begin, end);
	if(n==0) return;

	// std::random_device rd;
	auto perm = cm::random_permutation(n/*, rd()*/);
	auto rand_seq = util::delayed_seq(n, [&](size_t i) -> decltype(auto){
		// return *(begin+perm[i]); // CHECK: restore before release
		return *(begin+i);
	});

	size_t cnt_skip = 0;
	if(g.empty())
	{
		auto init = rand_seq.begin();
		ep = id_map.insert(init->get_id());
		medoid = md_t(util::inner_t{}, init->get_coord(), dim);
		g.add_node(ep, node_t{init->get_coord()});
		cnt_skip = 1;
	}

	size_t batch_begin=0, batch_end=cnt_skip, size_limit=std::max<size_t>(n*0.02,20000);
	float progress = 0.0;
	while(batch_end<n)
	{
		batch_begin = batch_end;
		batch_end = std::min({n, (size_t)std::ceil(batch_begin*batch_base)+1, batch_begin+size_limit});

		util::debug_output("Batch insertion: [%u, %u)\n", batch_begin, batch_end);
		insert_batch_impl(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end, ps, use_approx_hash, use_workset);

		if(batch_end>n*(progress+0.05))
		{
			progress = float(batch_end)/n;
			fprintf(stderr, "Built: %3.2f%%\n", progress*100);
		}
	}
}

template<class Desc>
template<typename Iter>
void vamana<Desc>::insert_batch_impl(Iter begin, Iter end, bool use_approx_hash, bool use_workset)
{
	const size_t size_batch = std::distance(begin,end);
	seq<nid_t> nids(size_batch);

	// per_visited.resize(size_batch);
	// per_eval.resize(size_batch);
	// per_size_C.resize(size_batch);

	// before the insertion, prepare the needed data
	// `nids[i]` is the nid of the node corresponding to the i-th 
	// point to insert in the batch, associated with level[i]
	id_map.insert(util::delayed_seq(size_batch, [&](size_t i){
		return (begin+i)->get_id();
	}));

	cm::parallel_for(0, size_batch, [&](uint32_t i){
		nids[i] = id_map.get_nid((begin+i)->get_id());
	});

	g.add_nodes(util::delayed_seq(size_batch, [&](size_t i){
		// GUARANTEE: begin[*].get_coord is only invoked for assignment once
		return std::pair{nids[i], node_t{(begin+i)->get_coord()}};
	}));

	const auto n_prev = g.num_nodes();
	// TODO: make util::filter compatible with ranges
/*	auto coord_drift = cm::reduce(
		std::ranges::subrange(begin, end) |
		std::views::filter([&](auto &&p){return !!id_map.contain_nid(p.get_id());}) |
		std::views::transform([&](auto &&p){return md_t(util::inner_t{},p.get_coord(),dim);})
	);*/
	auto it_refs = util::delayed_seq(size_batch, [&](size_t i){
		return begin+i;
	});
	auto refs_new = util::filter(it_refs, [&](Iter it){
		return !!id_map.find_nid(it->get_id());
	});
	auto coords_new = util::delayed_seq(
		refs_new.size(),
		[&](size_t i){return md_t(util::inner_t{},refs_new[i]->get_coord(),dim);}
	);
	md_t coord_drift = cm::reduce(coords_new, md_t(dim));

	// below we (re)generate edges incident to nodes in the current batch
	// add adges from the new points
	// TODO: change edge_added to conn_added
	seq<seq<std::pair<nid_t,edge>>> edge_added(size_batch);
	seq<std::pair<nid_t,seq<edge>>> nbh_forward(size_batch);
	seq<algo::statistic> stats(size_batch);
	cm::parallel_for(0, size_batch, [&](size_t i){
		const nid_t u = nids[i];

		search_control sctrl;
		sctrl.log_stat = &stats[i];
		sctrl.use_approx_hash = use_approx_hash;
		sctrl.use_workset = use_workset;
		seq<conn> res = algo::beamSearch2(gen_f_nbhs(), gen_f_dist(u), seq<nid_t>{ep}, L, sctrl).second;

		prune_control pctrl{
			.alpha = 1,
			.alpha_min = alpha,
			.alpha_exp = 1/1.2
		};
		seq<conn> conn_u = algo::occlude_list(
		// seq<conn> conn_u = algo::prune_heuristic(
			std::move(res), get_deg_bound(), 
			gen_f_nbhs(), gen_f_dist(u), pctrl
		);
		// record the edge for the backward insertion later
		auto &edge_cur = edge_added[i];
		edge_cur.clear();
		edge_cur.reserve(conn_u.size());
		for(const auto &[d,v] : conn_u)
		#ifdef SUPPORT_DELETION
			edge_cur.emplace_back(v, edge{{d,u}, deltick});
		#else
			edge_cur.emplace_back(v, edge{d,u});
		#endif

		// store for batch insertion
		nbh_forward[i] = {u, edge_cast(std::move(conn_u))};
	});
	util::debug_output("Adding forward edges\n");
	g.set_edges(std::move(nbh_forward));

	for(size_t i=0; i<size_batch; ++i)
	{
		util::debug_output("%u:\t%u\t%u\n", 
			nids[i], stats[i].cnt_eval, stats[i].cnt_visited, stats[i].cnt_traversed
		);
	}
	cnt_eval += cm::reduce(stats | std::views::transform(
		[&](const auto &s){return s.cnt_eval;}
	));
	cnt_visited += cm::reduce(stats | std::views::transform(
		[&](const auto &s){return s.cnt_visited;}
	));
	cnt_traversed += cm::reduce(stats | std::views::transform(
		[&](const auto &s){return s.cnt_traversed;}
	));

	// now we add edges in the other direction
	auto edge_added_flatten = util::flatten(std::move(edge_added));
	auto edge_added_grouped = util::group_by_key(std::move(edge_added_flatten));

	using agent_t = std::remove_cvref_t<decltype(g.get_edges(nid_t()))>;
	seq<std::pair<nid_t,agent_t>> nbh_backward(edge_added_grouped.size());

	cm::parallel_for(0, edge_added_grouped.size(), [&](size_t j){
		nid_t v = edge_added_grouped[j].first;
		auto &nbh_v_add = edge_added_grouped[j].second;

		auto edge_agent_v = g.get_edges(v);
		// TODO: consider moving the filter into conn_cast
		// TODO: properly pass the move semantics
		/*
		auto edge_v = util::to<seq<edge>>(
			edge_agent_v |
			std::views::filter([&](const edge &e){
				return id_map.contain_nid(e.u);
			})
		);
		*/
		auto edge_v = util::to<seq<edge>>(std::move(edge_agent_v));
		// TODO: prevent assignment twice
		/*
		if(edge_v.size()<get_deg_bound()*0.3)
			edge_v = edge_cast(
				algo::beamSearch(gen_f_nbhs(), gen_f_dist(v), seq<nid_t>{ep}, L)
			);
		*/
		edge_v.insert(edge_v.end(),
			std::make_move_iterator(nbh_v_add.begin()),
			std::make_move_iterator(nbh_v_add.end())
		);
		/*
		seq<conn> conn_v = algo::prune_simple(
			conn_cast(std::move(edge_v)),
			get_deg_bound()
		);
		*/
		prune_control pctrl{
			.alpha = 1,
			.alpha_min = alpha,
			.alpha_exp = 1/1.2
		};
		seq<conn> conn_v = algo::occlude_list(
		// seq<conn> conn_v = algo::prune_heuristic(
			conn_cast(std::move(edge_v)), get_deg_bound(),
			gen_f_nbhs(), gen_f_dist(v), pctrl//prune_control{.alpha=alpha}
		);
		edge_agent_v = edge_cast(std::move(conn_v));
		nbh_backward[j] = {v, std::move(edge_agent_v)};
	});
	util::debug_output("Adding backward edges\n");
	g.set_edges(std::move(nbh_backward));

	// finally, update the entrances
	util::debug_output("Updating entry point after insertion\n");
	const auto n_curr = g.num_nodes();
	((medoid*=n_prev)+=coord_drift)/=n_curr;
	util::vec<seq<typename point_t::elem_t>> t(util::inner_t{}, medoid.data(), dim);
	auto res = algo::beamSearch2(
		gen_f_nbhs(),
		[&](nid_t v){return Desc::distance(
			t.data(), g.get_node(v)->get_coord(), dim);
		},
		seq<nid_t>{ep},
		1
	).second;
	ep = std::min_element(res.begin(), res.end())->u;
	util::debug_output("new ep: %u\n", ep);
}

template<class Desc>
template<typename Iter, class PointSeq>
void vamana<Desc>::insert_batch_impl(Iter begin, Iter end, const PointSeq &ps, bool use_approx_hash, bool use_workset)
{
	const size_t size_batch = std::distance(begin,end);
	seq<nid_t> nids(size_batch);

	// before the insertion, prepare the needed data
	id_map.insert(util::delayed_seq(size_batch, [&](size_t i){
		return (begin+i)->get_id();
	}));

	cm::parallel_for(0, size_batch, [&](uint32_t i){
		nids[i] = id_map.get_nid((begin+i)->get_id());
	});

	g.add_nodes(util::delayed_seq(size_batch, [&](size_t i){
		return std::pair{nids[i], node_t{(begin+i)->get_coord()}};
	}));

	const auto n_prev = g.num_nodes();
	auto it_refs = util::delayed_seq(size_batch, [&](size_t i){
		return begin+i;
	});
	auto refs_new = util::filter(it_refs, [&](Iter it){
		return !!id_map.find_nid(it->get_id());
	});
	auto coords_new = util::delayed_seq(
		refs_new.size(),
		[&](size_t i){return md_t(util::inner_t{},refs_new[i]->get_coord(),dim);}
	);
	md_t coord_drift = cm::reduce(coords_new, md_t(dim));

	// Generate gen_f_nbhs with timestamp filter
	auto gen_f_nbhs_timestamp = [&]() {
		return [&](nid_t u) {
			// deletion filter (from original gen_f_nbhs)
			#ifdef SUPPORT_DELETION
			auto deletion_filter = std::views::filter([&](const edge &e){
				auto &ls = e.livestamp;
				if(ls==0) return false;
				if(ls==deltick) return true;
				ls = id_map.contain_nid(e.u)? deltick: 0;
				return ls!=0;
			});
			#endif
			
			// timestamp filter for edges
			auto timestamp_filter = std::views::filter([&](const edge &e) {
				if (!id_map.contain_nid(e.u)) return false;
				
				// get the point id and check timestamps directly from point sequence
				pid_t pid = id_map.get_pid(e.u);
				const auto &point = ps[pid];
				int32_t insert_time = point.get_insert_time();
				int32_t delete_time = point.get_delete_time();
				int32_t timestamp = ps[u].get_insert_time();
				
				// check timestamp condition: insert_time <= timestamp < delete_time
				return insert_time <= timestamp && delete_time > timestamp;
			});
			
			// transform edge to nid_t
			auto t = std::views::transform([&](const edge &e){return e.u;});

			// apply filters in sequence: deletion -> timestamp -> transform
			if constexpr(std::is_reference_v<decltype(g.get_edges(u))>)
				#ifdef SUPPORT_DELETION
				return std::ranges::ref_view(g.get_edges(u)) | deletion_filter | t;
				#else
				return std::ranges::ref_view(g.get_edges(u)) | t;
				#endif
			else
				#ifdef SUPPORT_DELETION
				return std::ranges::owning_view(g.get_edges(u)) | deletion_filter | t;
				#else
				return std::ranges::owning_view(g.get_edges(u)) | t;
				#endif
		};
	};

	// below we (re)generate edges incident to nodes in the current batch
	seq<seq<std::pair<nid_t,edge>>> edge_added(size_batch);
	seq<std::pair<nid_t,seq<edge>>> nbh_forward(size_batch);
	seq<algo::statistic> stats(size_batch);
	cm::parallel_for(0, size_batch, [&](size_t i){
		const nid_t u = nids[i];

		search_control sctrl;
		sctrl.log_stat = &stats[i];
		sctrl.use_approx_hash = use_approx_hash;
		sctrl.use_workset = use_workset;
		// Use timestamp-filtered gen_f_nbhs
		seq<conn> res = algo::beamSearch2(gen_f_nbhs_timestamp(), gen_f_dist(u), seq<nid_t>{ep}, L, sctrl).second;

		prune_control pctrl{
			.alpha = 1,
			.alpha_min = alpha,
			.alpha_exp = 1/1.2
		};
		seq<conn> conn_u = algo::occlude_list(
			std::move(res), get_deg_bound(), 
			gen_f_nbhs_timestamp(), gen_f_dist(u), pctrl
		);
		// record the edge for the backward insertion later
		auto &edge_cur = edge_added[i];
		edge_cur.clear();
		edge_cur.reserve(conn_u.size());
		for(const auto &[d,v] : conn_u)
		#ifdef SUPPORT_DELETION
			edge_cur.emplace_back(v, edge{{d,u}, deltick});
		#else
			edge_cur.emplace_back(v, edge{d,u});
		#endif

		// store for batch insertion
		nbh_forward[i] = {u, edge_cast(std::move(conn_u))};
	});
	util::debug_output("Adding forward edges\n");
	g.set_edges(std::move(nbh_forward));

	for(size_t i=0; i<size_batch; ++i)
	{
		util::debug_output("%u:\t%u\t%u\n", 
			nids[i], stats[i].cnt_eval, stats[i].cnt_visited, stats[i].cnt_traversed
		);
	}
	cnt_eval += cm::reduce(stats | std::views::transform(
		[&](const auto &s){return s.cnt_eval;}
	));
	cnt_visited += cm::reduce(stats | std::views::transform(
		[&](const auto &s){return s.cnt_visited;}
	));
	cnt_traversed += cm::reduce(stats | std::views::transform(
		[&](const auto &s){return s.cnt_traversed;}
	));

	// now we add edges in the other direction
	auto edge_added_flatten = util::flatten(std::move(edge_added));
	auto edge_added_grouped = util::group_by_key(std::move(edge_added_flatten));

	using agent_t = std::remove_cvref_t<decltype(g.get_edges(nid_t()))>;
	seq<std::pair<nid_t,agent_t>> nbh_backward(edge_added_grouped.size());

	cm::parallel_for(0, edge_added_grouped.size(), [&](size_t j){
		nid_t v = edge_added_grouped[j].first;
		auto &nbh_v_add = edge_added_grouped[j].second;

		auto edge_agent_v = g.get_edges(v);
		auto edge_v = util::to<seq<edge>>(std::move(edge_agent_v));
		edge_v.insert(edge_v.end(),
			std::make_move_iterator(nbh_v_add.begin()),
			std::make_move_iterator(nbh_v_add.end())
		);
		prune_control pctrl{
			.alpha = 1,
			.alpha_min = alpha,
			.alpha_exp = 1/1.2
		};
		seq<conn> conn_v = algo::occlude_list(
			conn_cast(std::move(edge_v)), get_deg_bound(),
			gen_f_nbhs_timestamp(), gen_f_dist(v), pctrl
		);
		edge_agent_v = edge_cast(std::move(conn_v));
		nbh_backward[j] = {v, std::move(edge_agent_v)};
	});
	util::debug_output("Adding backward edges\n");
	g.set_edges(std::move(nbh_backward));

	util::debug_output("Updating entry point after insertion\n");
	const auto n_curr = g.num_nodes();
	((medoid*=n_prev)+=coord_drift)/=n_curr;
	util::vec<seq<typename point_t::elem_t>> t(util::inner_t{}, medoid.data(), dim);
	auto res = algo::beamSearch2(
		gen_f_nbhs_timestamp(),
		[&](nid_t v){return Desc::distance(
			t.data(), g.get_node(v)->get_coord(), dim);
		},
		seq<nid_t>{ep},
		1
	).second;
	ep = std::min_element(res.begin(), res.end())->u;
	util::debug_output("new ep: %u\n", ep);
}

#ifdef SUPPORT_DELETION
template<class Desc>
template<typename Iter>
void vamana<Desc>::erase(Iter begin, Iter end)
{
	static_assert(std::is_convertible_v<
		typename std::iterator_traits<Iter>::value_type, pid_t
	>);
	/*
	static_assert(std::is_base_of_v<
		std::random_access_iterator_tag,
		typename std::iterator_traits<Iter>::iterator_category
	>);*/

	auto nids = std::ranges::subrange(begin,end) |
		std::views::transform([&](const pid_t &p){return id_map.get_nid(p);});
	
	del_list.insert(del_list.end(), nids.begin(), nids.end());
	id_map.erase(begin, end);
}

template<class Desc>
void vamana<Desc>::erase_bottom()
{
	auto nids = del_list | std::views::all;
	auto r = nids | std::views::transform([&](nid_t u){
		return md_t(util::inner_t{}, g.get_node(u)->get_coord(), dim);
	});
	auto drift = cm::reduce(r);
/*
	for(nid_t u : nids)
		relic[u] = util::to<seq<edge>>(std::move(g.get_edges(u)));
*/
	((medoid*=g.num_nodes())-=drift);

	util::debug_output("Updating entry point after erasion\n");
	medoid /= g.num_nodes();
	util::vec<seq<typename point_t::elem_t>> t(util::inner_t{}, medoid.data(), dim);
	auto [workset,chosen] = algo::beamSearch2(
		gen_f_nbhs(),
		[&](nid_t v){return Desc::distance(
			t.data(), g.get_node(v)->get_coord(), dim);
		},
		seq<nid_t>{id_map.front_nid()},
		1
	);
	ep = std::min_element(chosen.begin(), chosen.end())->u;
	util::debug_output("new ep: %u\n", ep);

	g.remove_nodes(nids.begin(), nids.end());
	deltick++;
	del_list.clear();
	/*
	auto &&es = g.get_edges(ep);// TODO: fix the type
	std::unordered_set hash(es.begin(), es.end());
	g.set_edges(ep, refine_edge(ep,util::to<seq<edge>>(
		chosen | std::views::filter([&](const conn &c){return !hash.contains(c.u);})
	)));
	*/
}
#endif

template<class Desc>
template<class Seq>
Seq vamana<Desc>::search(
	const coord_t &cq, uint32_t k, uint32_t ef, const search_control &ctrl) const
{
	auto nbhs = ctrl.flat_coord
		? beamSearch(gen_f_nbhs(), gen_f_dist_flat(cq), seq<nid_t>{ep}, ef, ctrl).second
		: beamSearch(gen_f_nbhs(), gen_f_dist(cq), seq<nid_t>{ep}, ef, ctrl).second;

	//nbhs = algo::prune_simple(std::move(nbhs), k/*, ctrl*/); // TODO: set ctrl
	cm::sort(nbhs.begin(), nbhs.end());

	using result_t = typename Seq::value_type;
	static_assert(util::is_direct_list_initializable_v<
		result_t, dist_t, pid_t
	>);
	//Seq res(nbhs.size());
	Seq res;
	for(size_t i=0; i<nbhs.size(); ++i)
	{
		const auto &nbh = nbhs[i];
		if(id_map.contain_nid(nbh.u))
		{
			res.push_back({nbh.d, id_map.get_pid(nbh.u)});
			if(res.size()>=k) break;
		}
	};

	return res;
}

template<class Desc>
void vamana<Desc>::save(const std::string &idx_path) const
{
	io::buffered_writer w(idx_path);
	/* Meta info */
	w(0x04); // version
	w("vamanaRG", 8);  // 8-byte algorithm identity
	w(typeid(Desc).hash_code()^typeid(graph_t).hash_code());

	/* Build parameters */
	w(dim);
	w(R);
	w(L);
	w(alpha);

	/* Auxiliary info */
	w(ep);
	// medoid will be calucated on load

	/* id_map */
	w(id_map.size());
	w(util::to<seq<std::pair<const pid_t,nid_t>>>(id_map.get_map()));

	/* Graph */
	graph::save(g, w);
}

} // namespace ANN

#endif // _ANN_ALGO_VAMANA_HPP
