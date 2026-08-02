#ifndef _ANN_ALGO_HPP
#define _ANN_ALGO_HPP

#include <cstdio>
#include <cassert>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <unordered_set>
#include <utility>
#include <ranges>
#include <vector>

#include "ANN.hpp"
#include "custom/custom.hpp"
#include "util/debug.hpp"
#include "util/intrin.hpp"
using ANN::util::debug_output;

namespace ANN {
namespace algo {

namespace detail{
// Only call f_dist.prefetch(v) if that member exists, so f_dist types
// without it (other algos' dist evaluators, or plain lambdas) keep
// compiling unchanged.
template<typename D, typename Nid, typename=void>
struct has_prefetch : std::false_type{};
template<typename D, typename Nid>
struct has_prefetch<D, Nid, std::void_t<
	decltype(std::declval<const D&>().prefetch(std::declval<Nid>()))
>> : std::true_type{};

template<typename D, typename Nid>
void try_prefetch(const D &f_dist, Nid v){
	if constexpr(has_prefetch<D,Nid>::value)
		f_dist.prefetch(v);
}
} // namespace detail

struct statistic{
	uint32_t cnt_eval = 0;
	uint32_t cnt_visited = 0;
	uint32_t cnt_traversed = 0;
};

struct search_control
{
	bool verbose_output = false;
	bool skip_search = false;
	float beta = 1;
	// bool filtered = true;
	// bool searching = false;
	std::optional<float> radius;
	statistic *log_stat = nullptr;
	std::optional<uint32_t> indicate_ep;
	std::optional<uint32_t> limit_eval;
	bool use_approx_hash = true;
	bool use_workset = true;
	// Opt-in only: caller must guarantee every currently-alive node's
	// coordinate lives at a fixed stride from a single base address (true for
	// a vamana index whose points all come from one contiguous, persistent
	// buffer -- e.g. one bulk mmap'd load -- with no deletion/reinsertion from
	// a different buffer since). Not something vamana can verify on its own;
	// wrong here is a silent wrong-answer bug, not a crash. See
	// vamana::gen_f_dist_flat.
	bool flat_coord = false;
};

struct prune_control {
	bool prune_nbh = false;
	bool recycle_pruned = false;
	float alpha = 1;
	float alpha_min = 0;
	float alpha_exp = 0;
};

template<class L=lookup_custom_tag<>, class E, class D, class Seq>
auto beamSearch(
	E &&f_nbhs, D &&f_dist, const Seq &eps, uint32_t ef, const search_control &ctrl={})
{
	using cm = custom<typename L::type>;
	using nid_t = std::ranges::range_value_t<Seq>;
	using conn = util::conn<nid_t>;

	const auto nid_invalid = std::numeric_limits<nid_t>::max();
	// visited[] is an approximate (collision-clobbering) membership table: once
	// the number of hops exceeds its slot count, an already-seen node can get
	// evicted and re-discovered, paying for its distance a second time. Sizing
	// was pinned at 128 slots regardless of ef -- fine up to beam~128, but
	// beam=640/1280 run ~645/1284 hops through only 128 slots, and the
	// resulting extra distance calls (measured: +10%/+27% cmps vs ParlayANN at
	// those two beams specifically, flat below that) is exactly what ParlayANN's
	// hashset sized `2*(10+beamSize)*max_degree` avoids by scaling with beam
	// size instead of a fixed cap. Matching that formula's growth rate (their
	// R=32 default gives bits(n) ~= ceil(log2(ef))+6) here, without needing R
	// threaded into beamSearch, and keeping this a fixed-size table rather than
	// their open-addressed auto-growing one.
	const uint32_t width = ef;
	// Capped at 16 bits (65536 slots, 256KB) so the table stays cache-resident
	// even at large ef -- uncapped, beam=1280 reaches 2^17 slots (512KB), and
	// `visited[hu]==u`'s own cache-miss rate (measured via perf annotate:
	// 2.62% of beamSearch's time at beam=80 vs 5.87% at beam=1280) eats into
	// the redundant-cmps savings the larger table exists to buy in the first
	// place. 16 bits still leaves beam<=640 untouched (their natural size is
	// already <=2^16) and only trims beam=1280's table, at a measured cost of
	// +2.25% cmps for a net +1.4% query throughput there.
	const uint32_t bits = width>2? std::min<uint32_t>(std::ceil(std::log2(width))+6, 16): 2;
	const uint32_t mask = (1u<<bits)-1;
	std::vector<nid_t> visited(mask+1, nid_invalid);
	std::unordered_set<nid_t> visited_det;
	uint32_t cnt_visited=0, cnt_traversed=0;
	typename cm::seq<conn> workset, chosen;
	std::set<conn> cand; // TODO: test dual heaps
	std::unordered_set<nid_t> is_inw; // TODO: test merge instead
	// TODO: get statistics about the merged size
	// TODO: switch to the alternative if exceeding a threshold
	if(ctrl.use_workset)
		workset.reserve(ef+1);

	auto visit = [&](nid_t u){
		if(ctrl.use_approx_hash){
			const auto hu = cm::hash64(u) & mask;
			if(visited[hu]==u) return false;
			visited[hu] = u;
			cnt_visited++;
			return true;
		}
		else{
			bool inserted = visited_det.insert(u).second;
			if(inserted) cnt_visited++;
			return inserted;
		}
	};

	for(nid_t pe : eps) {
		if(!visit(pe)) continue;
		const auto d = f_dist(pe);
		cand.insert({d, pe});
		if(ctrl.use_workset)
			workset.push_back({d, pe});
		is_inw.insert(pe);
	}
	if(ctrl.use_workset)
		std::make_heap(workset.begin(), workset.end());

	uint32_t cnt_eval = 0;
	uint32_t limit_eval = ctrl.limit_eval.value_or(std::numeric_limits<uint32_t>::max());
	// candidates found this hop: prefetched here, distances computed below once
	// the prefetches have had a chance to land (see try_prefetch above).
	std::vector<nid_t> pending;
	while (cand.size() > 0) {
		if (cand.begin()->d > workset[0].d * ctrl.beta) break;

		if (++cnt_eval > limit_eval) break;

		nid_t u = cand.begin()->u;
		chosen.push_back(*cand.begin());
		cand.erase(cand.begin());

		pending.clear();
		util::iter_each(f_nbhs(u), [&](nid_t pv) {
			cnt_traversed++;
			if(!visit(pv)) return;
			detail::try_prefetch(f_dist, pv);
			pending.push_back(pv);
		});

		for(nid_t pv : pending) {
			const auto d = f_dist(pv);
			auto back_it = cand.end();
			if(ctrl.use_workset)
			{
				if (!(workset.size() < ef || d < workset[0].d)) continue;
			}
			else
			{
				if(!(cand.size()<ef || d<(--back_it)->d)) continue;
			}
			if (!is_inw.insert(pv).second) continue;

			cand.insert({d,pv});
			if(ctrl.use_workset)
			{
				workset.push_back({d,pv});
				std::push_heap(workset.begin(), workset.end());
				if(workset.size()>ef)
				{
					std::pop_heap(workset.begin(), workset.end());
					is_inw.erase(workset.back().u);
					workset.pop_back();
				}
			}
			if(cand.size()>ef)
			{
				if(ctrl.use_workset)
				{
					cand.erase(std::prev(cand.end()));
				}
				else
				{
					assert(back_it!=cand.end());
					is_inw.erase(back_it->u);
					cand.erase(back_it);
				}
			}
		}
	}

	if(ctrl.log_stat)
	{
		ctrl.log_stat->cnt_eval = cnt_eval;
		ctrl.log_stat->cnt_visited = cnt_visited;
		ctrl.log_stat->cnt_traversed = cnt_traversed;
	}
	return std::pair(workset,chosen);
}

template<class L=lookup_custom_tag<>, class E, class D, class Seq>
auto beamSearch2(
	E &&f_nbhs, D &&f_dist, const Seq &eps, uint32_t ef, const search_control &ctrl={})
{
	using cm = custom<typename L::type>;
	using nid_t = std::ranges::range_value_t<Seq>;
	using conn = util::conn<nid_t>;
	uint32_t cnt_eval = 0;
	uint32_t cnt_visited = 0;
	uint32_t cnt_traversed = 0;

	const auto nid_invalid = std::numeric_limits<nid_t>::max();
	// same reasoning as beamSearch's width/bits above -- kept consistent here.
	const uint32_t bits = ef>2? std::ceil(std::log2(ef))+6: 2;
	const uint32_t mask = (1u<<bits)-1;
	// was `typename cm::seq<...>` (parlay::sequence) for every local below:
	// its fill/resize-growth always dispatch through parallel_for, including
	// new_frontier.resize(max_cap) further down, which runs on EVERY loop
	// iteration (once per hop), not just once per call. beamSearch2 runs
	// inside insert_batch_impl's outer parallel_for over inserted points, so
	// each of those nested parallel_for calls is scheduler overhead competing
	// with the already-saturated outer parallelism -- once per hop, per
	// inserted point. std::vector's fill/resize are plain sequential loops.
	// Measured on deep 10M build: 51.96s -> 44.80s (-13.8%); bigann 10M:
	// 63.24s -> 57.26s (-9.4%); graph byte-identical in both (same edge count,
	// avg/max degree, entry point, entry point's neighbor list).
	std::vector<nid_t> hash_filter(mask+1, nid_invalid);
	auto is_seen = [&](nid_t u) -> bool{
		const auto hu = cm::hash64(u)&mask;
		if(hash_filter[hu]==u) return true;
		hash_filter[hu] = u;
		return false;
	};

	std::vector<conn> frontier;
	frontier.reserve(std::max<size_t>(eps.size(), ef));
	for(nid_t pe : eps)
	{
		const auto h_pe = cm::hash64(pe) & mask;
		if(hash_filter[h_pe]==pe) continue;
		hash_filter[h_pe] = pe;
		const auto d = f_dist(pe);
		frontier.push_back({d,pe});
	}
	cm::sort(frontier.begin(), frontier.end());
	const size_t frontier_buffer_size = std::max<size_t>({
		frontier.size(), size_t(ef)*2, size_t(1)
	});
	size_t frontier_size = frontier.size();
	frontier.resize(frontier_buffer_size);

	std::vector<conn> unvisited_frontier(std::max<size_t>(ef, 1));
	unvisited_frontier[0] = frontier[0];
	size_t unvisited_frontier_size = 1;

	std::vector<conn> visited;
	visited.reserve(2*ef);

	size_t dist_cmps = eps.size();
	uint32_t num_visited = 0;

	std::vector<conn> new_frontier(frontier_buffer_size), candidates;
	std::vector<nid_t> keep;
	// frontier and new_frontier are fixed storage whose logical length is kept
	// separately.  They alternate roles after every merge, so the hot loop does
	// not allocate or default-initialize a fresh output range once per graph hop.
	// 2*ef covers the usual ef + max-degree merge input; resize below still grows
	// both buffers safely if a graph has degree > ef.
	candidates.reserve(ef);
	keep.reserve(ef);
	while(unvisited_frontier_size>0)
	{
		const conn &curr = unvisited_frontier[0];
		visited.insert(
			std::upper_bound(visited.begin(),visited.end(),curr),
			curr
		);
		num_visited++;
		cnt_eval++;

		keep.clear();
		util::iter_each(f_nbhs(curr.u), [&](nid_t v){
			cnt_traversed++;
			if(!is_seen(v)) {
				cnt_visited++;
				detail::try_prefetch(f_dist, v);
				keep.push_back(v);
			}
		});

		candidates.clear();
		for(nid_t v : keep)
		{
			const auto dv = f_dist(v);
			dist_cmps++;
			if(frontier_size<ef || dv<frontier[frontier_size-1].d)
				candidates.push_back({dv,v});
		}
		cm::sort(candidates.begin(), candidates.end());

		size_t max_cap = frontier_size + candidates.size();
		if(new_frontier.size()<max_cap)
			new_frontier.resize(max_cap);
		auto new_frontier_size = std::set_union(
			frontier.begin(), frontier.begin()+frontier_size,
			candidates.begin(), candidates.end(),
			new_frontier.begin()
		) - new_frontier.begin(); // TODO: early stop at size of ef

		// new_frontier_size = std::min<size_t>(ef, new_frontier_size);
		// if a k is given (i.e. k != 0) then trim off entries that have a
		// distance greater than cut * curr-kth-smallest-distance.
		// Only used during query and not during build.
		/*
		if(QP.k > 0 && new_frontier_size > QP.k)
			new_frontier_size = std::upper_bound(
				new_frontier.begin(), new_frontier.begin()+new_frontier_size,
				std::pair{0, QP.cut * new_frontier[QP.k].second}
			) - new_frontier.begin();
		*/
		frontier_size = std::min<size_t>(new_frontier_size,ef);
		frontier.swap(new_frontier);

		size_t num_remains = std::set_difference(
			frontier.begin(), frontier.begin()+frontier_size,
			visited.begin(), visited.end(),
			unvisited_frontier.begin()
		) - unvisited_frontier.begin();

		unvisited_frontier_size = num_remains;
	}
	
	if(ctrl.log_stat)
	{
		ctrl.log_stat->cnt_eval = cnt_eval;
		ctrl.log_stat->cnt_visited = cnt_visited;
		ctrl.log_stat->cnt_traversed = cnt_traversed;
	}
	// convert back to the caller's expected seq<conn> type -- ONE copy per
	// call, not per hop, unlike the internal std::vector locals above.
	return std::make_pair(
		typename cm::seq<conn>(frontier.begin(), frontier.begin()+frontier_size),
		typename cm::seq<conn>(visited.begin(), visited.end())
	);
}

namespace detail{
template<class Seq1, class Seq2, class Comp=std::less<>>
bool is_intersected(const Seq1 &s1, const Seq2 &s2, Comp comp={})
{
	auto first1 = std::ranges::begin(s1);
	auto last1 = std::ranges::end(s1);
	auto first2 = std::ranges::begin(s2);
	auto last2 = std::ranges::end(s2);
	while(first1!=last1 && first2!=last2)
	{
		if (comp(*first1, *first2))
			++first1;
		else
		{
			if (!comp(*first2, *first1))
				return true;
			++first2;
		}
	}
	return false;
}
}

template<class L = lookup_custom_tag<>, class E, class D, class G, class Seq>
auto beamSearch3(E &&f_nbhs, D &&f_dist, G &&f_label, const Seq &eps, uint32_t ef,
								const search_control &ctrl = {}) {
	using cm = custom<typename L::type>;
	using nid_t = std::ranges::range_value_t<Seq>;
	using conn = util::conn<nid_t>;

	const auto nid_invalid = std::numeric_limits<nid_t>::max();
	const uint32_t bits = ef > 2 ? std::ceil(std::log2(ef)) * 2 - 2 : 2;
	const uint32_t mask = (1u << bits) - 1;
	typename cm::seq<nid_t> visited(mask + 1, nid_invalid);
	uint32_t cnt_visited = 0;
	typename cm::seq<conn> workset;
	std::set<conn> cand;               // TODO: test dual heaps
	std::unordered_set<nid_t> is_inw;  // TODO: test merge instead
	// TODO: get statistics about the merged size
	// TODO: switch to the alternative if exceeding a threshold
	workset.reserve(ef + 1);
	const auto &F = f_label();

	for(nid_t pe : eps) {
		const auto h_pe = cm::hash64(pe) & mask;
		if (visited[h_pe]==pe) continue;
		// P: base, F: query
		const auto &P = f_label(pe);
		// if ((ctrl.filtered && !ctrl.searching) || inter.size() > 0) {
		if (detail::is_intersected(P,F)) {
			visited[h_pe] = pe;
			const auto d = f_dist(pe);
			cand.insert({d, pe});
			workset.push_back({d, pe});
			is_inw.insert(pe);
		}
	}
	if (workset.size() == 0) {
		return workset;
	}
	std::make_heap(workset.begin(), workset.end());
	uint32_t cnt_eval = 0;
	uint32_t limit_eval = ctrl.limit_eval.value_or(std::numeric_limits<uint32_t>::max());

	while (cand.size() > 0) {
		// if (cand.begin()->d > workset[0].d * ctrl.beta) break;

		if (++cnt_eval > limit_eval) break;

		nid_t u = cand.begin()->u;
		cand.erase(cand.begin());

		util::iter_each(f_nbhs(u), [&](nid_t pv) {
			const auto h_pv = cm::hash64(pv) & mask;
			if (visited[h_pv] == pv) return;
			visited[h_pv] = pv;
			cnt_visited++;

			const auto d = f_dist(pv);
			if (!(workset.size() < ef || d < workset[0].d)) return;
			if (!is_inw.insert(pv).second) return;

			const auto &P = f_label(pv);
			// if ((ctrl.filtered && !ctrl.searching) || inter.size() > 0) {
			if (detail::is_intersected(P,F)) {
				cand.insert({d, pv});
				workset.push_back({d, pv});
				std::push_heap(workset.begin(), workset.end());
				if (workset.size() > ef) {
					std::pop_heap(workset.begin(), workset.end());
					is_inw.erase(workset.back().u);
					workset.pop_back();
				}
				if (cand.size() > ef) {
					cand.erase(std::prev(cand.end()));
				}
			}
		});
	}

	// if (ctrl.log_per_stat) {
	//   const auto qid = *ctrl.log_per_stat;
	//   per_visited[qid] += cnt_visited;
	//   per_eval[qid] += cand.size() + cnt_eval;
	//   per_size_C[qid] += cnt_eval;
	// }

	return workset;
}

namespace detail {

	template<class T>
	struct second_elem {
		static auto helper(T &&t) {
			auto [_, y] = t;
			return y;
		}
		using type = decltype(helper(std::declval<T>()));
	};

	template<class T>
	using second_elem_t = typename second_elem<T>::type;

}  // namespace detail

template<class L = lookup_custom_tag<>, class Seq>
Seq prune_simple(Seq cand, uint32_t size, const prune_control &ctrl = {}) {
	(void)ctrl;
	using nid_t = detail::second_elem_t<std::ranges::range_value_t<Seq>>;
	using conn = util::conn<nid_t>;
	static_assert(std::is_same_v<std::ranges::range_value_t<Seq>, conn>);

	if (cand.size() > size) {
		std::nth_element(cand.begin(), cand.begin() + size, cand.end());
		cand.resize(size);
	}
	return cand;
}

template<class L=lookup_custom_tag<>, class Seq, class E, class D>
auto/*Seq*/ prune_heuristic(
	Seq cand, uint32_t size, E &&f_nbhs, D &&f_dist, const prune_control &ctrl = {})
{
	using cm = custom<typename L::type>;
	using nid_t = detail::second_elem_t<std::ranges::range_value_t<Seq>>;
	using conn = util::conn<nid_t>;
	static_assert(std::is_same_v<std::ranges::range_value_t<Seq>, conn>);

	/*std::unordered_set<nid_t> nids;
	for(const auto &c : cand)
		assert(nids.insert(c.u).second);*/
	cm::sort(cand.begin(), cand.end());

	Seq res, pruned;
	std::unordered_set<nid_t> nbh;
	res.reserve(size);

	for(float alpha=ctrl.alpha; alpha>ctrl.alpha_min; alpha*=ctrl.alpha_exp)
	{
		for(const conn &c : cand)
		{
			const auto d_cu = c.d*alpha;

			bool is_pruned = false;
			for(const conn &r : res)
			{
				const auto d_cr = f_dist(c.u, r.u);
				if(d_cr<d_cu)
				{
					is_pruned = true;
					break;
				}
			}

			if(!is_pruned && ctrl.prune_nbh)
				is_pruned = nbh.find(c.u)!=nbh.end();

			if(!is_pruned)
			{
				if(ctrl.prune_nbh)
					util::iter_each(f_nbhs(c.u), [&](nid_t pv){nbh.insert(pv);});
				res.push_back(c);
				if(res.size()==size) break;
			}
			else pruned.push_back(c);
		}
	}

	if(ctrl.recycle_pruned)
	{
		size_t cnt_recycle = std::min(pruned.size(), size-res.size());
		auto split = pruned.begin() + cnt_recycle;
		std::nth_element(pruned.begin(), split, pruned.end());
		res.insert(
			res.end(),
			std::make_move_iterator(pruned.begin()),
			std::make_move_iterator(split)
		);
	}

	return res;
}

template<class L=lookup_custom_tag<>, class Seq, class D, class F>
auto /*Seq*/ prune_heuristic_filtered(
		Seq cand, uint32_t size, D &&f_dist, F &&f_label, const prune_control &ctrl={})
{
	using cm = custom<typename L::type>;
	using nid_t = detail::second_elem_t<std::ranges::range_value_t<Seq>>;
	using conn = util::conn<nid_t>;
	static_assert(std::is_same_v<std::ranges::range_value_t<Seq>, conn>);
	auto alpha = ctrl.alpha;

	// const auto &lb_p = f_label();

	cm::sort(cand.begin(), cand.end());

	Seq res, recycle;
	uint32_t retry = 0;
	while(res.size() < size && retry++<1)
	{
		for(const conn &c : cand)
		{
			const auto d_cu = c.d*alpha;
			const auto &lb_pp = f_label(c.u);

			bool is_pruned = false;
			for(const conn &r : res)
			{
				/*
				using label_t = std::ranges::range_value_t<std::decay_t<decltype(lb_p)>>;
				typename cm::seq<label_t> inter;
				inter.reserve(std::max(lb_pp.size(),lb_p.size()));
				std::set_intersection(
					lb_pp.begin(), lb_pp.end(),
					lb_p.begin(), lb_p.end(),
					std::back_inserter(inter)
				);
				const auto &lb_ps = f_label(r.u);
				if(!std::ranges::includes(lb_ps,inter)) continue;
				*/
				const auto &lb_ps = f_label(r.u);
				if(detail::is_intersected(lb_pp, lb_ps)) continue;

				const auto d_cr = f_dist(c.u,r.u);
				if(d_cr<d_cu)
				{
					is_pruned = true;
					break;
				}
			}

			if(!is_pruned)
			{
				res.push_back(c);
				if(res.size() == size) break;
			}
			else recycle.push_back(c);
		}
		alpha *= 0.85;
		cand = std::move(recycle);
		recycle.clear();
	}
	return res;
}

template<class L=lookup_custom_tag<>, class Seq, class E, class D>
auto/*Seq*/ occlude_list(
	Seq cand, uint32_t size, E &&f_nbhs, D &&f_dist, const prune_control &ctrl={})
{
	using cm = custom<typename L::type>;
	using nid_t = detail::second_elem_t<std::ranges::range_value_t<Seq>>;
	using conn = util::conn<nid_t>;
	static_assert(std::is_same_v<std::ranges::range_value_t<Seq>, conn>);
	(void)f_nbhs;

	if(cand.size()<size)
		return cand;

	cm::sort(cand.begin(), cand.end());

	Seq res;
	// initialize to zero so the occlusion test starts from a clean state
	typename cm::seq<float> occlude_factor(cand.size(), 0.0f);
	res.reserve(size);

	float alpha = ctrl.alpha;
	// float alpha = 1;
	while(alpha>=ctrl.alpha_min && alpha>0 && res.size()<size)
	{
		for(size_t i=0; i<cand.size() && res.size()<size; ++i)
		{
			if(occlude_factor[i] > 1.0f/alpha)
				continue;

			occlude_factor[i] = std::numeric_limits<float>::max();
			res.push_back(cand[i]);

			for(size_t j=i+1; j<cand.size(); ++j)
			{
				if(occlude_factor[j] > 1.0f/ctrl.alpha)
					continue;

				float djk = f_dist(cand[j].u, cand[i].u);
				occlude_factor[j] = (djk == 0) ? 
					std::numeric_limits<float>::max() :
					std::max(occlude_factor[j], cand[j].d/djk);
			}
		}
		alpha *= ctrl.alpha_exp;
		// alpha *= 1/1.2;
	}
	return res;
}


}  // namespace algo
}  // namespace ANN

#endif  // _ANN_ALGO_HPP
