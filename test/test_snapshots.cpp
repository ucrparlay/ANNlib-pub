#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <sstream>
#include <iterator>
#include <ranges>
#include <vector>
#include <queue>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <stdexcept>
#include <parlay/primitives.h>
#include <parlay/parallel.h>
#include "graph/adj.hpp"
#include "graph/chrono.hpp"
#include "algo/vamana.hpp"
#include "util/intrin.hpp"
#include "util/shared.hpp"
#include "dist.hpp"
#include "parlay.hpp"
#include "benchUtils.h"
#include "cpam.hpp"
#include "get_mem_usage.h"
#include "utils.hpp"
using ANN::vamana;

parlay::sequence<size_t> per_visited;
parlay::sequence<size_t> per_eval;
parlay::sequence<size_t> per_size_C;

namespace ANN::external{

auto def_custom_tag()
{
	return custom_tag_parlay{};
}

} // namespace ANN::external

template<typename T>
point_converter_default<T> to_point;

template<typename T>
class gt_converter{
public:
	using type = parlay::sequence<T>;
	template<typename Iter>
	type operator()([[maybe_unused]] uint32_t id, Iter begin, Iter end)
	{
		using type_src = typename std::iterator_traits<Iter>::value_type;
		static_assert(std::is_convertible_v<type_src,T>, "Cannot convert to the target type");

		const uint32_t n = std::distance(begin, end);

		// T *gt = new T[n];
		auto gt = parlay::sequence<T>(n);
		for(uint32_t i=0; i<n; ++i)
			gt[i] = *(begin+i);
		return gt;
	}
};

template<class DescLegacy>
struct desc{
	using point_t = point<typename DescLegacy::type_elem>;
	using coord_t = typename point_t::coord_t;
	using dist_t = float;
	static dist_t distance(const coord_t &cu, const coord_t &cv, uint32_t dim){
		return DescLegacy::distance(cu, cv, dim);
	}

	template<typename Nid, class Ext, class Edge>
	using graph_t = ANN::graph::adj_seq<Nid,Ext,Edge>;

	template<typename Nid, class Ext, class Edge>
	using graph_aux = ANN::graph::adj_map<Nid,Ext,Edge>;
};

template<class DescLegacy>
struct desc_chrono_array: desc<DescLegacy>{
	template<typename Nid, class Ext, class Edge=Nid>
	using graph_t = ANN::graph::chrono_list<Nid,Ext,Edge>;

	template<typename Nid, class Ext, class Edge=Nid>
	using graph_aux = ANN::graph::chrono_list<Nid,Ext,Edge>;
};

template<class DescLegacy>
struct desc_chrono_prefix: desc<DescLegacy>{
	template<typename Nid, class Ext, class Edge=Nid>
	using graph_t = ANN::graph::chrono_list<Nid,Ext,Edge,ANN::util::shared_vector>;

	template<typename Nid, class Ext, class Edge=Nid>
	using graph_aux = ANN::graph::chrono_list<Nid,Ext,Edge,ANN::util::shared_vector>;
};

template<class DescLegacy>
struct desc_cpam: desc<DescLegacy>{
	template<typename Nid, class Ext, class Edge=Nid>
	using graph_t = graph_cpam<Nid,Ext,Edge>;

	template<typename Nid, class Ext, class Edge=Nid>
	using graph_aux = graph_cpam<Nid,Ext,Edge>;
};

template<typename U>
void run_test(commandLine parameter) // intend to be pass-by-value manner
{
	const char *file_in = parameter.getOptionValue("-in");
	const size_t size_init = parameter.getOptionLongValue("-init", 0);
	const size_t size_step = parameter.getOptionLongValue("-step", 0);
	size_t size_max = parameter.getOptionLongValue("-max", 0);
	const uint32_t m = parameter.getOptionIntValue("-m", 40);
	const uint32_t efc = parameter.getOptionIntValue("-efc", 60);
	const float alpha = parameter.getOptionDoubleValue("-alpha", 1);
	const float batch_base = parameter.getOptionDoubleValue("-b", 2);
	// const char* file_query = parameter.getOptionValue("-q");
	// const uint32_t k = parameter.getOptionIntValue("-k", 10);
	// const uint32_t ef = parameter.getOptionIntValue("-ef", m*20);
	// const auto efs = parse_array(parameter.getOptionValue("-efs"), atoi);
	const bool do_snapshot = !!parameter.getOptionIntValue("-snapshot", 0);

	printf("Snapshot %s\n", do_snapshot? "enabled": "disabled");
	
	parlay::internal::timer t("run_test:prepare", true);

	using T = typename U::point_t::elem_t;
	auto [ps,dim] = load_point(file_in, to_point<T>, size_max);
	t.next("Load the base set");
	printf("%s: [%lu,%u]\n", file_in, ps.size(), dim);

	if(ps.size()<size_max)
	{
		size_max = ps.size();
		printf("size_max is corrected to %lu\n", size_max);
	}

	visit_point(ps, size_max, dim);
	t.next("Prefetch vectors");

	std::vector<std::pair<uint32_t,vamana<U>>> snapshots;
	snapshots.reserve(size_max/size_step+2);
	vamana<U> g(dim, m, efc, alpha);
	puts("Initialize vamana");

	for(size_t size_last=0, size_curr=size_init;
		size_curr<=size_max;
		size_last=size_curr, size_curr+=size_step)
	{
		printf("Increasing size from %lu to %lu\n", size_last, size_curr);

		puts("Insert points");
		parlay::internal::timer t("run_test:insert", true);
		auto ins_begin = ps.begin()+size_last;
		auto ins_end = ps.begin()+size_curr;
		g.insert(ins_begin, ins_end, batch_base);
		t.next("Finish insertion");

		if(do_snapshot)
		{
			auto ng = g;
			snapshots.emplace_back(size_curr, std::move(g));
			t.next("Finish snapshotting");
			g = std::move(ng);
		}

		if constexpr(requires{g.print_node(0);})
		{
			g.print_node(2);
			if(g.num_nodes()>32)
				g.print_node(32);
		}

		puts("Collect statistics");
		g.print_stat();

		printf("Current memory usage: %lu bytes\n", getCurrentRSS());
		printf("Peak memory usage: %lu bytes\n", getPeakRSS());

		puts("---");
	}
	snapshots.emplace_back(size_max, std::move(g));

	// print_stat_snapshots(snapshots);
	if constexpr(requires{g.print_node(0);})
		snapshots.back().second.print_node(32);

	using pid_t = typename U::point_t::id_t;
	// std::vector<std::priority_queue<std::pair<float,pid_t>>> cands;
	uint32_t k = 10;
	uint32_t ef = 250;
	auto q = ps | std::views::take(100);
	for(const auto &[sz,g] : snapshots)
	{
		printf("inserted size: %u, actual size: %lu\n", sz, g.num_nodes());
		auto r = ps | std::views::take(sz);
		puts("Search for neighbors");
		auto res = find_nbhs(g, q, k, ef);
		for(const auto &u : res[0])
			printf("%8u", u.pid);
		putchar('\n');
		puts("Computing groundtruth");
		auto gt = ConstructKnng<U>(r, q, dim, k);
		for(pid_t u : gt[0])
			printf("%8u", u);
		putchar('\n');
		puts("Compute recall");
		calc_recall(q, res, gt, k);
	}
}

template<typename U>
void run_test_with_graph(commandLine parameter)
{
	const char *graph_type = parameter.getOptionValue("-graph");
	if(!strcmp(graph_type,"simple"))
		run_test<desc<U>>(parameter);
	else if(!strcmp(graph_type,"pam"))
		run_test<desc_cpam<U>>(parameter);
	else if(!strcmp(graph_type,"chrono_array"))
		run_test<desc_chrono_array<U>>(parameter);
	else if(!strcmp(graph_type,"chrono_prefix"))
		run_test<desc_chrono_prefix<U>>(parameter);
	else throw std::invalid_argument("Unsupported graph type");
}

template<typename T>
void run_test_with_metric(commandLine parameter)
{
	const char *dist_func = parameter.getOptionValue("-dist");
	if(!strcmp(dist_func,"L2"))
		run_test_with_graph<descr_l2<T>>(parameter);
	else if(!strcmp(dist_func,"angular"))
		run_test_with_graph<descr_ang<T>>(parameter);
	// else if(!strcmp(dist_func,"ndot"))
		// run_test_with_graph<descr_ndot<T>>();
	else throw std::invalid_argument("Unsupported distance type");
}

int main(int argc, char **argv)
{
	for(int i=0; i<argc; ++i)
		printf("%s ", argv[i]);
	putchar('\n');

	commandLine parameter(argc, argv, 
		"-type <elemType> -dist <distance>"
		"-ml <m_l> -m <m> -efc <ef_construction> -alpha <alpha> "
		"-in <baseset> -q <queries> "
		"-init <init_size> -step <step_size> -max <max_size>"
		"-k <recall@k> -ef <ef_query> [-beta <beta>,...]"
	);

	const char* type = parameter.getOptionValue("-type");
	if(!strcmp(type,"uint8"))
		run_test_with_metric<uint8_t>(parameter);
	else if(!strcmp(type,"int8"))
		run_test_with_metric<int8_t>(parameter);
	else if(!strcmp(type,"float"))
		run_test_with_metric<float>(parameter);

	else throw std::invalid_argument("Unsupported element type");
	return 0;
}
