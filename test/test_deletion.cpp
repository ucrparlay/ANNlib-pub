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
#include "algo/vamana.hpp"
#include "util/intrin.hpp"
#include "util/seq.hpp"
#include "dist.hpp"
#include "parlay.hpp"
#include "benchUtils.h"
#include "cpam.hpp"
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

template<typename U>
void run_test(commandLine parameter) // intend to be pass-by-value manner
{
	const char *file_in = parameter.getOptionValue("-in");
	const char *idx_path = parameter.getOptionValue("-index");
	const size_t size_init = parameter.getOptionLongValue("-init", 0);
	const size_t size_step = parameter.getOptionLongValue("-step", 0);
	size_t size_max = parameter.getOptionLongValue("-max", 0);
	const char* file_query = parameter.getOptionValue("-q");
	const uint32_t k = parameter.getOptionIntValue("-k", 10);
	const auto efs = parse_array(parameter.getOptionValue("-efs"), atoi);
	const char* gt_folder = parameter.getOptionValue("-gtf");
	
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

	auto [q,_] = load_point(file_query, to_point<T>);
	t.next("Load queries");
	printf("%s: [%lu,%u]\n", file_query, q.size(), _);

	visit_point(ps, size_max, dim);
	visit_point(q, q.size(), dim);
	t.next("Prefetch vectors");

	printf(GREEN "Loading index from %s\n" RESET, idx_path);

	vamana<U> base(idx_path, [&](auto i){
		return ps[i].get_coord();
	});
	// vamana<U> base(128, 32, 50, 0.85);
	// base.insert(ps.begin(), ps.end());
	puts("Initialize vamana");

	for(size_t size_last=0, size_curr=size_init;
		size_curr<size_max;
		size_last=size_curr, size_curr+=size_step)
	{
		printf("Cutting range expands from [0,%lu) to [0,%lu)\n", size_last, size_curr);

		puts("Copy base index");
		parlay::internal::timer t("run_test:deletion", true);
		auto g = base;
		t.next("Finish copying");

		g.print_stat();
		t.next("Collect statistics");

		auto pids = ANN::util::init<parlay::sequence<uint32_t>>(
			size_curr, [&](size_t i){return ps[i].get_id();}
		);
		g.erase(pids.begin(), pids.end());
		t.next("Finish deletion");

		puts("Generate groundtruth");
		// auto baseset = ps | std::views::drop(size_curr);
		// auto gt = ConstructKnng<U>(baseset, q, dim, k);
		std::stringstream path_gt;
		path_gt << gt_folder << "last_" << size_curr << ".ibin:ibin";
		fprintf(stderr, path_gt.str().c_str());
		auto [gt,maxk] = load_point(path_gt.str().c_str(), gt_converter<uint32_t>{});
		t.next("Finish gt generation");
		
		for(auto ef : efs)
		{
			puts("Search for neighbors");
			auto res = find_nbhs(g, q, k, ef);
			puts("Compute recall");
			calc_recall_tie<U>(q, res, gt, k, ps, dim);
		}

		g.consolidate();
		t.next("Finish consolidation");

		for(auto ef : efs)
		{
			puts("Search for neighbors");
			auto res = find_nbhs(g, q, k, ef);
			puts("Compute recall");
			calc_recall_tie<U>(q, res, gt, k, ps, dim);
		}
		puts("---");
	}
}

int main(int argc, char **argv)
{
	for(int i=0; i<argc; ++i)
		printf("%s ", argv[i]);
	putchar('\n');

	commandLine parameter(argc, argv, 
		"-type <elemType> -dist <distance>"
		"-in <baseset> -q <queries> "
		"-index <index_path>"
		"-init <init_size> -step <step_size> -max <max_size>"
		"-k <recall@k> -ef <ef_query> [-beta <beta>,...]"
		"-gtf <gt folder>"
	);

	const char *dist_func = parameter.getOptionValue("-dist");
	auto run_test_helper = [&](auto type){ // emulate a generic lambda in C++20
		using T = decltype(type);
		if(!strcmp(dist_func,"L2"))
			run_test<desc<descr_l2<T>>>(parameter);
		
		else if(!strcmp(dist_func,"angular"))
			run_test<desc<descr_ang<T>>>(parameter);
		else if(!strcmp(dist_func,"ndot"))
			run_test<desc<descr_ndot<T>>>(parameter);
		
		else throw std::invalid_argument("Unsupported distance type");
	};

	const char* type = parameter.getOptionValue("-type");
	if(!strcmp(type,"uint8"))
		run_test_helper(uint8_t{});
	else if(!strcmp(type,"int8"))
		run_test_helper(int8_t{});
	else if(!strcmp(type,"float"))
		run_test_helper(float{});

	else throw std::invalid_argument("Unsupported element type");
	return 0;
}
