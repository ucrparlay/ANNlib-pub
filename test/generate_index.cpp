#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <iterator>
#include <ranges>
#include <vector>
#include <stdexcept>
#include <parlay/primitives.h>
#include <parlay/parallel.h>
#include "graph/adj.hpp"
#include "algo/vamana.hpp"
#include "util/intrin.hpp"
#include "dist.hpp"
#include "parlay.hpp"
#include "benchUtils.h"
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
	size_t cnt_points = parameter.getOptionLongValue("-n", 0);
	const uint32_t m = parameter.getOptionIntValue("-m", 40);
	const uint32_t efc = parameter.getOptionIntValue("-efc", 60);
	const float alpha = parameter.getOptionDoubleValue("-alpha", 1);
	// const float batch_base = parameter.getOptionDoubleValue("-b", 2);
	const char *file_out = parameter.getOptionValue("-out");
	if(!file_out)
		puts(RED "file_out is not specified" RESET);
	
	parlay::internal::timer t("run_test:prepare", true);

	using T = typename U::point_t::elem_t;
	auto [ps,dim] = load_point(file_in, to_point<T>, cnt_points);
	t.next("Load the base set");
	printf("%s: [%lu,%u]\n", file_in, ps.size(), dim);

	if(ps.size()<cnt_points)
	{
		cnt_points = ps.size();
		printf("cnt_points is corrected to %lu\n", cnt_points);
	}

	visit_point(ps, cnt_points, dim);
	t.next("Prefetch vectors");

	puts("Start to Build vamana");
	vamana<U> g(dim, m, efc, alpha);
	g.insert(ps.begin(), ps.end());
	t.next("Built index");

	print_stat(g);
	t.next("Print statistics");

	if(file_out)
	{
		printf(GREEN "Saving to %s\n" RESET, file_out);
		g.save(file_out);
		t.next("Index saved");
	}

	for(uint32_t id : {1,5,6,8,12,26})
	{
		uint32_t nid = id - 1;
		printf("[%u]", nid);
		const auto nbhs = g.get_nbhs(nid);
		for(auto nbh : nbhs)
			printf(" %u", nbh.u);
		putchar('\n');
	}
}

int main(int argc, char **argv)
{
	for(int i=0; i<argc; ++i)
		printf("%s ", argv[i]);
	putchar('\n');

	commandLine parameter(argc, argv, 
		"-type <elemType> -dist <distance> "
		"-ml <m_l> -m <m> -efc <ef_construction> -alpha <alpha> "
		"-in <baseset> -n <num_points> "
		"-out <index_path>"
	);

	const char *dist_func = parameter.getOptionValue("-dist");
	auto run_test_helper = [&](auto type){ // emulate a generic lambda in C++20
		using T = decltype(type);
		if(!strcmp(dist_func,"L2"))
			run_test<desc<descr_l2<T>>>(parameter);

		else if(!strcmp(dist_func,"angular"))
			run_test<desc<descr_ang<T>>>(parameter);/*
		else if(!strcmp(dist_func,"ndot"))
			run_test<desc<descr_ndot<T>>>(parameter);*/

		else throw std::invalid_argument("Unsupported distance type");
	};

	const char* type = parameter.getOptionValue("-type");
	/*if(!strcmp(type,"uint8"))
		run_test_helper(uint8_t{});
	else if(!strcmp(type,"int8"))
		run_test_helper(int8_t{});
	else */if(!strcmp(type,"float"))
		run_test_helper(float{});

	else throw std::invalid_argument("Unsupported element type");
	return 0;
}
