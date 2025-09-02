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
#include "util/intrin.hpp"
#include "util/seq.hpp"
#include "dist.hpp"
#include "parlay.hpp"
#include "benchUtils.h"
#include "utils.hpp"

namespace ANN::external{

auto def_custom_tag()
{
	return custom_tag_parlay{};
}

} // namespace ANN::external

template<typename T>
point_converter_default<T> to_point;

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
	const size_t size_init = parameter.getOptionLongValue("-init", 0);
	const size_t size_step = parameter.getOptionLongValue("-step", 0);
	size_t size_max = parameter.getOptionLongValue("-max", 0);
	const char* file_query = parameter.getOptionValue("-q");
	const char* gt_folder = parameter.getOptionValue("-gtf");
	const uint32_t k = parameter.getOptionIntValue("-k", 10);
	
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

	puts(GREEN "Start to generate groundtruth" RESET);

	std::vector<size_t> sizes;
	for(size_t size_curr=size_init;
		size_curr<=size_max;
		size_curr+=size_step)
	{
		sizes.push_back(size_curr);
	}
	//std::sort(sizes.begin(), sizes.end(), std::greater<size_t>{}); // [REV]
	std::sort(sizes.begin(), sizes.end(), std::less<size_t>{});
	// printf("Cutting range expands from [0,%lu) to [0,%lu)\n", size_last, size_curr);

	//size_t last = ps.size(); // [REV]
	size_t last = 0;
	using pid_t = typename U::point_t::id_t;
	std::vector<std::priority_queue<std::pair<float,pid_t>>> cands;
	for(size_t curr : sizes)
	{
		using std::views::take;
		using std::views::drop;

		// auto r = ps | drop(curr) | take(last-curr); // [REV]
		auto r = ps | drop(last) | take(curr-last);
		printf("last: %lu, curr: %lu, size: %lu\n", last, curr, r.size());

		parlay::internal::timer t("run_test:gen_gt", true);
		auto res = ConstructKnng<U>(r, q, dim, k, cands);
		t.next("Finish search");

		std::ostringstream ss(gt_folder, std::ios_base::ate);
		//ss << "last_" << curr << ".ibin"; [REV]
		ss << "first_" << curr << ".ibin";
		printf("Writing to %s ...\n", ss.str().c_str());
		write_ibin(ss.str(), res);
		t.next("Finish writing to the file");

		last = curr;
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
		"-init <init_size> -step <step_size> -max <max_size>"
		"-gtf <gt folder>"
		"-k <recall@k> -ef <ef_query> [-beta <beta>,...]"
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
