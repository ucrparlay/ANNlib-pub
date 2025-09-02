#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <string>
#include <random>
#include <ranges>
#include <set>
#include "custom/undef.hpp"
#include "algo/algo.hpp"

int main(int argc, char **argv)
{
	using namespace ANN;

	// float range = 5;
	// float alpha = 0.82;
	float range = atof(argv[1]);
	float alpha = atof(argv[2]);
	// scanf("%f", &alpha);
	algo::prune_control ctrl{.alpha=alpha};

	std::mt19937 gen(std::random_device{});
	std::uniform_real_distribution<> dist(-range, range);

	std::vector<float> coord = {0};
	for(int i=1; i<20; ++i)
		coord.push_back(dist(gen));

	std::vector<util::conn<uint32_t>> conn;
	for(int i=1; i<20; ++i)
		conn.push_back({coord[i]*coord[i], uint32_t(i)});
	std::sort(conn.begin(), conn.end());

	auto f_nbhs = [](...){return std::views::empty<uint32_t>;};
	auto f_dist = [&](uint32_t u, uint32_t v){
		auto d = coord[u] - coord[v];
		return d*d;
	};
	auto nbh_ph = algo::prune_heuristic(conn, 10, f_nbhs, f_dist, ctrl);
	auto nbh_ol = algo::occlude_list(conn, 10, f_nbhs, f_dist, ctrl);

	puts("cands");
	for(const auto &e : conn)
		printf("[%u]\t%.4f\n", e.u, coord[e.u]);
	putchar('\n');
	puts("ph");
	for(const auto &e : nbh_ph)
		printf("[%u]\t%.4f\n", e.u, coord[e.u]);
	putchar('\n');
	puts("ol");
	for(const auto &e : nbh_ol)
		printf("[%u]\t%.4f\n", e.u, coord[e.u]);
	return 0;
}
