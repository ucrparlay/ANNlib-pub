#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../utils.hpp"
#include "algo/algo.hpp"
#include "algo/vamana.hpp"
#include "graph/adj.hpp"
#include "util/intrin.hpp"

using ANN::vamana;

template<class U, class Seq>
auto run_vamana_insert(uint32_t dim, uint32_t r, uint32_t efc, float alpha, float batch_base,
                       size_t size_init, size_t size_step, size_t size_max, const Seq &ps) {
  puts("Initialize Vamana");
  vamana<U> g(dim, r, efc, alpha);

  parlay::internal::timer t;

  for (size_t size_last = 0, size_curr = size_init; size_curr <= size_max;
       size_last = size_curr, size_curr += size_step) {
    auto ins_begin = ps.begin() + size_last;
    auto ins_end = ps.begin() + size_curr;
    g.insert(ins_begin, ins_end, batch_base);
  }
  t.next("Finish insertion");

  puts("Collect statistics");
  print_stat(g);

  return g;
}

template<class G, class Seq, class GT>
void run_vamana_search(const G &g, uint32_t k, uint32_t ef, const Seq &q, const GT &gt) {
  puts("Search for neighbors");
  auto res = find_nbhs(g, q, k, ef);

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------\n");
}