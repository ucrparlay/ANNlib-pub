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
// #include "algo/HNSW.hpp"
// #include "algo/algo.hpp"
#include "graph/adj.hpp"
#include "util/intrin.hpp"

using ANN::HNSW;

template<class U, class Seq>
auto run_hnsw_insert(uint32_t dim, float m_l, uint32_t m, uint32_t efc, float alpha,
                     float batch_base, size_t size_init, size_t size_step, size_t size_max,
                     const Seq &ps) {
  HNSW<U> layers(dim, m_l, m, efc, alpha);
  puts("Initialize HNSW");
  parlay::internal::timer t;

  for (size_t size_last = 0, size_curr = size_init; size_curr <= size_max;
       size_last = size_curr, size_curr += size_step) {
    auto ins_begin = ps.begin() + size_last;
    auto ins_end = ps.begin() + size_curr;
    layers.insert(ins_begin, ins_end, batch_base);
  }

  t.next("Finish insertion");

  puts("Collect statistics");
  print_layers(layers);

  return layers;
}

template<class G, class Seq, class GT>
void run_hnsw_search(const G &layers, uint32_t k, uint32_t ef, const Seq &q, const GT &gt) {
  puts("Search for neighbors");
  auto res = find_nbhs(layers, q, k, ef);

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------\n");
}