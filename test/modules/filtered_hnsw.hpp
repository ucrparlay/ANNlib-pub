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
// #include "algo/algo.hpp"
// #include "algo/hnsw_filtered.hpp"
#include "graph/adj.hpp"
#include "util/intrin.hpp"

using ANN::filtered_hnsw;

template<class U, class S1, class S2, class S3>
auto run_filtered_hnsw_insert(uint32_t dim, float m_l, uint32_t m, uint32_t efc, float alpha,
                              float batch_base, size_t size_init, size_t size_step, size_t size_max,
                              const S1& ps, const S2& F_b, const S3& medoid) {
  // using nid_t = typename filtered_hnsw<U>::nid_t;
  // using pid_t = typename filtered_hnsw<U>::pid_t;
  // using label_t = typename filtered_hnsw<U>::label_t;

  filtered_hnsw<U> layers(dim, m_l, m, efc, alpha);
  puts("Initialize Filtered HNSW");

  parlay::internal::timer t;

  // for (size_t size_last = 0, size_curr = size_init; size_curr <= size_max;
  //      size_last = size_curr, size_curr += size_step) {
  //   auto ins_begin = ps.begin() + size_last;
  //   auto ins_end = ps.begin() + size_curr;

  //   const auto& insert_labels =
  //       ANN::util::to<S2>(std::ranges::subrange(F_b.begin() + size_last, F_b.begin() + size_curr));

  //   // layers.entrance.clear();
  //   // layers.entrance = ANN::util::to<decltype(layers.entrance)>(
  //   //     std::ranges::subrange(medoid.begin(), medoid.end()));
  //   layers.insert(ins_begin, ins_end, insert_labels, batch_base);
  // }

  layers.insert(ps.begin(), ps.end(), F_b, medoid, batch_base);

  puts("Collect statistics");
  print_layers(layers);

  t.next("Finish insertion");

  return layers;
}

template<class G, class E, class S1, class S2, class S3, class GT>
void run_filtered_hnsw_search(G layers, const E& medoid, uint32_t k, uint32_t ef, const S1& q,
                              const S2& F_q, const S3& P_b, const GT& gt) {
  parlay::internal::timer t;
  puts("Search for neighbors");
  // layers.entrance.clear();
  // auto medoid = layers.template find_medoid(P_b, 0.5);
  // layers.entrance =
  //     ANN::util::to<decltype(layers.entrance)>(std::ranges::subrange(medoid.begin(), medoid.end()));
  auto res = find_nbhs(layers, q, k, ef, F_q /*, true*/);
  t.next("Finish searching");

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------\n");
}