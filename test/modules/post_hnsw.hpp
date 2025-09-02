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

template<class G, class S1, class S2, class GT>
void run_post_hnsw(G layers, uint32_t k, uint32_t ef, const S1 &q, const S2 &F_b, const S2 &F_q,
                   const GT &gt) {
  puts("Search for neighbors and do post processing");
  auto res = post_processing(layers, q, k, ef, F_b, F_q);

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------\n");
}