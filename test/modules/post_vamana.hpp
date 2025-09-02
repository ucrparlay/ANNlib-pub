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
// #include "algo/vamana.hpp"
#include "graph/adj.hpp"
#include "util/intrin.hpp"

using ANN::vamana;

template<class G, class S1, class S2, class GT>
void run_post_vamana(G g, uint32_t k, uint32_t ef, const S1 &q, const S2 &F_b, const S2 &F_q,
                     const GT &gt) {
  puts("Search for neighbors and do post processing");
  auto res = post_processing(g, q, k, ef, F_b, F_q);

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------\n");
}