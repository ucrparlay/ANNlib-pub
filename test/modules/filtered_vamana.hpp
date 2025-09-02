#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../utils.hpp"
#include "algo/algo.hpp"
#include "algo/vamana_filtered.hpp"
#include "graph/adj.hpp"
#include "util/intrin.hpp"
#include "util/seq.hpp"

using ANN::filtered_vamana;

template<class U, class S1, class S2>
auto run_filtered_vamana_insert(uint32_t dim, uint32_t m, uint32_t efc, float alpha,
                                float batch_base, size_t size_init, size_t size_step,
                                size_t size_max, const S1 &ps_reg, const S2 &F_b) {
  // using nid_t = typename filtered_vamana<U>::nid_t;
  // using pid_t = typename filtered_vamana<U>::pid_t;
  // using label_t = typename filtered_vamana<U>::label_t;
  // constexpr bool filtered = true;

  puts("Initialize Filtered Vamana");
  filtered_vamana<U> g(dim, m, efc, alpha);

  struct mixer{
    const S1 &ps;
    const S2 &fl;
    size_t i;

    decltype(auto) get_id() const{
        return ps[i].get_id();
    }
    decltype(auto) get_coord() const{
        return ps[i].get_coord();
    }
    decltype(auto) get_label() const{
        return fl[i];
    }
  };
  auto ps = ANN::util::delayed_seq(ps_reg.size(), [&](size_t i){
    return mixer{ps_reg, F_b, i};
  });

  parlay::internal::timer t;

  for (size_t size_last = 0, size_curr = size_init; size_curr <= size_max;
       size_last = size_curr, size_curr += size_step) {
    auto ins_begin = ps.begin() + size_last;
    auto ins_end = ps.begin() + size_curr;

    const auto &insert_labels =
        ANN::util::to<S2>(std::ranges::subrange(F_b.begin() + size_last, F_b.begin() + size_curr));

    g.insert(ins_begin, ins_end, batch_base);
  }

  // parlay::sequence<bool> used(ps.size(), false);
  // // parlay::sequence<size_t> prefix_array(ps.size(), 0);

  // std::set<nid_t> s(entrance.first.begin(), entrance.first.end());
  // decltype(entrance.first) medoid(s.begin(), s.end());
  // std::cout << "Entrance point size: " << medoid.size() << std::endl;

  // const size_t N = medoid.size();
  // const size_t M = ps.size() - medoid.size();

  // // TODO: entrance points insertion
  // S1 eps(N);
  // S2 l_eps(N);

  // std::cout << "Assign entrance points and their labels" << std::endl;
  // parlay::parallel_for(0, N, [&](size_t i) {
  //   auto offset = medoid[i];
  //   assert(offset < ps.size());
  //   eps[i] = ps[offset];
  //   l_eps[i] = F_b[offset];
  //   used[offset] = true;
  //   // prefix_array[i] = 1;
  // });

  // // insert entrance points
  // std::cout << "Insert entrance points to index" << std::endl;
  // g.insert(eps.begin(), eps.end(), l_eps, batch_base /*, filtered*/);
  // // g.insert(ps.begin(), ps.end(), F_b, batch_base /*, filtered*/);

  // // calculate prefix sum
  // // auto prefix_sum = parlay::scan_inclusive(prefix_array);

  // // TODO: remaining points insertion
  // S1 rem(M);
  // S2 l_rem(M);

  // // remaining point indices
  // auto remaining_indices = parlay::filter(parlay::iota(M), [&](size_t i) {
  //   // return prefix_array[i] == 0;
  //   return !used[i];
  // });

  // std::cout << "Assign remaining points and their labels" << std::endl;
  // parlay::parallel_for(0, M, [&](size_t i) {
  //   size_t j = remaining_indices[i];
  //   rem[i] = ps[j];
  //   l_rem[i] = F_b[j];
  // });

  // for (size_t size_last = 0, size_curr = size_init; size_curr < rem.size(); size_last =
  // size_curr, size_curr += size_step) {
  //   auto ins_begin = rem.begin() + size_last;
  //   auto ins_end = rem.begin() + size_curr;
  //   const auto &insert_labels =
  //       ANN::util::to<S2>(std::ranges::subrange(l_rem.begin() + size_last, l_rem.begin() +
  //       size_curr));

  //   // insert remaining points
  //   // std::cout << "Inserting remaining points to the index..." << std::endl;
  //   g.insert2(ins_begin, ins_end, insert_labels, entrance.second, batch_base /*, filtered*/);
  // }

  // insert remaining points
  // std::cout << "Insert remaining points to index" << std::endl;
  // g.insert2(rem.begin(), rem.end(), l_rem, entrance.second, batch_base/*, filtered*/);
  // g.insert(rem.begin(), rem.end(), l_rem, batch_base/*, filtered*/);

  puts("Collect statistics");
  print_stat(g);

  t.next("Finish insertion");

  return g;
}

template<class G, class S1, class S2, class GT>
void run_filtered_vamana_search(const G &g, uint32_t k, uint32_t ef, const S1 &q, const S2 &F_q, const GT &gt, uint32_t rounds = 5) {
  // constexpr bool filtered = true;
  parlay::internal::timer t;

  puts("Search for neighbors");
  // g.entrance.clear();
  // auto medoid = g.template find_medoid(P_b, 0.2);
  // g.entrance =
  //     ANN::util::to<decltype(g.entrance)>(std::ranges::subrange(medoid.begin(), medoid.end()));
  auto res = find_nbhs(g, q, k, ef, F_q, rounds /*, filtered*/);
  t.next("Finish searching");

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------\n");
}
